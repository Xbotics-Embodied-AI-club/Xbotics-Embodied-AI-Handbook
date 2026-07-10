"""数据集回放验证 + BPU 性能采集（板端运行，不动实物）。

用真实数据集的观测（top/wrist 图 + state）喂经 OE 量化的 BPU ACT 模型，
比对 BPU 预测的 100 步 action chunk 与数据集真值动作是否一致；并分别采集
VisionEncoder / TransformerLayers / 完整 ACT 的纯 BPU 前向耗时（warmup+采样），
以及由 chunk 推出的 30fps 占空比——即性能测评视频脚本所需的那些参数。

推理走 rdk_LeRobot_tools 官方 bpu_control_robot.py 里的 BPUACTPolicy，不另写推理逻辑。
数据集为 LeRobot v3.0（单 parquet + 单 mp4/相机），直接读 parquet(state/action) + mp4(帧)，
绕开 lerobot 0.4.4 读不了 v3.0 的 LeRobotDataset。
"""

import os
import sys
import time

import cv2
import numpy as np
import pyarrow.parquet as pq
import torch

# 路径相对脚本自身目录（S100/S600 同一份脚本通用）。每块板的工作目录里放：
#   bpu_output/（对应板的 .hbm + 归一化 npy）、rdk_tools_s600/、replay_frames/、本脚本
BASE = os.path.dirname(os.path.abspath(__file__))
BPU_ACT_PATH = os.path.join(BASE, "bpu_output")
TOOLS_DIR = os.path.join(BASE, "rdk_tools_s600")
FRAMES_DIR = os.path.join(BASE, "replay_frames")  # 开发机 ffmpeg 解 AV1 预抽的 PNG（板上解不了 AV1）
DATASET = "$DATASETS_ROOT/hf-hub/lerobot/so101/put_black_cuboid_into_basket"
EPISODE = 0
REPLAY_FRAMES = [0, 47, 94, 141, 188]  # episode 0 内的全局帧号（须与开发机预抽的 PNG 对应）
# 模型相机输入 <- 数据集相机名。此 cuboid 模型/bpu_output 与数据集 top/wrist 标注相反
# （实测：交叉映射后 pred 与数据集真值 MAE 16.8°→10°），对应 deploy_guide §3/§6 的 top/wrist 互换告警。
MODEL_CAM_FROM_DATASET = {"top": "wrist", "wrist": "top"}
WARMUP, SAMPLES = 20, 200  # 纯 BPU 前向基准（对齐官方 §8 口径）
FPS = 30

sys.path.insert(0, TOOLS_DIR)
from bpu_control_robot import BPUACTPolicy, detect_cameras_from_model, detect_n_action_steps


def decode_frame(name, global_index):
    bgr = cv2.imread(f"{FRAMES_DIR}/{name}_{global_index}.png")  # 预抽 PNG
    if bgr is None:
        raise RuntimeError(f"missing frame PNG: {name}_{global_index}.png")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def to_batch(state6, frames_rgb):
    batch = {"observation.state": torch.tensor(state6, dtype=torch.float32).unsqueeze(0)}
    for name, rgb in frames_rgb.items():
        t = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        batch[f"observation.images.{name}"] = t
    return batch


camera_names = detect_cameras_from_model(BPU_ACT_PATH)
n_action_steps = detect_n_action_steps(BPU_ACT_PATH, None)
print(f"cameras={camera_names}  n_action_steps={n_action_steps}")

policy = BPUACTPolicy(BPU_ACT_PATH, n_action_steps, camera_names)

df = pq.read_table(f"{DATASET}/data/chunk-000/file-000.parquet").to_pandas()
ep = df[df["episode_index"] == EPISODE].reset_index(drop=True)
actions = np.stack(ep["action"].apply(np.asarray).to_numpy())   # [T,6]
states = np.stack(ep["observation.state"].apply(np.asarray).to_numpy())
gidx = ep["index"].to_numpy()                                   # 全局帧号 = mp4 帧号
T = len(ep)
print(f"episode {EPISODE}: {T} frames")

# —— 一致性：BPU 预测 chunk vs 数据集真值动作 ——
# ACT 闭环里每步重规划，开环跑满 100 步会自然发散；故同时看 1 步/10 步/整段 MAE。
print("\n=== 数据集回放一致性（BPU 预测 vs 数据集真值动作，单位=度，量程约 "
      f"{actions.min():.0f}..{actions.max():.0f}°）===")
sample_t = [t for t in REPLAY_FRAMES if t < T]
mae1, mae10, maefull = [], [], []
joint_abs = []
for t in sample_t:
    frames = {name: decode_frame(MODEL_CAM_FROM_DATASET[name], int(gidx[t])) for name in camera_names}
    policy.reset()
    first = policy.select_action(to_batch(states[t], frames))
    chunk = torch.stack([first] + list(policy._action_queue)).numpy()  # [100,6]
    horizon = min(n_action_steps, T - t)
    gt = actions[t : t + horizon]
    pred = chunk[:horizon]
    e1 = np.abs(pred[0] - gt[0]).mean()
    e10 = np.abs(pred[: min(10, horizon)] - gt[: min(10, horizon)]).mean()
    ef = np.abs(pred - gt).mean()
    mae1.append(e1); mae10.append(e10); maefull.append(ef)
    h10 = min(10, horizon)
    joint_abs.append(np.abs(pred[:h10] - gt[:h10]).mean(axis=0))
    print(f"  frame {t:>4}: 1步MAE={e1:5.2f}°  10步MAE={e10:5.2f}°  整段({horizon})MAE={ef:5.2f}°  "
          f"pred[0]={np.round(pred[0],1)}")
print(f"  >> 平均: 1步 {np.mean(mae1):.2f}°   10步 {np.mean(mae10):.2f}°   整段 {np.mean(maefull):.2f}°")
print("  关节级 1..10 步平均 MAE(度):", np.round(np.mean(joint_abs, axis=0), 2))

t0 = int(sample_t[0])
frames0 = {name: decode_frame(MODEL_CAM_FROM_DATASET[name], int(gidx[t0])) for name in camera_names}
policy.reset()
f0 = policy.select_action(to_batch(states[t0], frames0))
chunk0 = torch.stack([f0] + list(policy._action_queue)).numpy()

# —— 性能：纯 BPU 前向，分别计时 VisionEncoder / TransformerLayers / 完整 ——
print(f"\n=== 纯 BPU 前向基准（{WARMUP} warmup + {SAMPLES} 采样）===")
nbatch = policy._normalize_inputs(to_batch(states[t0], frames0))
img_in = {name: nbatch[f"observation.images.{name}"].numpy().copy() for name in camera_names}
state_in = nbatch["observation.state"].numpy().copy()


def bench(fn):
    for _ in range(WARMUP):
        fn()
    ts = []
    for _ in range(SAMPLES):
        s = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - s) * 1000.0)
    a = np.array(ts)
    return a.mean(), a.std(), 1000.0 / a.mean()


def run_vision(name):
    return next(iter(policy.bpu_policy.run({"images": img_in[name]},
                model_name="BPU_ACTPolicy_VisionEncoder")["BPU_ACTPolicy_VisionEncoder"].values()))


feats = {name: run_vision(name) for name in camera_names}
tf_inputs = {"states": state_in, **{f"{n}_features": feats[n] for n in camera_names}}


def f_vision():
    run_vision(camera_names[0])


def f_transformer():
    policy.bpu_policy.run(tf_inputs, model_name="BPU_ACTPolicy_TransformerLayers")


def f_full():
    fs = {n: run_vision(n) for n in camera_names}
    policy.bpu_policy.run({"states": state_in, **{f"{n}_features": fs[n] for n in camera_names}},
                          model_name="BPU_ACTPolicy_TransformerLayers")


for label, fn in [
    ("VisionEncoder (单相机)", f_vision),
    ("TransformerLayers", f_transformer),
    (f"完整 ACT ({len(camera_names)} 相机视觉 + 1 transformer)", f_full),
]:
    m, s, thr = bench(fn)
    print(f"  {label:<34} {m:7.3f} ms  (±{s:5.3f})  {thr:8.2f} inf/s")
    if label.startswith("完整"):
        chunk_s = n_action_steps / FPS
        duty = m / (chunk_s * 1000.0) * 100.0
        print(f"     -> {n_action_steps} 步 chunk @ {FPS}fps 覆盖 {chunk_s:.2f}s；"
              f"占空比 {duty:.3f}%  空闲 {100-duty:.2f}%")
