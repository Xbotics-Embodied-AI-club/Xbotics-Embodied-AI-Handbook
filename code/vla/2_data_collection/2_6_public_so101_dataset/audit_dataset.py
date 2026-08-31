"""训练前审计合并数据集，四项检查全过才开训。

对应第9讲《操作数据闭环》2.2.2 节——那里讲了绝对跟踪与锚定跟踪两种动作口径，
以及口径没被记下来会怎么坑到部署；下面第一项检查就是把口径量出来。

第一项是重点：逐关节比对 action 与 observation.state。

遥操作录制时如果开了"首帧锚定"（连接瞬间算一次 offset = 当前位姿 − 首个 action，之后
每个目标都加这个 offset），那么记录下来的 action 里就烤进了一个每次上电都不同的常量偏移。
模型会把这个偏移当成策略的一部分学走；部署时若按绝对关节目标直接下发，机械臂就会走到
训练分布里从没出现过的关节区域，然后闭环发散。真机上腕关节就是这么被推飞的。

所以开训前必须先量清楚 action 与 state 的关系，把动作语义（绝对 / 相对锚定）定下来，
并且随 checkpoint 一起记档，部署侧照同一口径配。

判据：逐关节 |mean(action − state)| 与该关节动作幅度之比。全都很小 = 绝对语义；
某个关节冒出显著常量偏移 = 被锚定过，按相对语义部署。
"""

import json
import os
from pathlib import Path

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

MERGED_DIR = Path(os.environ["DATASETS_ROOT"]) / "so101" / "datasets" / "merged_9task"
MERGED_REPO_ID = "so101/pickplace_9task"
REPORT_PATH = Path(os.environ["DATASETS_ROOT"]) / "so101" / "outputs" / "audit.json"

# 偏移占动作幅度的比例超过这个数，就认为该关节存在系统性偏移（不是噪声）
OFFSET_RATIO_THRESHOLD = 0.05

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

# 先确认合并集在盘上。LeRobotDataset 找不到本地 meta/ 时会转去 Hub 拉同名仓库，
# 而这个 repo_id 只存在于本地，于是报一个和真实原因八竿子打不着的 404。
if not (MERGED_DIR / "meta" / "info.json").exists():
    raise SystemExit(f"缺合并集 {MERGED_DIR}（先跑 merge_datasets.py）")

dataset = LeRobotDataset(MERGED_REPO_ID, root=MERGED_DIR)
report = {"num_frames": dataset.num_frames, "num_episodes": dataset.num_episodes}
print(f"数据集：{dataset.num_frames} 帧 / {dataset.num_episodes} 集", flush=True)

# ---- 检查一：逐关节 action vs state，定动作语义 ----
print("\n[1/4] 逐关节 action − state 偏移", flush=True)
N_SAMPLE = 6000
indices = np.linspace(0, dataset.num_frames - 1, N_SAMPLE, dtype=int)
actions = []
states = []
for i in indices:
    frame = dataset[int(i)]
    action = frame["action"]
    # action 可能带 chunk 维（训练时取一段），只取当前帧那一步
    if action.ndim > 1:
        action = action[0]
    actions.append(action.numpy())
    states.append(frame["observation.state"].numpy())
actions = np.stack(actions)
states = np.stack(states)

offset = actions - states
offset_mean = offset.mean(axis=0)
offset_std = offset.std(axis=0)
action_range = actions.max(axis=0) - actions.min(axis=0)
ratio = np.abs(offset_mean) / np.maximum(action_range, 1e-6)
corr = np.array([np.corrcoef(actions[:, j], states[:, j])[0, 1] for j in range(actions.shape[1])])

print(f"{'joint':<16}{'mean(a-s)':>12}{'std':>10}{'range':>10}{'ratio':>9}{'corr':>8}", flush=True)
joints = []
for j, name in enumerate(JOINT_NAMES):
    print(
        f"{name:<16}{offset_mean[j]:>12.4f}{offset_std[j]:>10.4f}"
        f"{action_range[j]:>10.3f}{ratio[j]:>9.3f}{corr[j]:>8.3f}",
        flush=True,
    )
    joints.append(
        {
            "joint": name,
            "offset_mean": float(offset_mean[j]),
            "offset_std": float(offset_std[j]),
            "action_range": float(action_range[j]),
            "offset_ratio": float(ratio[j]),
            "corr": float(corr[j]),
        }
    )

offset_joints = [JOINT_NAMES[j] for j in range(len(JOINT_NAMES)) if ratio[j] > OFFSET_RATIO_THRESHOLD]
semantics = "ABS" if not offset_joints else "REL"
print(f"\n动作语义判定：{semantics}", flush=True)
if offset_joints:
    print(f"存在系统性偏移的关节：{offset_joints} → 部署按相对锚定，勿直接下发绝对目标", flush=True)
else:
    print("各关节偏移都在噪声量级 → action 与 state 同一绝对坐标，部署可直接下发绝对目标", flush=True)
report["action_semantics"] = semantics
report["offset_joints"] = offset_joints
report["joints"] = joints

# ---- 检查二：任务配比与集长 ----
print("\n[2/4] 任务配比与集长", flush=True)
episodes = dataset.meta.episodes
lengths = np.array(episodes["length"])
print(f"集长：min={lengths.min()} median={int(np.median(lengths))} max={lengths.max()} mean={lengths.mean():.1f}", flush=True)
report["episode_length"] = {
    "min": int(lengths.min()),
    "median": int(np.median(lengths)),
    "max": int(lengths.max()),
    "mean": float(lengths.mean()),
}

tasks = dataset.meta.tasks
print(f"任务数：{len(tasks)}", flush=True)
for task_name in tasks.index:
    print(f"  {task_name}", flush=True)
report["num_tasks"] = int(len(tasks))
report["task_names"] = [str(t) for t in tasks.index]

# 极短集会污染 chunk 采样（不足一个 action chunk），单独点出来
short = int((lengths < 50).sum())
print(f"短于 50 帧的集：{short}", flush=True)
report["episodes_shorter_than_50"] = short

# ---- 检查三：stats 是否退化 ----
# lerobot 曾有过对大量级 int64 列求平方时溢出、把 std 静默算成 0 的 bug；
# std 为 0 意味着归一化会除零或把整个维度压成常量，必须开训前发现。
print("\n[3/4] stats 非退化", flush=True)
degenerate = []
for key in ["action", "observation.state"]:
    stats = dataset.meta.stats[key]
    mean = np.asarray(stats["mean"]).ravel()
    std = np.asarray(stats["std"]).ravel()
    print(f"{key}:", flush=True)
    print(f"  mean={np.round(mean, 4).tolist()}", flush=True)
    print(f"  std ={np.round(std, 4).tolist()}", flush=True)
    for j, s in enumerate(std):
        if s < 1e-6:
            degenerate.append(f"{key}[{j}]")
report["degenerate_stats"] = degenerate
if degenerate:
    raise SystemExit(f"stats 退化（std≈0）：{degenerate}")
print("action / observation.state 的 std 全部非零", flush=True)

# ---- 检查四：真实 dataloader 迭代（双路视频解码） ----
# 光看 meta 不算数，要按训练那条路径真解出视频帧来。
print("\n[4/4] dataloader 真实迭代", flush=True)
loader = torch.utils.data.DataLoader(dataset, batch_size=8, num_workers=4, shuffle=True)
n_batches = 20
camera_keys = []
for batch_idx, batch in enumerate(loader):
    if batch_idx == 0:
        for key, value in sorted(batch.items()):
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {tuple(value.shape)} {value.dtype}", flush=True)
                if key.startswith("observation.images."):
                    camera_keys.append(key)
    if batch_idx + 1 >= n_batches:
        break
iterated = batch_idx + 1
print(f"迭代 {iterated} batch 无报错；相机 {len(camera_keys)} 路：{camera_keys}", flush=True)
if iterated != n_batches:
    raise SystemExit(f"dataloader 只迭代出 {iterated}/{n_batches} batch")
if len(camera_keys) != 2:
    raise SystemExit(f"期望 top + wrist 两路相机，实得 {camera_keys}")
report["batches_iterated"] = iterated
report["camera_keys"] = camera_keys

REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False))
print(f"\n审计报告 -> {REPORT_PATH}", flush=True)
print(f"AUDIT_VERIFIED semantics={semantics}", flush=True)
