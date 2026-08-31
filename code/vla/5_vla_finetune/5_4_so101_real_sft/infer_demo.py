"""拿训好的 checkpoint 回放一集数据，看它预测的动作跟遥操真值差多少。

本仓没有 SO-ARM101 真机，所以这里做的是离线回放：把某一集的画面逐帧喂给模型，
把它输出的动作和当时人遥操的真实动作画在一起对比。

每个关节报三个数：模型的平均误差、"照抄当前关节角"这个什么都没学的基线的平均误差、
以及两者的比值。比值小于 1 才说明模型确实学到了东西。

注意验收脚本 verify_checkpoint.py 是在全库均匀抽样，这里是连续回放单独一集 ——
两边样本不同，数值不会完全一样，但短板关节的排序应该对得上。
"""

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi0.modeling_pi0 import PI0Policy

# 采用点是 025000 而不是最后那个 030000：逐点验收显示指标在 25k 触底、之后开始退化。
MODEL_DIR = Path(os.environ["DATASETS_ROOT"]) / "so101" / "outputs" / "pi0_9task" / "checkpoints" / "025000" / "pretrained_model"
OUT_DIR = Path(os.environ["DATASETS_ROOT"]) / "so101" / "outputs" / "infer_demo"
EPISODE = 0

# 数据训练时已经拷进内存了，直接读那份
DATA_DIR = Path(os.environ["DATASETS_ROOT"]) / "so101" / "datasets" / "merged_9task"
REPO_ID = "so101/pickplace_9task"

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

OUT_DIR.mkdir(parents=True, exist_ok=True)

# 只加载要回放的那一集。整库加载要吃 68G 内存（验收脚本就是那么干的，
# 并发跑过两次 OOM），单集几乎不占。
dataset = LeRobotDataset(REPO_ID, root=DATA_DIR, episodes=[EPISODE])
task_text = dataset.meta.tasks.index[0] if hasattr(dataset.meta.tasks, "index") else "?"
print(f"回放第 {EPISODE} 集，共 {dataset.num_frames} 帧，任务：{task_text}", flush=True)

policy = PI0Policy.from_pretrained(MODEL_DIR)
policy.eval()
policy.to("cuda")
preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=policy.config,
    pretrained_path=MODEL_DIR,
)

preds, gts, states = [], [], []
for i in range(dataset.num_frames):
    frame = dataset[i]
    gt = frame["action"]
    if gt.ndim > 1:
        gt = gt[0]
    batch = {k: (v.unsqueeze(0) if isinstance(v, torch.Tensor) else v) for k, v in frame.items()}
    with torch.no_grad():
        action = postprocessor(policy.select_action(preprocessor(batch)))
    preds.append(action.squeeze(0).float().cpu().numpy())
    gts.append(gt.numpy())
    states.append(frame["observation.state"].numpy())
    policy.reset()

preds, gts, states = np.stack(preds), np.stack(gts), np.stack(states)

print(f"\n{'joint':<16}{'模型MAE':>10}{'照抄MAE':>10}{'比值':>8}", flush=True)
summary = {}
for j, name in enumerate(JOINT_NAMES):
    pred_mae = float(np.abs(preds[:, j] - gts[:, j]).mean())
    copy_mae = float(np.abs(states[:, j] - gts[:, j]).mean())
    ratio = pred_mae / copy_mae
    summary[name] = {"pred_mae": pred_mae, "copy_mae": copy_mae, "ratio": ratio}
    print(f"{name:<16}{pred_mae:>10.3f}{copy_mae:>10.3f}{ratio:>8.3f}", flush=True)

worst = max(summary.items(), key=lambda kv: kv[1]["ratio"])
print(f"\n最弱关节：{worst[0]}（比值 {worst[1]['ratio']:.3f}）", flush=True)

(OUT_DIR / f"infer_demo_ep{EPISODE}.json").write_text(json.dumps(summary, indent=2))

fig, axes = plt.subplots(2, 3, figsize=(15, 7))
for j, (name, ax) in enumerate(zip(JOINT_NAMES, axes.ravel())):
    ax.plot(gts[:, j], label="teleop ground truth", linewidth=2)
    ax.plot(preds[:, j], label="pi0 prediction", linestyle="--")
    ax.set_title(f"{name}  (ratio {summary[name]['ratio']:.3f})")
    ax.set_xlabel("frame")
axes[0][0].legend()
fig.tight_layout()
fig_path = OUT_DIR / f"replay_ep{EPISODE}_025000.png"
fig.savefig(fig_path, dpi=110)
print(f"曲线图 {fig_path}", flush=True)

# 这个 demo 能说明 checkpoint 加载正常、归一化没退回恒等、输出量纲和真值对得上。
# 它不能说明真机可用：这里每帧都 policy.reset() 后独立预测，是开环逐帧比对，
# 真机上动作会连续下发、误差逐步累积。piper 那边离线相关性 0.995 的模型真机照样抓空。
print("\n注意：这是开环逐帧回放，不等于真机闭环可用。", flush=True)
