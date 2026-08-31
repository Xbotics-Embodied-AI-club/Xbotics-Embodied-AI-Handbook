"""验收一个 checkpoint：文件齐不齐、归一化是不是真的、预测准不准。

为什么非要单独验归一化：π0 把状态和动作按数据集统计量做标准化，真正的 mean/std 不在
模型权重里，而在两个几 KB 的伴随文件里（前处理的 normalizer、后处理的 unnormalizer）。
少了这两个文件，加载时会静默退回恒等归一化——模型照样能跑、loss 照样好看，但输出动作的
尺度整体是错的，机械臂只会停在原地或直接跑飞。这种失败在真机上出现过：整轮训练白跑，
而训练日志里看不出任何异常。所以每存一个 checkpoint 就立刻验一次，不等训练全跑完再看。

在讲12 2.3 节被引用。

用法（在 code/ 目录下跑）：
    python vla/5_vla_finetune/5_4_so101_real_sft/verify_checkpoint.py <checkpoint>/pretrained_model
    python vla/5_vla_finetune/5_4_so101_real_sft/verify_checkpoint.py <末checkpoint>/pretrained_model <冒烟>/metrics.json
第二个参数是欠训 checkpoint 的指标，当作及格地板（终检用）。
"""

import json
import sys
import os
from pathlib import Path

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from safetensors.torch import load_file

model_dir = Path(sys.argv[1])

# 可选第二个参数：一个更早的（欠训的）checkpoint 的 metrics.json，当作及格地板。
# 给了就要求本 checkpoint 每个关节都比它更好；不给就只报数、只查恒等基线那条硬底线。
floor_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None

# 用哪份数据验，从 checkpoint 自己的 train_config.json 里读——必须和训练时是同一份，
# 拿九任务合并集去验单任务模型会得出没有意义的数字。
train_cfg = json.loads((model_dir / "train_config.json").read_text())
DATA_DIR = Path(train_cfg["dataset"]["root"])
REPO_ID = train_cfg["dataset"]["repo_id"]

# 优先用训练时准备好的数据目录；它被清掉的话（训练结束后常见）回落到原始目录。
if not (DATA_DIR / "meta" / "info.json").exists():
    fallback = Path(os.environ["DATASETS_ROOT"]) / "so101" / "datasets" / DATA_DIR.name.replace("so101_", "")
    if (fallback / "meta" / "info.json").exists():
        DATA_DIR = fallback
    else:
        raw = Path(os.environ["DATASETS_ROOT"]) / "so101" / "datasets" / "raw" / DATA_DIR.name.replace("so101_", "")
        DATA_DIR = raw
print(f"验收用数据：{DATA_DIR}", flush=True)

# ---- 第一关：文件齐全 ----
print(f"[1/3] 文件完整性：{model_dir}", flush=True)
present = sorted(p.name for p in model_dir.iterdir())
for name in present:
    size = (model_dir / name).stat().st_size
    print(f"  {name}  {size:,} B", flush=True)

required = ["model.safetensors", "config.json", "train_config.json"]
missing = [name for name in required if name not in present]

# 归一化伴随文件：名字里带 normalizer 的 safetensors，前后各一个
normalizer_files = [name for name in present if name.endswith(".safetensors") and "normalizer" in name]
print(f"  归一化伴随文件 {len(normalizer_files)} 个：{normalizer_files}", flush=True)
if len(normalizer_files) < 2:
    missing.append("两个 *normalizer_processor.safetensors（缺则归一化退回恒等，动作尺度全错）")
if missing:
    raise SystemExit(f"缺文件：{missing}")
print("  文件齐全", flush=True)

# ---- 第二关：归一化里是真统计量，不是恒等 ----
print("\n[2/3] 归一化是否为真统计量", flush=True)
# 同 audit：本地 meta/ 缺失时 LeRobotDataset 会转去 Hub 拉，报一个误导性的 404
if not (DATA_DIR / "meta" / "info.json").exists():
    raise SystemExit(f"找不到验收用的数据集 {DATA_DIR}")
dataset = LeRobotDataset(REPO_ID, root=DATA_DIR)
dataset_action_mean = np.asarray(dataset.meta.stats["action"]["mean"]).ravel()
print(f"  数据集 action mean = {np.round(dataset_action_mean, 4).tolist()}", flush=True)

found_real_stats = False
for name in normalizer_files:
    tensors = load_file(model_dir / name)
    print(f"  {name}:", flush=True)
    for key, tensor in sorted(tensors.items()):
        flat = tensor.float().flatten()
        preview = np.round(flat[:6].numpy(), 4).tolist()
        print(f"    {key}: shape={tuple(tensor.shape)} 前几个值={preview}", flush=True)
        # 恒等归一化的特征是 mean 全 0、std 全 1；真统计量不会这么整齐
        if "mean" in key and flat.abs().max() > 1e-6:
            found_real_stats = True
        if "std" in key and (flat - 1.0).abs().max() > 1e-6:
            found_real_stats = True
if not found_real_stats:
    raise SystemExit("归一化是恒等（mean 全 0 / std 全 1）——这个 checkpoint 部署会跑飞")
print("  含真统计量", flush=True)

# ---- 第三关：离线预测质量，跟「照抄当前关节角」比 ----
#
# 不能用逐关节 corr(pred, action) 当判据。这批遥操数据里 action 是主臂目标位姿、
# state 是从臂实测位姿，两者天然贴在一起：审计实测「原样照抄 state」这个什么都没学的
# 恒等基线，逐关节 corr 就有 0.936–0.995，全部能过 0.9。那条门槛没有区分力。
#
# 换成三个相对量，基线由数据自己定义，不用外部拍阈值：
#   A ratio      = MAE(pred, action) / MAE(state, action)   照抄基线恒为 1，模型要明显更低
#   B delta_corr = corr(pred − state, action − state)       只看「下一步往哪挪」学到没有
#   C delta_std  = std(pred − state) / std(action − state)  抓「输出≈state」的坍缩
print("\n[3/3] 离线预测 vs 真值（对照照抄基线）", flush=True)
# 策略类按 checkpoint 自己的 config 里写的 type 来取，这样 π0 和 SmolVLA 都能验。
# 写死某一个类的话，拿它去加载另一种 checkpoint 会在读配置字段时崩。
policy_type = json.loads((model_dir / "config.json").read_text())["type"]
print(f"  policy type = {policy_type}", flush=True)
policy = get_policy_class(policy_type).from_pretrained(model_dir)
policy.eval()
policy.to("cuda")
preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=policy.config,
    pretrained_path=model_dir,
)

N_SAMPLE = 200
indices = np.linspace(0, dataset.num_frames - 1, N_SAMPLE, dtype=int)
preds = []
gts = []
states = []
for i in indices:
    frame = dataset[int(i)]
    gt = frame["action"]
    if gt.ndim > 1:
        gt = gt[0]
    batch = {k: (v.unsqueeze(0) if isinstance(v, torch.Tensor) else v) for k, v in frame.items()}
    with torch.no_grad():
        processed = preprocessor(batch)
        action = policy.select_action(processed)
        action = postprocessor(action)
    preds.append(action.squeeze(0).float().cpu().numpy())
    gts.append(gt.numpy())
    states.append(frame["observation.state"].numpy())
    policy.reset()
preds = np.stack(preds)
gts = np.stack(gts)
states = np.stack(states)

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
print(f"{'joint':<16}{'ratio':>9}{'delta_corr':>12}{'delta_std':>11}{'corr_naive':>12}", flush=True)

metrics = {}
for j, name in enumerate(JOINT_NAMES):
    pred_err = np.abs(preds[:, j] - gts[:, j]).mean()
    copy_err = np.abs(states[:, j] - gts[:, j]).mean()
    ratio = float(pred_err / copy_err)

    pred_delta = preds[:, j] - states[:, j]
    gt_delta = gts[:, j] - states[:, j]
    delta_corr = float(np.corrcoef(pred_delta, gt_delta)[0, 1])
    delta_std = float(pred_delta.std() / gt_delta.std())

    # 一并打出那个没区分力的旧指标，方便对照看它有多容易过
    corr_naive = float(np.corrcoef(preds[:, j], gts[:, j])[0, 1])

    metrics[name] = {"ratio": ratio, "delta_corr": delta_corr, "delta_std": delta_std, "corr_naive": corr_naive}
    print(f"{name:<16}{ratio:>9.3f}{delta_corr:>12.3f}{delta_std:>11.3f}{corr_naive:>12.3f}", flush=True)

metrics_path = model_dir.parent / "metrics.json"
metrics_path.write_text(json.dumps(metrics, indent=2))
print(f"\n指标写入 {metrics_path}", flush=True)

# 硬底线：不管有没有地板文件，都不能比「照抄当前关节角」还差
failures = [f"{n} ratio={m['ratio']:.3f} ≥ 1（比照抄基线还差）" for n, m in metrics.items() if m["ratio"] >= 1.0]
failures += [f"{n} delta_corr={m['delta_corr']:.3f} ≤ 0（没学到往哪挪）" for n, m in metrics.items() if m["delta_corr"] <= 0.0]

if floor_path is None:
    print("未给地板文件，只查了照抄基线这条硬底线（欠训 checkpoint 正常路径）", flush=True)
else:
    floor = json.loads(floor_path.read_text())
    print(f"\n对照地板 {floor_path}", flush=True)
    print(f"{'joint':<16}{'ratio':>9}{'floor':>9}{'delta_corr':>12}{'floor':>9}", flush=True)
    for name, m in metrics.items():
        f = floor[name]
        print(f"{name:<16}{m['ratio']:>9.3f}{f['ratio']:>9.3f}{m['delta_corr']:>12.3f}{f['delta_corr']:>9.3f}", flush=True)
        if m["ratio"] >= f["ratio"]:
            failures.append(f"{name} ratio={m['ratio']:.3f} 未低于地板 {f['ratio']:.3f}")
        if m["delta_corr"] <= f["delta_corr"]:
            failures.append(f"{name} delta_corr={m['delta_corr']:.3f} 未高于地板 {f['delta_corr']:.3f}")

if failures:
    raise SystemExit("离线判据不过门：\n  " + "\n  ".join(failures))

worst_ratio = max(m["ratio"] for m in metrics.values())
worst_delta_corr = min(m["delta_corr"] for m in metrics.values())
print(f"VERIFY_PASSED worst_ratio={worst_ratio:.3f} worst_delta_corr={worst_delta_corr:.3f}", flush=True)
print("注意：本判据只说明模型学到了非平凡的东西，不等于可上真机（本仓无 SO-ARM101 真机）", flush=True)
