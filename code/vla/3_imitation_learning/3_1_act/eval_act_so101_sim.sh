#!/usr/bin/env bash
# 把训好的 ACT 在 SO-101 仿真器里跑一次足量评测（第10讲 3.2 节）。
#
# 训练途中那些评测点每次只有 10 局，用来画趋势；这一步是出结论的，默认 50 局。
#
# ★ 四个 `--env.*` 必须与产数据、与训练时逐字一致，否则量的是另一套设置下的成功率：
#   `control_mode` 配错 ⇒ 绝对关节角被当成归一化增量、逐维 clip 到 ±1；
#   `observation_width/height` 配错 ⇒ 策略看到的画面与训练时不同分辨率。
# ★ 不给 `--env.task_description`：仿真器按场景查它自己那张表，取值与数据集
#   `tasks.parquet` 逐字相同。手写指令与数据集差一个字，量到的就是
#   「策略没见过这句话」时的成功率。
# ★ 这里**不需要 `--rename_map`**：ACT 从零训，相机键直接沿用数据集里的
#   `observation.images.top` / `.wrist`。第12讲那轮要改名，是因为 `smolvla_base`
#   的预训练权重认的是 `camera1` / `camera2`。
#
# 用法：bash vla/3_imitation_learning/3_1_act/eval_act_so101_sim.sh <checkpoint 目录> [<局数>]
set -euo pipefail

CKPT="${1:?用法: bash eval_act_so101_sim.sh <checkpoint 目录> [<局数>]}"
N="${2:-50}"
VENV="${VENV:-$HOME/.venv/Xbotics2-handbook}"
OUT="${OUT_DIR:-/work/EAI-exp-002/outputs/eval-act-sim-cube40}"
ENV_ID=SO101PickPlaceCube40-v1

export CUDA_VISIBLE_DEVICES="${GPUS:-1}"

[ -f "$CKPT/config.json" ] || { echo "★ $CKPT 不像是 pretrained_model 目录"; exit 1; }

echo "===== $ENV_ID（$N 局）====="
"$VENV/bin/lerobot-eval" \
  --policy.path="$CKPT" \
  --policy.device=cuda \
  --env.type=so101_sim \
  --env.task="$ENV_ID" \
  --env.control_mode=pd_joint_pos \
  --env.observation_width=640 \
  --env.observation_height=480 \
  --env.episode_length=500 \
  --env.fps=30 \
  --eval.n_episodes="$N" \
  --eval.batch_size=5 \
  --output_dir="$OUT/$ENV_ID" 2>&1 | tail -3

"$VENV/bin/python" - "$OUT/$ENV_ID/eval_info.json" <<'PY'
import json
import sys
from pathlib import Path

f = Path(sys.argv[1])
if not f.is_file():
    sys.exit(f"★ 没有 {f} —— 这一场没跑完")
doc = json.loads(f.read_text())
# 顶层键是 `overall`。取错键时 `.get` 会给 None、脚本照常打印就过去了 ——
# 那正是「空结果被当成结论」，所以取不到直接报错。
if "overall" not in doc:
    sys.exit(f"★ {f} 里没有 `overall`，实际顶层键是 {list(doc)}")
agg = doc["overall"]
# 局数在 overall 里，不在顶层的 per_episode —— 那个键根本不存在，
# 于是 len([]) 打出「0 局」。空取值当成结论，就是这么来的。
print(f"  pc_success = {agg['pc_success']:.1f}%   avg_max_reward = {agg['avg_max_reward']:.3f}"
      f"   （{agg['n_episodes']} 局）")
print("EVAL_ACT_END")
PY
