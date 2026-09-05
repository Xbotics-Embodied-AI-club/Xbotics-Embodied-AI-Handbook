#!/usr/bin/env bash
# 在 SO-101 上从零训 ACT，训练集是仿真＋真机混成的一份 —— 第10讲 3.2 节。
#
# 任务只有一个：把方块放进料箱。训练集是**仿真与真机混在一起**的：
# 仿真产线自动产出的 498 集，加上公开的 SO-101 真机遥操 300 集，共 798 集。
# 两侧的指令文本逐字相同（`Pick up a cube and place in the bin`），落在同一个任务号下 ——
# 这是它们能混成一份的前提，也是第9讲讲的那件事：格式、量纲、单位、归一化对齐了，
# 仿真数据就能当作一种真机录制来用。
#
# 与同目录 `train_act_cuboid.sh` 的区别只有数据来源。算法、网络、超参都不动 ——
# 动作维度、状态维度、相机路数由数据集自己的 feature 描述定，这正是 3.2 节要说明的事。
#
# 数据集与产线源码：https://huggingface.co/datasets/Harrysunshine/so101-sim-pickplace-v2
#
# ★ ACT **没有预训练基座**，从随机初始化开始训，所以要走的样本数比微调一个 VLA 多。
#   50000 步 × batch 64 ≈ 320 万样本，对这份 27 万帧的数据集是 11 个 epoch 出头。
#   步数不照抄同目录那份真机脚本的 100000 —— 那份的数据集大小不同，
#   **该对齐的是过多少遍数据，不是跑多少步**。
#
# ★ 训练途中开评测（每 10000 步、10 局）。ACT 的训练损失一路降但成功率早就横住 ——
#   只看 loss 判断还该不该继续训，会一直得到「该」。要画那条曲线就得有这些点。
#
# 用法：
#     bash vla/3_imitation_learning/3_1_act/train_act_so101_sim.sh
#     STEPS=50000 GPU=0 bash .../train_act_so101_sim.sh      # 换卡、少训一点
set -euo pipefail

BASE="${BASE:-/work/EAI-exp-002}"
SIM="${SIM:-$BASE/datasets/sim-hf/cube40}"
REAL="${REAL:-/work/so101/datasets/raw/pick_up_a_cube_and_place_in_the_bin}"
DATA="${DATA:-$BASE/datasets/sim-real-cube}"
OUT="${OUT:-$BASE/outputs/act-sim-real-cube}"
VENV="${VENV:-$HOME/.venv/Xbotics2-handbook}"

BATCH="${BATCH:-64}"
STEPS="${STEPS:-50000}"
GPU="${GPU:-1}"

# ── 合并仿真与真机 ────────────────────────────────────────────────────
# 走 `0_dataset_gen/merge_dataset.py`（官方合并命令的薄封装），不自己拼 parquet ——
# 自己拼要同时改 episodes / tasks / stats 三张表，漏一张**不报错**，只在训练时
# 表现为某些集读不到任务文本。官方合并的第一步逐字比 fps / robot_type / features，
# 任一不等就抛错，所以这一步同时也是"两侧口径真的一致"的检查。
if [ ! -f "$DATA/meta/info.json" ]; then
  for d in "$SIM" "$REAL"; do
    [ -f "$d/meta/info.json" ] || { echo "★ 缺 $d"; exit 1; }
  done
  rm -rf "$DATA"
  "$VENV/bin/python" "$(dirname "$0")/../../0_dataset_gen/merge_dataset.py" "$DATA" "$SIM" "$REAL"
fi
python3 -c "
import json, sys
m = json.load(open('$DATA/meta/info.json'))
print(f\"  训练集 {m['total_episodes']} 集 / {m['total_frames']} 帧 / {m['total_tasks']} 个任务\")
# 两侧指令逐字相同 ⇒ 合完只该有一个任务。多于一个说明文本没对上，混不成一份。
sys.exit(0 if m['total_tasks'] == 1 else 1)
" || { echo "★ 合并后不是 1 个任务 —— 仿真与真机的指令文本没对上"; exit 1; }

export CUDA_VISIBLE_DEVICES="$GPU"

# 四个 `--env.*` 必须与产数据时逐字一致，途中评测才量得准：`control_mode` 配错，
# 绝对关节角会被当成归一化增量、逐维 clip 到 ±1，手臂以包线最大速度朝错误方向走。
# 不给 `--env.task_description`：仿真器按场景查自己那张表，与数据集 tasks 逐字相同。
#
# 图像归一化用数据集自己的统计量而不是 ImageNet：渲染图的色彩分布离自然照片更远。
# 这一条与同目录真机脚本一致。
"$VENV/bin/lerobot-train" \
  --policy.type=act \
  --policy.device=cuda \
  --policy.use_amp=true \
  --policy.push_to_hub=false \
  --dataset.repo_id=Harrysunshine/so101-sim-pickplace-v2 \
  --dataset.root="$DATA" \
  --dataset.use_imagenet_stats=false \
  --env.type=so101_sim \
  --env.task=SO101PickPlaceCube40-v1 \
  --env.control_mode=pd_joint_pos \
  --env.observation_width=640 \
  --env.observation_height=480 \
  --env.episode_length=500 \
  --env.fps=30 \
  --batch_size="$BATCH" \
  --steps="$STEPS" \
  --eval_freq=10000 \
  --eval.n_episodes=10 \
  --eval.batch_size=5 \
  --save_freq=10000 \
  --log_freq=100 \
  --num_workers="${WORKERS:-8}" \
  --output_dir="$OUT" \
  --job_name=act-sim-real-cube \
  --wandb.enable=true \
  --wandb.mode=online \
  --wandb.project=EAI-exp-002 \
  --wandb.disable_artifact=true \
  "$@"

echo "TRAIN_ACT_DONE -> $OUT"
echo "接着验收：bash vla/3_imitation_learning/3_1_act/eval_act_so101_sim.sh $OUT/checkpoints/last/pretrained_model"
