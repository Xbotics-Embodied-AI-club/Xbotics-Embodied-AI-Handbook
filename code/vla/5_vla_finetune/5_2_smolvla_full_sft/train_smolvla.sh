#!/usr/bin/env bash
# 全量微调 SmolVLA → SO-101 仿真抓取（ReachCube）。
# 从社区预训练权重 lerobot/smolvla_base 出发，用 rl/3_2 生成的 SO-101 仿真数据集
# 做全参数微调（无 LoRA / 无 PEFT），产出一个能在同一仿真里评测的策略。
# 在 code/ 目录运行：bash vla/5_vla_finetune/5_2_smolvla_full_sft/train_smolvla.sh
set -euo pipefail

# 数据集：rl/3_offpolicy/3_2_so101_offpolicy/datagen/gen_dataset.py 的产物，
# 一个标准 LeRobotDataset（单目相机 base_camera + 6 维关节状态/动作）。
DATASET_ROOT="$DATASETS_ROOT/so101_sim/_gen/SO101ReachCube-v1/dataset"

# 微调结果落通用数据根，不进 git。
OUTPUT_DIR="$DATASETS_ROOT/models/trained/so101_sim_smolvla/SO101ReachCube-v1"

# 步数按收敛取；97GB 显存可开大 batch。数据是同一仿真内分布，20k 步足够拟合。
STEPS=20000
BATCH_SIZE=64

# 训练控制台输出（含每 log_freq 步的 loss）留一份到 scratch，供 plot_loss.py 画曲线。
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$HERE/scratch"
LOG="$HERE/scratch/train_smolvla.log"

# 关键一步：rename_map。
# smolvla_base 预训练权重按 camera1/2/3 三路相机命名；本仿真数据集只有一路 base_camera。
# 把 base_camera 改名映射到 camera1 即可对齐；缺 camera2/3 允许——特征校验只要求
# 数据提供的相机是权重期望相机的子集。
uv run lerobot-train \
    --policy.path=lerobot/smolvla_base \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --dataset.repo_id=so101_sim/reachcube \
    --dataset.root="$DATASET_ROOT" \
    --rename_map='{"observation.images.base_camera": "observation.images.camera1"}' \
    --batch_size=$BATCH_SIZE \
    --steps=$STEPS \
    --save_freq=5000 \
    --log_freq=200 \
    --num_workers=8 \
    --wandb.enable=false \
    --output_dir="$OUTPUT_DIR" 2>&1 | tee "$LOG"

# 结果：$OUTPUT_DIR/checkpoints/last/pretrained_model 是最终策略；
# checkpoints/<step>/ 是各存点。训练日志（loss 曲线数据）在 $OUTPUT_DIR 下。
