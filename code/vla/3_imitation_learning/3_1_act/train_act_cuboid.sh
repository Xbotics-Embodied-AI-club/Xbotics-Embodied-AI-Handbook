#!/usr/bin/env bash
# 用 SO-101 真机采的 cuboid 数据训 ACT（第10讲 3.2 节）。
# 算法和 LIBERO 那条线完全一样，只换了数据来源：动作维度、状态维度、相机路数
# 都由数据集自己的 feature 描述决定，超参沿用默认。
#
# 从头训：   bash vla/3_imitation_learning/3_1_act/train_act_cuboid.sh
# 接着上次： RESUME=true bash vla/3_imitation_learning/3_1_act/train_act_cuboid.sh
#
# 几个可以用环境变量覆盖的值，用来适配手头这台机器：
#   BATCH_SIZE   显存不够就调小，训练时长可以换显存
#   NUM_WORKERS  DataLoader 子进程数；共享内存偏小的机器上设 0 最稳
#   SAVE_FREQ    存 checkpoint 的间隔，按你能接受的「崩了要重跑多久」来定
set -euo pipefail

export MUJOCO_GL=egl

OUTPUT_DIR=vla/3_imitation_learning/3_1_act/outputs/act_cuboid_local
RESUME="${RESUME:-false}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SAVE_FREQ="${SAVE_FREQ:-1200}"

# 续训要读回上次的训练配置，才能把 optimizer、scheduler 和步数一并恢复；
# 只加载权重会让学习率调度从头开始，等于换了个训练过程。
CONFIG_PATH="${OUTPUT_DIR}/checkpoints/last/pretrained_model/train_config.json"

TRAIN_CMD=(
  uv run lerobot-train
  # 显式指定本地数据集目录，避免读到缓存里的同名旧数据集。
  --dataset.repo_id=local/cuboid
  --dataset.root=vla/3_imitation_learning/3_1_act/local/cuboid
  # 真机图像的分布和自然照片差得远，用数据集自己的统计量归一化。
  --dataset.use_imagenet_stats=false
  --dataset.video_backend=pyav
  --policy.type=act
  --policy.device=cuda
  --policy.use_amp=true
  --policy.push_to_hub=false
  --output_dir="${OUTPUT_DIR}"
  --job_name=act_cuboid_local
  --batch_size="${BATCH_SIZE}"
  --num_workers="${NUM_WORKERS}"
  --steps=100000
  --save_freq="${SAVE_FREQ}"
  --log_freq=5
  --save_checkpoint=true
  # 真机这条线默认不传 wandb，避免采集现场没网时卡住。
  --wandb.enable=false
)

if [[ "${RESUME}" == "true" ]]; then
  if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "找不到续训配置: ${CONFIG_PATH}" >&2
    exit 1
  fi

  TRAIN_CMD+=(
    --resume=true
    --config_path="${CONFIG_PATH}"
  )
fi

"${TRAIN_CMD[@]}"
