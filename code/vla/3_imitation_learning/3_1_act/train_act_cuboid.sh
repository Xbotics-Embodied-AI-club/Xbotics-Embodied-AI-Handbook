#!/usr/bin/env bash
# 用 bash 运行这个脚本。
set -euo pipefail
# 任一命令出错就直接退出，避免后面继续跑。

export MUJOCO_GL=egl
# 让 MuJoCo 走 EGL 显示后端，适合无桌面环境。

OUTPUT_DIR=vla/3_imitation_learning/3_1_act/outputs/act_cuboid_local
`# 新训练和续训都使用这个输出目录。`
RESUME="${RESUME:-false}"
`# 默认从头训练；要继续训练时运行 RESUME=true bash vla/3_imitation_learning/3_1_act/train_act_cuboid.sh。`
CONFIG_PATH="${OUTPUT_DIR}/checkpoints/last/pretrained_model/train_config.json"
`# 继续训练时读取 last checkpoint 里的训练配置。`
SAVE_FREQ="${SAVE_FREQ:-1200}"
`# 默认每 1200 step 保存一次；按当前 batch 64 速度约半小时，可用 SAVE_FREQ=2000 覆盖。`
NUM_WORKERS="${NUM_WORKERS:-0}"
`# 默认不用 DataLoader 子进程，避免 /dev/shm 太小导致 worker bus error。`
BATCH_SIZE="${BATCH_SIZE:-64}"
`# batch size 默认 64；256 在当前 GPU 显存占用下会 OOM，可用 BATCH_SIZE=128 覆盖。`

TRAIN_CMD=(
  uv run lerobot-train
  --dataset.repo_id=local/cuboid \
`# 直接读取本地 cuboid 数据集。` \
  --dataset.root=vla/3_imitation_learning/3_1_act/local/cuboid \
`# 显式指定你的本地数据集目录，避免读到缓存里的旧 local/cuboid。` \
  --dataset.use_imagenet_stats=false \
`# 不使用 ImageNet 的归一化统计量。` \
  --dataset.video_backend=pyav \
`# 用 pyav 作为视频读取后端。` \
  --policy.type=act \
`# 训练 ACT policy。` \
  --policy.device=cuda \
`# 训练放到 GPU 上跑。` \
  --policy.use_amp=true \
`# 开启混合精度训练。` \
  --policy.push_to_hub=false \
`# 不把模型上传到 Hugging Face Hub。` \
  --output_dir="${OUTPUT_DIR}" \
`# 训练输出目录写死，checkpoint 也会放在这个目录下面。` \
  --job_name=act_cuboid_local \
`# 训练任务名写死。` \
  --batch_size="${BATCH_SIZE}" \
`# batch size；默认 64 更稳，机器空闲时可以调大。` \
  --num_workers="${NUM_WORKERS}" \
`# DataLoader worker 数量；本机 /dev/shm 较小，默认 0 更稳。` \
  --steps=100000 \
`# 总训练步数参考 train_act_libero.py。` \
  --save_freq="${SAVE_FREQ}" \
`# 按 SAVE_FREQ 保存 checkpoint。` \
  --log_freq=5 \
`# 每 5 步打印一次日志，参考 train_act_libero.py。` \
  --save_checkpoint=true \
`# 开启 checkpoint 保存。` \
  --wandb.enable=false
`# 关闭 wandb。`
)

if [[ "${RESUME}" == "true" ]]; then
  if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "找不到续训配置: ${CONFIG_PATH}" >&2
    exit 1
  fi

  TRAIN_CMD+=(
    --resume=true
`# 从 checkpoint 继续训练。`
    --config_path="${CONFIG_PATH}"
`# 使用 last checkpoint 的 train_config.json 恢复 optimizer/scheduler/step。`
  )
fi

"${TRAIN_CMD[@]}"
