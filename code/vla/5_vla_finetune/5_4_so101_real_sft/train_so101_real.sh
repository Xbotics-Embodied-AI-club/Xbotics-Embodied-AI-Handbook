#!/usr/bin/env bash
# 用一批公开的 SO-101 真机遥操数据，全参微调 π0 / SmolVLA / ACT 三个策略。
#
# 训练走 lerobot 官方 `lerobot-train` 入口，不自写训练脚本——全参微调、图像增强、
# cosine 学习率这些都是官方 config 里现成的开关，用命令行参数传进去就够了。
#
# 跑法（改下面的 MODEL 变量选模型，然后直接执行）：
#     bash vla/5_vla_finetune/5_4_so101_real_sft/train_so101_real.sh
set -euo pipefail

# ── 选哪个策略：pi0 / smolvla / act ────────────────────────────────────
MODEL=pi0

# 数据与产物都落在共享数据根下（DATASETS_ROOT 由环境给出，不在这里兜默认值）。
DATA_DIR="$DATASETS_ROOT/so101/datasets/merged_9task"
OUT_DIR="$DATASETS_ROOT/so101/outputs/${MODEL}_9task"

# 用几张卡训。单卡直接跑 lerobot-train，多卡用 accelerate 起 DDP。
NUM_GPUS=1

# ── 三个策略各自的超参 ────────────────────────────────────────────────
# π0 与 SmolVLA 是预训练 VLA，学习率要小；ACT 从零训，学习率大一个量级、步数也更多。
if [ "$MODEL" = "pi0" ]; then
  PRETRAINED="$DATASETS_ROOT/models/downloaded/huggingface/lerobot/pi0"
  LR=2.5e-5
  STEPS=30000
  POLICY_ARGS=(
    # 用 --policy.path 而不是 --policy.pretrained_path：lerobot 只认前者，
    # 认了才会连 checkpoint 自带的 config.json 一起读进来（否则配置退回默认值）。
    --policy.path="$PRETRAINED"
    --policy.dtype=bfloat16
    # 全参微调：视觉塔和 VLM 一起训，两个冻结开关都关掉。
    --policy.freeze_vision_encoder=false
    --policy.train_expert_only=false
    --policy.optimizer_lr="$LR"
    --policy.scheduler_decay_steps="$STEPS"
    --policy.scheduler_warmup_steps=2000
  )
elif [ "$MODEL" = "smolvla" ]; then
  PRETRAINED="$DATASETS_ROOT/models/downloaded/huggingface/lerobot/smolvla_base"
  LR=1e-4
  STEPS=30000
  POLICY_ARGS=(
    --policy.path="$PRETRAINED"
    --policy.optimizer_lr="$LR"
    --policy.scheduler_decay_steps="$STEPS"
    --policy.scheduler_warmup_steps=2000
  )
else
  # ACT 没有预训练基座，从随机初始化开始训，所以步数给得更多。
  STEPS=50000
  POLICY_ARGS=(
    --policy.type=act
  )
fi

BATCH=8
NUM_WORKERS=8
# 每 2500 步存一次点：训练没有自动早停，最好的那个点常常不是最后一个，
# 点存得密才有得挑。
SAVE_FREQ=2500

# ── 图像增强：把光度扰动开足 ──────────────────────────────────────────
# 这批数据出自别人的机位，光照和白平衡跟自己的臂必然不同，所以亮度/对比度范围比官方
# 默认（0.8–1.2）放宽到 0.7–1.3；白平衡靠 hue + saturation 模拟色温漂移（torchvision
# 没有独立的白平衡算子，这是标准做法）。hue 保持 ±0.05，再大就失真到不像真实相机了。
# 注意 tfs 是整体替换而不是与默认合并，六个变换必须一次写全，漏写的会直接消失。
TFS='{
  "brightness":{"weight":1.0,"type":"ColorJitter","kwargs":{"brightness":[0.7,1.3]}},
  "contrast":{"weight":1.0,"type":"ColorJitter","kwargs":{"contrast":[0.7,1.3]}},
  "hue":{"weight":1.0,"type":"ColorJitter","kwargs":{"hue":[-0.05,0.05]}},
  "saturation":{"weight":1.0,"type":"ColorJitter","kwargs":{"saturation":[0.5,1.5]}},
  "sharpness":{"weight":1.0,"type":"SharpnessJitter","kwargs":{"sharpness":[0.5,1.5]}},
  "affine":{"weight":1.0,"type":"RandomAffine","kwargs":{"degrees":[-5.0,5.0],"translate":[0.05,0.05]}}
}'

TRAIN_ARGS=(
  "${POLICY_ARGS[@]}"
  --policy.device=cuda
  --policy.push_to_hub=false
  --dataset.repo_id=so101/pickplace_9task
  --dataset.root="$DATA_DIR"
  --dataset.video_backend=torchcodec
  --dataset.image_transforms.enable=true
  --dataset.image_transforms.max_num_transforms=5
  --dataset.image_transforms.tfs="$TFS"
  --batch_size="$BATCH"
  --num_workers="$NUM_WORKERS"
  --steps="$STEPS"
  --save_freq="$SAVE_FREQ"
  --log_freq=100
  --output_dir="$OUT_DIR"
  --job_name="so101-$MODEL"
  --wandb.enable=true
  --wandb.project=so101
)

if [ "$NUM_GPUS" = "1" ]; then
  lerobot-train "${TRAIN_ARGS[@]}"
else
  accelerate launch --multi_gpu --num_processes "$NUM_GPUS" \
    "$(command -v lerobot-train)" "${TRAIN_ARGS[@]}"
fi

echo "训练完成，产物在 $OUT_DIR"
echo "接着验收 checkpoint： python vla/5_vla_finetune/5_4_so101_real_sft/verify_checkpoint.py $OUT_DIR/checkpoints/last/pretrained_model"
