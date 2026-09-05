#!/usr/bin/env bash
# LoRA 微调（SmolVLA / pi0），数据与评测口径与全参微调那一轮**逐字相同**。
#
# 为什么能与全参那轮对读：除了 `--peft.*` 这几个开关，数据集、步数、warmup、图像增强、
# 卡数、effective batch、评测环境参数全部照抄 —— 两轮之间唯一的差别就是"怎么改权重"。
#
# ★ 上游默认的 `target_modules` 对**换机器人**这件事开得太窄，实测学不动。
#   第一版照抄默认（`_get_default_peft_targets`，modeling_smolvla.py:484）：
#   只挂动作专家的 q/v ＋ 五个具身投影，可训练参数 742,656（占全模型 0.16%）。
#   跑到 8000 步三个评测点全是 0.0%，而全参微调在第一个评测点就 60%。
#   决定性的对读是 loss —— **LoRA 在 7K 步是 0.198，全参在 1K 步已经 0.194**：
#   不是学歪了，是欠拟合到差一个数量级的进度。
#
#   基座 checkpoint 里可挂 q/v 的一共三族（读权重名单数出来的）：
#     动作专家 lm_expert                32 个 ← 默认只挂了这一族
#     文本塔  vlm.model.text_model      32 个
#     视觉塔  vlm.model.vision_model    48 个
#   全参那一轮的结论正是「解冻视觉塔才跨得过新机器人的视觉差异」（见 2.4.1 节），
#   而默认配置把视觉塔和文本塔整个冻着，只让动作专家一个人扛。三族一起挂。
#
# ★ 五个具身投影改成**全训**而不是 rank-16 近似。它们是把这台机器人的关节数
#   映射进模型维度的层，`full_training_modules` 的用途注释里点名的就是这种层
#   （configs/default.py:94）。低秩近似一个本来就不大的投影没有意义。
#
# ★ 学习率随之提到 3e-4。全参那轮 5e-5 是为了别冲垮预训练表征；LoRA 的适配器
#   随机初始化、基座冻着，1e-4 在上面那组数字里已经证明太慢。
#
# 用法：bash train_lora.sh <smolvla|pi0> [额外的 lerobot-train 参数]
set -euo pipefail

MODEL="${1:?用法: bash train_lora.sh <smolvla|pi0> [额外参数]}"
shift || true
# 数据与产物的根。沿用同目录 `train_so101_real.sh` 的约定：DATASETS_ROOT 由环境给出。
BASE="${BASE:-$DATASETS_ROOT/so101-sft}"
# 指定 VENV 就用它 bin 下的命令，不指定就用 PATH 上的（`uv sync` 之后即在 PATH）。
BIN="${VENV:+$VENV/bin/}"
DATA=$BASE/datasets/sim-real-10task
# 训练前把数据集镜像到内存盘再读 —— 只是这台机器上的加速手段，不是流程的一部分。
# 给 `FAST_ROOT=` 空值就直接读 $DATA。内存盘装不下就别开：截断了**不报错**，
# 只表现为某些集读不到。
ROOT="${FAST_ROOT-/dev/shm/so101-sft/sim-real-10task}"
OUT="$BASE/outputs/lora-$MODEL"

case "$MODEL" in
  smolvla) PRETRAINED=lerobot/smolvla_base ;;
  pi0)     PRETRAINED=lerobot/pi0 ;;
  *)       echo "★ 只支持 smolvla / pi0，收到 $MODEL"; exit 1 ;;
esac

export CUDA_VISIBLE_DEVICES="${GPUS:-2,3,4,5,6,7}"
NGPU=$(printf '%s' "$CUDA_VISIBLE_DEVICES" | awk -F, '{print NF}')
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=online
# 代理不在这里写死：`http_proxy` 由机器的 /etc/environment 预置（AGENTS.md）。
# ⚠️ 非登录 shell 读不到 /etc/environment —— 用 nohup/setsid 后台起时若拉不到 HF，
#    在**调用这个脚本的那一侧**导出，不要把某台机器的代理地址钉进脚本。

[ -f "$DATA/meta/info.json" ] || { echo "★ $DATA 不在 —— 先跑 w4_build_and_train.sh 的合并那一步"; exit 1; }
if [ -n "$ROOT" ]; then
  mkdir -p "$(dirname "$ROOT")"
  rsync -a --delete "$DATA/" "$ROOT/"
  [ -f "$ROOT/meta/info.json" ] || { echo "★ 镜像后没有 meta/info.json"; exit 1; }
else
  ROOT="$DATA"
fi

# 图像增强与全参那轮逐字相同。`tfs` 是整体替换不是合并，六个必须一次写全。
TFS='{
  "brightness":{"weight":1.0,"type":"ColorJitter","kwargs":{"brightness":[0.7,1.3]}},
  "contrast":{"weight":1.0,"type":"ColorJitter","kwargs":{"contrast":[0.7,1.3]}},
  "hue":{"weight":1.0,"type":"ColorJitter","kwargs":{"hue":[-0.05,0.05]}},
  "saturation":{"weight":1.0,"type":"ColorJitter","kwargs":{"saturation":[0.5,1.5]}},
  "sharpness":{"weight":1.0,"type":"SharpnessJitter","kwargs":{"sharpness":[0.5,1.5]}},
  "affine":{"weight":1.0,"type":"RandomAffine","kwargs":{"degrees":[-5.0,5.0],"translate":[0.05,0.05]}}
}'

rm -rf "$OUT"
"${BIN}accelerate" launch --num_processes="$NGPU" --mixed_precision=bf16 \
  "${BIN}lerobot-train" \
  --policy.path="$PRETRAINED" \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --peft.method_type=LORA \
  --peft.r=64 \
  --peft.target_modules='(model\.vlm_with_expert\.(lm_expert|vlm\.model\.text_model|vlm\.model\.vision_model\.encoder)\..*\.(q|v)_proj)' \
  --peft.full_training_modules='["state_proj","action_in_proj","action_out_proj","action_time_mlp_in","action_time_mlp_out"]' \
  --dataset.repo_id="xbotics/so101-sim-real-10task" \
  --dataset.root="$ROOT" \
  --dataset.image_transforms.enable=true \
  --dataset.image_transforms.max_num_transforms=5 \
  --dataset.image_transforms.tfs="$TFS" \
  --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' \
  --env.type=so101_sim \
  --env.task=SO101PickPlaceCube40-v1 \
  --env.control_mode=pd_joint_pos \
  --env.observation_width=640 \
  --env.observation_height=480 \
  --env.episode_length=500 \
  --env.fps=30 \
  --batch_size=16 \
  --policy.optimizer_lr=3e-4 \
  --policy.scheduler_warmup_steps=2000 \
  --policy.scheduler_decay_steps=30000 \
  --steps=30000 \
  --eval_freq=5000 \
  --eval.n_episodes=10 \
  --eval.batch_size=5 \
  --save_freq=2500 \
  --log_freq=100 \
  --num_workers=8 \
  --wandb.enable=true \
  --wandb.mode=online \
  --wandb.project="${WANDB_PROJECT:-so101}" \
  --wandb.disable_artifact=true \
  --job_name="lora-$MODEL" \
  --output_dir="$OUT" \
  "$@"

echo "TRAIN_LORA_DONE $MODEL -> $OUT"
