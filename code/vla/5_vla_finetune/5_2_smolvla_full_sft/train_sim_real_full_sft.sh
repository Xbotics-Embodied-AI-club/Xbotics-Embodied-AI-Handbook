#!/usr/bin/env bash
# 把仿真与真机数据合成一份，对 SmolVLA / $\pi_0$ 做**全参数微调** —— 第12讲 2.4 节那一轮。
#
# 与同目录 `train_smolvla.sh` 的分别：那个只用仿真数据，是把流程跑通的最小闭环；
# 这个是 2.4 节实际跑的那一轮 —— 仿真三个抓放任务 + 公开的真机九个任务**全部混在一起**，
# 共 12 份、10 个不同任务。两侧共有 cube 与 can（指令文本逐字相同），仿真独有 small cube，
# 真机独有另外七个。要说明的是：口径对齐之后，仿真数据就能当作一种真机录制来用，
# 一个模型、一套权重同时覆盖两边独有的任务。
#
# ★ **全量微调是显式开的。** `lerobot/smolvla_base` 的 config 里
#   `freeze_vision_encoder: true` / `train_expert_only: true`，lerobot 的默认值也是这两个 ——
#   不显式覆盖就只训动作专家（实测 99,880,992 / 450,046,176 ≈ 22%），视觉塔全程冻着。
#   而这个实验要跨的正是仿真渲染与真机图像的域差，冻着视觉塔等于让动作专家硬扛。
#   **判据是训练日志第一屏的 `num_learnable_params`，不是命令行里传了什么。**
#
# ★ 解冻之后**学习率必须大降**。3.4e-4 那档是按动作专家（随机初始化的头）定的；
#   对预训练好的视觉塔与语言塔是灾难性的大。`get_optim_params` 返回扁平的
#   `self.parameters()`，没有分组口子 ⇒ 只能整体取一个折中值。
#
# ★ **必须写 `--policy.optimizer_lr`，不能写 `--optimizer.lr`。** 后者会被静默忽略：
#   `configs/train.py` 的 `validate()` 里 `use_policy_training_preset`（默认 True）
#   会把命令行给的整个优化器配置覆盖掉。命令行照收不报错，wandb 里记的仍是默认值。
#
# 用法：
#     bash train_sim_real_full_sft.sh                 # 默认 smolvla
#     MODEL=pi0 bash train_sim_real_full_sft.sh       # 换基座
#     MODEL=pi0 BATCH=2 GPUS=2,3 bash ...             # 显存紧就调小 batch / 少用几张卡
set -euo pipefail

MODEL="${MODEL:-smolvla}"
REPO="${REPO:-Harrysunshine/so101-sim-pickplace-v2}"
# 数据与产物的根。沿用同目录 `train_so101_real.sh` 的约定：DATASETS_ROOT 由环境给出。
BASE="${BASE:-$DATASETS_ROOT/so101-sft}"
# 指定 VENV 就用它 bin 下的命令，不指定就用 PATH 上的（`uv sync` 之后即在 PATH）。
BIN="${VENV:+$VENV/bin/}"
SIM="$BASE/datasets/sim-hf"
# 真机 9 个任务的原始数据根。公开出处见讲义 2.4 节末尾的 ModelScope 链接。
REAL="${REAL:-$DATASETS_ROOT/so101/datasets/raw}"
DATA="$BASE/datasets/sim-real-10task"
# 可覆盖：探显存时另给一个落点，别撞上正式那轮的目录。
#   lerobot 见到 output_dir 已存在且 resume=false 会直接 FileExistsError 退出。
OUT="${OUT:-$BASE/outputs/full-sft-$MODEL}"
GPUS="${GPUS:-0,1,2,3,4,5,6}"

# 两个基座的差别只有这三样。batch 是**每进程**的（accelerate 默认 split_batches=False）。
case "$MODEL" in
  smolvla)
    PRETRAINED=lerobot/smolvla_base
    # 16 是探针实测出来的上限档，不是拍的：那几张卡各被别的容器占着 22~38 GB，
    # 只剩约 42 GB；全解冻后可训参数 403M（此前 100M），batch 64 直接 OOM。
    BATCH="${BATCH:-16}"; LR="${LR:-5e-5}"; DTYPE=""
    ;;
  pi0)
    # ★ 用**原版 pi0**，但要先把它的 config 补成新格式。
    #   `lerobot/pi0` 在 HF 上最新那次提交（2025-09-19「Migrate pi0 weights to pipeline」）
    #   把权重迁到了新结构，**config.json 却没跟着改** —— 里面 8 个字段本版 lerobot 已经
    #   不认，draccus 解码直接抛 DecodingError，训练在第一秒退出。
    #   证据：那份权重与一份能装起来的 pi0 checkpoint 逐一比对，777 个张量名完全相同 ⇒
    #   落后的只有那个 JSON。`fix_pi0_config.py` 现算一份新的（它会核对被删字段的取值，
    #   不静默丢掉非默认值）。**这不是显存问题，别去调 batch。**
    PRETRAINED="${PI0_DIR:-$BASE/models/pi0-base}"
    if [ ! -f "$PRETRAINED/config.json" ]; then
      RAW="${PI0_RAW:-$BASE/models/pi0-raw}"
      [ -f "$RAW/config.json" ] || "${BIN}hf" download lerobot/pi0 --local-dir "$RAW" >/dev/null
      mkdir -p "$PRETRAINED"
      ln -sf "$RAW/model.safetensors" "$PRETRAINED/model.safetensors"
      cp -f "$RAW"/policy_*.json "$PRETRAINED/"
      "${BIN}python" "$(dirname "$0")/../fix_pi0_config.py" "$RAW" "$PRETRAINED"
    fi
    # pi0 是 3B 量级，权重就 14 GB。全参微调下每卡还要放梯度与 Adam 双矩，
    # batch 必须比 SmolVLA 小一个档，且用 bf16。**先探再跑**：拿几十步看峰值显存，
    # 别直接开整轮 —— OOM 会在几小时之后才发生，白烧一晚上。
    BATCH="${BATCH:-2}"; LR="${LR:-2.5e-5}"; DTYPE="--policy.dtype=bfloat16"
    ;;
  *) echo "★ 只支持 smolvla / pi0，收到 $MODEL"; exit 1 ;;
esac

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="$GPUS"
NGPU=$(printf '%s' "$GPUS" | awk -F, '{print NF}')
# 代理不在这里写死：`http_proxy` 由机器的 /etc/environment 预置（AGENTS.md）。
# ⚠️ 非登录 shell 读不到它 —— 用 nohup/setsid 后台起时若拉不到 HF，在调用侧导出。

echo "== ① 从 HF 拉仿真数据（三个场景各一个子目录）=="
if [ ! -f "$SIM/cube40/meta/info.json" ]; then
  rm -rf "$SIM"
  "${BIN}hf" download "$REPO" --repo-type dataset --local-dir "$SIM" >/dev/null
fi
for s in cube40 cube20 cylinder40; do
  [ -f "$SIM/$s/meta/info.json" ] || { echo "★ 拉下来的仓里没有 $s"; exit 1; }
done

echo "== ② 三仿真 + 九真机，一次合成一份 =="
# **必须一次合完**：官方 merge 的产物不能再被 lerobot 的编辑工具处理
# （meta/episodes 指针会悬空），所以不能分两次合。
if [ ! -f "$DATA/meta/info.json" ]; then
  REAL_SETS=()
  for d in "$REAL"/*/; do
    [ -f "$d/meta/info.json" ] && REAL_SETS+=("${d%/}")
  done
  [ "${#REAL_SETS[@]}" -ge 9 ] || { echo "★ 只找到 ${#REAL_SETS[@]} 份真机数据，应有 9 份"; exit 1; }
  echo "  真机 ${#REAL_SETS[@]} 份 + 仿真 3 份"
  rm -rf "$DATA"
  "${BIN}python" "$(dirname "$0")/../../0_dataset_gen/merge_dataset.py" "$DATA" \
    "$SIM/cube40" "$SIM/cube20" "$SIM/cylinder40" "${REAL_SETS[@]}"
fi

# ★ 合完必须验一下真有数据。**源目录存在但为空时 rsync 会成功**，把镜像目录清空；
#   接着 lerobot 本地读不到 meta，就 `except (FileNotFoundError, NotADirectoryError)`
#   转去**按 repo_id 从 HF 拉另一份**，然后安安静静训满全程。这条路径是活的，撞到过。
ROOT="${FAST_ROOT-/dev/shm/so101-sft/sim-real-10task}"
if [ -n "$ROOT" ]; then
  # /work 这类网络挂载的大阵列随机小读很慢，整份搬进内存盘再训。内存盘随重启清空，
  # 正本留在 $DATA，这里只做幂等镜像。装不下就把 FAST_ROOT 设成空值直接读正本。
  mkdir -p "$(dirname "$ROOT")"
  rsync -a --delete "$DATA/" "$ROOT/"
else
  ROOT="$DATA"
fi
[ -f "$ROOT/meta/info.json" ] || {
  echo "★ $ROOT 里没有 meta/info.json —— 源目录 $DATA 是空的？"
  echo "  再往下走会从 HF 拉另一份数据训满全程，且不报错。"; exit 1; }
"${BIN}python" - "$ROOT" <<'PY'
import json
import sys
from pathlib import Path
d = json.loads((Path(sys.argv[1]) / "meta" / "info.json").read_text())
print(f"  训练数据：{d['total_episodes']} 集 / {d['total_frames']} 帧 / "
      f"{d['total_tasks']} 个任务 / fps={d['fps']}")
if d["total_episodes"] == 0:
    sys.exit("★ 集数为 0")
PY

# 图像增强逐字取自 `5_4_so101_real_sft/train_so101_real.sh` —— 那次真机 9 任务全参微调
# 用的就是这一组，理由是「数据出自别人的机位，光照和白平衡跟自己的臂必然不同」。
# 仿真渲染与真机图像之间是同一类差异、只会更大，所以照搬。
# ⚠️ `tfs` 是**整体替换**不是与默认合并，六个变换必须一次写全，漏写的会直接消失。
TFS='{
  "brightness":{"weight":1.0,"type":"ColorJitter","kwargs":{"brightness":[0.7,1.3]}},
  "contrast":{"weight":1.0,"type":"ColorJitter","kwargs":{"contrast":[0.7,1.3]}},
  "hue":{"weight":1.0,"type":"ColorJitter","kwargs":{"hue":[-0.05,0.05]}},
  "saturation":{"weight":1.0,"type":"ColorJitter","kwargs":{"saturation":[0.5,1.5]}},
  "sharpness":{"weight":1.0,"type":"SharpnessJitter","kwargs":{"sharpness":[0.5,1.5]}},
  "affine":{"weight":1.0,"type":"RandomAffine","kwargs":{"degrees":[-5.0,5.0],"translate":[0.05,0.05]}}
}'

echo "== ③ $NGPU 卡全量微调（$MODEL，每进程 batch $BATCH ⇒ effective $((BATCH * NGPU))）=="
# 四个 `--env.*` 里有三个「不给就安静跑错，表现为一个会被误读成策略没学会的低成功率」：
#   control_mode    数据集录的是绝对关节角。不给则用默认的归一化增量模式，
#                   绝对角被逐维 clip 到 ±1，手臂以包线最大速度朝错误方向走。
#   observation_*   lerobot 侧 EnvConfig 默认 128×128 且无条件下发 —— 不显式给会渲成
#                   正方形小图，而相机竖直视野角是在 640×480 下标定的，水平视野被压掉约四分之一。
#   fps             EnvConfig 默认 20，而仿真的 control_freq 与数据都是 30。
# rename_map：`smolvla_base` 按 camera1/2/3 预训练，本数据是 top/wrist 两路。特征校验
# 只要求数据提供的相机是权重期望相机的子集，所以缺 camera3 是允许的。
#
# 不开 `--eval.use_async_envs`：AsyncVectorEnv 默认 fork，而 ManiSkill 的每个 worker
# 都要自己的 CUDA 上下文，fork 出来的子进程不能再初始化 CUDA，实测直接 RuntimeError。
"${BIN}accelerate" launch --num_processes="$NGPU" --mixed_precision=bf16 \
  "${BIN}lerobot-train" \
  --policy.path="$PRETRAINED" \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --policy.optimizer_lr="$LR" \
  --policy.scheduler_warmup_steps=2000 \
  --policy.scheduler_decay_steps="${STEPS:-30000}" \
  $DTYPE \
  --dataset.repo_id=xbotics/so101-sim-real-10task \
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
  --batch_size="$BATCH" \
  --steps="${STEPS:-30000}" \
  --eval_freq="${EVAL_FREQ:-5000}" \
  --eval.n_episodes=10 \
  --eval.batch_size=5 \
  --save_freq=2500 \
  --log_freq=100 \
  --num_workers="${WORKERS:-8}" \
  --wandb.enable=true \
  --wandb.mode=online \
  --wandb.project="${WANDB_PROJECT:-so101}" \
  --wandb.disable_artifact=true \
  --job_name="full-sft-$MODEL" \
  --output_dir="$OUT" \
  "$@"

echo "TRAIN_FULL_SFT_DONE $MODEL -> $OUT"
echo "接着验收：bash $(dirname "$0")/eval_3scene.sh $OUT/checkpoints/last/pretrained_model 50"
