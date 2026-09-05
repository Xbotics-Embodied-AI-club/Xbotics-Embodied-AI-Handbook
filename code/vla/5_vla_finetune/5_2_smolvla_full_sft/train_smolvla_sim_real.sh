#!/usr/bin/env bash
# w4 上的一条龙：从 HF 拉仿真数据 → 与真机合成一份训练集 → 六卡全量微调 SmolVLA。
#
# 为什么合并放在 w4 而不是 w1：真机数据在 ModelScope、不进我们的 HF 仓，而 w4 本地
# 真机 9 个任务在数据根下本来就齐。在这边合，既不用把别人的数据
# 重新分发一遍，也少一次几百 MB 的跨机搬运。
#
# ★ **全量微调是显式开的。** `lerobot/smolvla_base` 的 config 里
#   `freeze_vision_encoder: true` / `train_expert_only: true`，lerobot 的默认值也是这两个 ——
#   不显式覆盖就只训 action expert（实测 99,880,992 / 450,046,176 ≈ 22%），视觉塔全程冻着。
#   attempt-1/2/3 名义上的"全量微调"都是这么跑的。而这个实验要跨的正是仿真渲染与
#   真机图像的域差，冻着视觉塔等于让 action expert 硬扛。
#
# ★ 解冻之后**学习率必须大降**。3.4e-4 是按 action expert（1 亿参数、随机初始化的头）
#   定的；对预训练好的 SigLIP + SmolLM2 那是灾难性的大。`get_optim_params` 返回的是
#   扁平的 `self.parameters()`，没有分组口子 ⇒ 只能整体取一个折中值。
#   取 5e-5：VLA 全参数微调的常用档，且仍高于纯视觉塔微调的 1e-5 量级。
#
# 用法：bash w4_build_and_train.sh [<HF 仓名>] [额外的 lerobot-train 参数]
set -euo pipefail

REPO="${1:-Harrysunshine/so101-sim-pickplace-v2}"
shift || true
# 数据与产物的根。沿用同目录 `train_so101_real.sh` 的约定：DATASETS_ROOT 由环境给出。
BASE="${BASE:-$DATASETS_ROOT/so101-sft}"
# 指定 VENV 就用它 bin 下的命令，不指定就用 PATH 上的（`uv sync` 之后即在 PATH）。
BIN="${VENV:+$VENV/bin/}"
SIM=$BASE/datasets/sim-hf
# 真机 9 个任务的原始数据根。公开出处见 2.4 节末尾的 ModelScope 链接。
REAL="${REAL:-$DATASETS_ROOT/so101/datasets/raw}"
OUT=$BASE/datasets/sim-real-10task

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 代理不在这里写死：`http_proxy` 由机器的 /etc/environment 预置（AGENTS.md）。
# ⚠️ 非登录 shell 读不到 /etc/environment —— 用 nohup/setsid 后台起时若拉不到 HF，
#    在**调用这个脚本的那一侧**导出，不要把某台机器的代理地址钉进脚本。

echo "== ① 从 HF 拉仿真数据（三个场景各一个子目录）=="
rm -rf "$SIM"
"${BIN}hf" download "$REPO" --repo-type dataset --local-dir "$SIM" >/dev/null
for s in cube40 cube20 cylinder40; do
  [ -f "$SIM/$s/meta/info.json" ] || { echo "★ 拉下来的仓里没有 $s"; exit 1; }
done

echo "== ② 三仿真 + 九真机（全部），一次合成一份 =="
# **全部混在一起**（用户 2026-09-05 定）：两个任务两边都有（cube / can），
# 仿真独有 small cube，真机独有另外七个 —— 共 12 份、10 个不同任务。
# 与真机那次 9 任务全参微调（handbook 的 5_4_so101_real_sft）同一个思路：
# 混进来的任务越多，视觉与语言两侧的覆盖越宽。
#
# **必须一次合完**：官方 merge 的产物不能再被 lerobot 的编辑工具处理
# （meta/episodes 指针会悬空），所以不能分两次合。
rm -rf "$OUT"
REAL_SETS=()
for d in "$REAL"/*/; do
  [ -f "$d/meta/info.json" ] && REAL_SETS+=("${d%/}")
done
[ "${#REAL_SETS[@]}" -ge 9 ] || { echo "★ 只找到 ${#REAL_SETS[@]} 份真机数据，应有 9 份"; exit 1; }
echo "  真机 ${#REAL_SETS[@]} 份 + 仿真 3 份"
"${BIN}python" "$(dirname "$0")/../../0_dataset_gen/merge_dataset.py" "$OUT" \
  "$SIM/cube40" "$SIM/cube20" "$SIM/cylinder40" "${REAL_SETS[@]}"

# 图像增强逐字取自 handbook 的 `5_4_so101_real_sft/train_so101_real.sh` —— 那次真机
# 9 任务全参微调用的就是这一组，理由是「数据出自别人的机位，光照和白平衡跟自己的臂
# 必然不同」。仿真渲染与真机图像之间是同一类差异、只会更大，所以照搬。
# ⚠️ `tfs` 是**整体替换**不是与默认合并，六个变换必须一次写全，漏写的会直接消失。
TFS='{
  "brightness":{"weight":1.0,"type":"ColorJitter","kwargs":{"brightness":[0.7,1.3]}},
  "contrast":{"weight":1.0,"type":"ColorJitter","kwargs":{"contrast":[0.7,1.3]}},
  "hue":{"weight":1.0,"type":"ColorJitter","kwargs":{"hue":[-0.05,0.05]}},
  "saturation":{"weight":1.0,"type":"ColorJitter","kwargs":{"saturation":[0.5,1.5]}},
  "sharpness":{"weight":1.0,"type":"SharpnessJitter","kwargs":{"sharpness":[0.5,1.5]}},
  "affine":{"weight":1.0,"type":"RandomAffine","kwargs":{"degrees":[-5.0,5.0],"translate":[0.05,0.05]}}
}'

echo "== ③ 六卡全量微调 =="
# batch 是**每进程**的（accelerate 默认 split_batches=False，上游 lerobot_train.py:352
# 自己也这么算）⇒ effective = 16 × 6 = 96。
# 16 是探针实测出来的上限档：那六张卡各被一个杀不掉的残留进程占着 22~38GB，
# 只剩约 42GB；全解冻后可训参数 403M（此前 100M），batch 64 直接 OOM。
# 步数 30000 与 warmup 2000 取自真机那次 9 任务全参微调，保持可比。
# 128 万帧 / 96 ⇒ 13338 步一个 epoch，30000 步约 2.25 epoch。
GPUS=2,3,4,5,6,7 \
DATASET_NAME=sim-real-10task \
DATASET_REPO=xbotics/so101-sim-real-10task \
ATTEMPT=5 \
VENV="$VENV" \
bash "$BASE/scripts/train_smolvla.sh" \
  --batch_size=16 \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --policy.optimizer_lr=5e-5 \
  --policy.scheduler_decay_steps=30000 \
  --steps=30000 \
  --eval_freq=5000 \
  --save_freq=2500 \
  --eval.n_episodes=10 \
  --eval.batch_size=5 \
  "$@"

echo "W4_BUILD_AND_TRAIN_DONE"
