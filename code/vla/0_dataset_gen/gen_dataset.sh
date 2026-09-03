#!/usr/bin/env bash
# 把 `prepare_episodes.py` 备好的规划逐集录成一份 LeRobotDataset，全程走官方 `lerobot-record`。
#
# 与 `regen_dataset.sh` 的区别只有动作来源：那个回放已交付数据集（`--teleop.episode`），
# 这个播脚本化专家的规划（`--teleop.actions_path`）。其余标志逐字相同 —— 编码、字段名、
# 单位、帧率因此没有第二个实现，也就无处走偏。
#
# ★ 一个进程只录一集：`lerobot-record` 按墙钟停表，而一次调用只认一个窗口值，
#   每集的窗口又必须按各自的规划长度给。
#
# ★ 窗口 = (规划帧数 + 热身补偿) / 30。热身补偿按机器给：循环第一帧要建管线、明显慢于
#   33.3ms，于是每集固定少那么几帧 —— 那是**开销**不是随机误差，重录救不了。
#   实测 RTX PRO 6000 少 1 帧。判据不受它影响：合格与否始终按「与规划差 ≤
#   recipe.FRAME_TOLERANCE 帧」判，那条与机器无关。
#
# ★ 每录完一集都确认数据集**真的多出一集**：`lerobot-record` 退出码为 0 却什么也没写出来
#   时，只看退出码是看不出来的。
#
# 用法：bash gen_dataset.sh <场景> <准备目录> <输出目录> [<热身补偿帧数>]
set -euo pipefail

SCENE="${1:?用法: bash gen_dataset.sh <场景> <准备目录> <输出目录> [<热身补偿帧数>]}"
PREP="${2:?缺准备目录}"
OUT="${3:?缺输出目录}"
WARMUP="${4:-1}"
FPS=30
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="$HOME/.venv/Xbotics2-handbook/bin/python"
RECORD="$HOME/.venv/Xbotics2-handbook/bin/lerobot-record"

# 逐行读，不做空格↔下划线的往返编码 —— 那种编码遇到本来就带下划线的取值会静默改写它。
{
  read -r ENV_ID
  read -r TASK_TEXT
  read -r VCODEC
  read -r TOLERANCE
} < <(cd "$HERE" && $PY -c "
import recipe
s = recipe.SCENES['$SCENE']
print(s['env_id']); print(s['task_text']); print(recipe.VIDEO_CODEC); print(recipe.FRAME_TOLERANCE)
")

N=$(find "$PREP/plans" -name 'ep*.npy' | wc -l)
[ "$N" -gt 0 ] || { echo "★ $PREP/plans 里没有规划文件"; exit 1; }
# 日志与账本放在数据集**外面**：lerobot 用 `exist_ok=False` 建数据集根目录，
# 提前在里面建 logs/ 会让第一集直接 FileExistsError。
LOGS="${OUT}-logs"
mkdir -p "$LOGS"
REPO="so101_expert/$SCENE"
ORDER="$LOGS/order"
: > "$ORDER"
echo "  $SCENE：$N 集待录，窗口按各自规划长度给，热身补偿 $WARMUP 帧"

first=1
for ep in $(seq 0 $((N - 1))); do
  want=$($PY -c "import json; print(json.load(open('$PREP/meta/ep${ep}.json'))['n_frames'])")
  secs=$($PY -c "print(round(($want + $WARMUP) / $FPS, 4))")
  $RECORD \
    --robot.type=so101_sim \
    --robot.discover_packages_path=so101_sim \
    --robot.task="$ENV_ID" \
    --robot.episode_length=1200 \
    --robot.initial_state_path="$PREP/states/ep${ep}.json" \
    --teleop.type=so101_dataset_player \
    --teleop.discover_packages_path=so101_sim \
    --teleop.actions_path="$PREP/plans/ep${ep}.npy" \
    --dataset.repo_id="$REPO" \
    --dataset.root="$OUT" \
    --dataset.single_task="$TASK_TEXT" \
    --dataset.vcodec="$VCODEC" \
    --dataset.streaming_encoding=true \
    --dataset.encoder_threads=2 \
    --dataset.num_episodes=1 \
    --dataset.episode_time_s="$secs" \
    --dataset.reset_time_s=0 \
    --dataset.push_to_hub=false \
    --resume=$([ $first -eq 1 ] && echo false || echo true) \
    --play_sounds=false >>"$LOGS/record.log" 2>&1
  first=0

  # 记账之前先确认真的多出一集 —— 否则账与数据会一起少，两边"自洽"地错下去。
  recorded=$(find "$OUT/data" -name '*.parquet' | wc -l)
  expect=$(( $(wc -l < "$ORDER") + 1 ))
  [ "$recorded" -eq "$expect" ] || {
    echo "★ 录 ep${ep} 之后数据集里有 $recorded 集，应当是 $expect 集"; exit 1; }

  got=$($PY -c "
import pyarrow.parquet as pq, pathlib
f = sorted(pathlib.Path('$OUT/data').rglob('*.parquet'))[-1]
print(pq.read_table(f, columns=['episode_index']).num_rows)")
  delta=$((got - want))
  verdict=$([ "${delta#-}" -le "$TOLERANCE" ] && echo ok || echo reject)
  echo "$ep $got $want $delta $verdict" >> "$ORDER"
  echo "    ep${ep}: 录到 $got 帧 / 规划 $want 帧（差 $delta，容差 $TOLERANCE）$verdict"
done

BAD=$(awk '$5 == "reject"' "$ORDER" | wc -l)
echo "  $SCENE：$N 集录完，超差 $BAD 集（账本 $ORDER）"
echo "GEN_DATASET_DONE $SCENE -> $OUT"
