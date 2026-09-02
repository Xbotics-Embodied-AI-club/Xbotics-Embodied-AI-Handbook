#!/usr/bin/env bash
# 把一个场景的仿真数据集**用 lerobot-record 重录一遍**，产出与真机逐字同构的数据集。
#
# 为什么要重录而不是接着用已交付那份：已交付的四份出自已退役的手工转换路径，
# 视频编码是 mpeg4 而真机是 av1，而官方合并（aggregate 的 validate_all_metadata）
# **逐字比 features 字典、codec 在其中** ⇒ 仿真与真机根本合不了。重录之后编码由
# lerobot-record 的默认 libsvtav1 给出，与真机同源，features 不需要事后抹平。
#
# 三步，全部走官方命令：
#
#   ① recover_scene.py  逐集反解初始场景（物体/料箱摆回原处）—— 一个场景一个进程，
#                       因为建 GPU 环境要几秒而反解本身是毫秒级的 CPU 正运动学。
#   ② lerobot-record    每集一个进程，--resume 追加进同一个分片数据集。
#                       分片是为了并行：多个进程写**同一个**数据集会互相踩，
#                       所以一个分片一个数据集，最后再合。
#   ③ lerobot-edit-dataset --operation.type merge   把分片合成该场景的成品。
#
# 并行度：单集是「建环境 + 跑 ~400 步 + 渲两路 640×480 + 编码」。三种编码模式实测
# 都是 ~18.8 Hz（默认 / streaming_encoding / 加写图线程各 237 / 235 / 237 帧），
# 说明瓶颈在**仿真步进与渲染**、不在写图 ⇒ 按核数开分片能线性提总吞吐。
#
# ★`lerobot-record` 是**墙钟驱动**的：`episode_time_s` 到了就停，而仿真跑不到 30 Hz。
#   按 `帧数/30` 给窗口会把轨迹截在 63%（实测 377 帧的集只录到 237）。所以窗口按
#   **可达速率**给（`RATE_FLOOR`，默认 15 Hz，比实测 18.8 留两成余量）。
#   代价是尾部多出一段静止帧 —— 那时手臂已回 home、物体已在箱里，动作等于当前位置，
#   与源集自己的 settle 段同性质，是一致的数据。收尾会核对每集帧数 ≥ 源集帧数，
#   truncate 掉的集会被报出来而不是悄悄入库。
#
# 用法：bash regen_dataset.sh <场景> [<分片数>] [<可达速率>]
#       场景取 recipe.py 里的键：cube40 / cube20 / cylinder40
set -euo pipefail

SCENE="${1:?用法: bash regen_dataset.sh <场景> [<分片数>]}"
SHARDS="${2:-8}"
RATE_FLOOR="${3:-15.0}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="$HOME/.venv/Xbotics2-handbook/bin/python"
RECORD="$HOME/.venv/Xbotics2-handbook/bin/lerobot-record"
EDIT="$HOME/.venv/Xbotics2-handbook/bin/lerobot-edit-dataset"

read -r ENV_ID SOURCE_NAME TASK_TEXT <<EOF
$($PY -c "
import sys; sys.path.insert(0, '$HERE')
import recipe
s = recipe.SCENES['$SCENE']
print(s['env_id'], s['source_name'], s['task_text'].replace(' ', '_'))
")
EOF
TASK_TEXT="${TASK_TEXT//_/ }"

SRC="$HF_LEROBOT_HOME/so101_sim/$SOURCE_NAME"
WORK="$DATASETS_ROOT/datasets/private/so101_sim_regen"
STATES="$WORK/states/$SCENE"
OUT="$WORK/$SCENE"

echo "场景 $SCENE · 环境 $ENV_ID · 源 $SRC"

# ① 反解初始场景
$PY "$HERE/recover_scene.py" "$SCENE" "$SRC" "$STATES"

EPISODES=($(ls "$STATES"/ep*.json | xargs -n1 basename | sed 's/^ep//;s/\.json$//' | sort -n))
echo "待录 ${#EPISODES[@]} 集，分 $SHARDS 片"

# ② 分片并行录制。每片一个数据集，片内逐集 --resume 追加。
record_shard () {
  local shard=$1
  local repo="so101_sim_regen/${SCENE}_shard${shard}"
  local root="$OUT/shard${shard}"
  local first=1
  for i in $(seq "$shard" "$SHARDS" $((${#EPISODES[@]} - 1))); do
    local ep="${EPISODES[$i]}"
    local state="$STATES/ep${ep}.json"
    # 窗口 = 源集帧数 / 可达速率。用 30 会截断（见文件头）。
    local secs
    secs=$($PY -c "
import json; print(round(json.load(open('$STATES/meta/ep${ep}.json'))['n_frames'] / $RATE_FLOOR, 3))
")
    $RECORD \
      --robot.type=so101_sim \
      --robot.discover_packages_path=so101_sim \
      --robot.task="$ENV_ID" \
      --robot.episode_length=1200 \
      --robot.initial_state_path="$state" \
      --robot.state_log_path="$OUT/logs/${SCENE}-ep${ep}.npz" \
      --teleop.type=so101_dataset_player \
      --teleop.discover_packages_path=so101_sim \
      --teleop.repo_id="so101_sim/$SOURCE_NAME" \
      --teleop.root="$SRC" \
      --teleop.episode="$ep" \
      --dataset.repo_id="$repo" \
      --dataset.root="$root" \
      --dataset.single_task="$TASK_TEXT" \
      --dataset.num_episodes=1 \
      --dataset.episode_time_s="$secs" \
      --dataset.reset_time_s=0 \
      --dataset.push_to_hub=false \
      --resume=$([ $first -eq 1 ] && echo false || echo true) \
      --play_sounds=false >>"$OUT/logs/shard${shard}.log" 2>&1
    first=0
  done
  echo "  片 $shard 完成"
}

mkdir -p "$OUT/logs"
for s in $(seq 0 $((SHARDS - 1))); do
  record_shard "$s" &
done
wait

# ③ 核对：每集录到的帧数必须 ≥ 源集帧数，否则那一集的轨迹被墙钟截断了。
$PY "$HERE/check_regen.py" "$SCENE" "$STATES" "$OUT"

# ④ 官方 merge 收口
REPO_IDS=$(for s in $(seq 0 $((SHARDS - 1))); do printf "so101_sim_regen/%s_shard%s," "$SCENE" "$s"; done | sed 's/,$//')
ROOTS=$(for s in $(seq 0 $((SHARDS - 1))); do printf "%s/shard%s," "$OUT" "$s"; done | sed 's/,$//')
$EDIT --operation.type=merge \
  --operation.repo_ids="[$REPO_IDS]" \
  --operation.roots="[$ROOTS]" \
  --new_repo_id="so101_sim_regen/$SCENE" \
  --new_root="$OUT/merged"

echo "REGEN_DONE $SCENE -> $OUT/merged"
