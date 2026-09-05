#!/usr/bin/env bash
# 把一个场景从零建成一份与真机同口径的 LeRobotDataset。这是本讲**唯一**的产线入口。
#
# 四步，动作只来自脚本化专家，落盘只经官方命令：
#
#   ① prepare_episodes.py  撒场景 + 规划动作（逐集一份强制初始状态与 .npy）
#   ② check_plans.py       **录之前**先把每集在仿真里复跑一遍判成败，一集一个进程并行
#   ③ compact_prep.py      只留判成功的，重新编号 0..N−1
#   ④ lerobot-record       每集一个进程，分片并行，`--resume` 追加进同一个分片数据集
#
# ★ 为什么筛在录之前：不合格的集一旦进了数据集就只能事后 `delete_episodes`，而官方
#   merge 之后集号指针会大面积悬空（实测 243/344），删必须赶在合并之前 —— 链条很脆。
#   ②用的是与录制**同一个后端、同一份强制初始状态**，判定与录制结果一致，不是近似。
#
# ★ 本脚本到分片为止，**不合并**。合并交给 close_scene.sh 收口。
#
# ★ 一个进程只能录一集：`lerobot-record` 按墙钟停表，而一次调用只认一个窗口值，
#   每集的窗口又必须按各自的规划长度给。
#
# ★ 分片是为了并行，一个分片一个数据集 —— 多个进程写同一个数据集会互相踩。
#   分片数受"循环还能不能跑满 30 Hz"限制：跑不满就少帧。实测单分片只差 1 帧
#   （开 streaming 编码），8 分片时循环掉到 22.7 Hz。每片用 CUDA_VISIBLE_DEVICES 绑一张卡。
#
# ★ 第五个参数是**热身补偿帧数**，按机器给。窗口本该正好是 `规划帧数/30`，但循环第一帧要
#   建管线、明显慢于 33.3ms，于是每集固定少那么几帧 —— 这是**开销**不是随机误差，
#   重录救不了。实测：RTX PRO 6000 少 1 帧，A800 少 6 帧（最差 11）。
#   判据不受它影响：合格与否始终按「与规划差 ≤ recipe.FRAME_TOLERANCE 帧」判。
#
# 用法：bash build_dataset.sh <场景> <要几集> [<分片数>] [<GPU 列表>] [<热身补偿帧数>]
#       场景取 recipe.py 里的键：cube40 / cube20 / cylinder40
set -euo pipefail

SCENE="${1:?用法: bash build_dataset.sh <场景> <要几集> [<分片数>] [<GPU 列表>] [<热身补偿>]}"
WANT="${2:?缺要几集}"
SHARDS="${3:-2}"
GPUS="${4:-0}"
WARMUP="${5:-1}"
IFS=',' read -r -a GPU_LIST <<< "$GPUS"
FPS=30
# 准备多少个槽位才够 WANT 集：规划门与成功门各自会筛掉一些。实测 cube40 是
# 44 个种子 → 40 集规划 → 37 集过成功门，即 0.84。留到 1.35 倍，宁可多备。
OVERSAMPLE=135
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="$HOME/.venv/Xbotics2-handbook/bin/python"
RECORD="$HOME/.venv/Xbotics2-handbook/bin/lerobot-record"

# 逐行读，不做空格↔下划线的往返编码 —— 那种编码遇到本来就带下划线的取值会静默改写它。
{
  read -r ENV_ID
  read -r TASK_TEXT
  read -r VCODEC
  read -r TOLERANCE
  read -r MAX_RETRIES
} < <($PY -c "
import sys; sys.path.insert(0, '$HERE')
import recipe
s = recipe.SCENES['$SCENE']
print(s['env_id']); print(s['task_text'])
print(recipe.VIDEO_CODEC); print(recipe.FRAME_TOLERANCE); print(recipe.MAX_RECORD_RETRIES)
")

WORK="${DATASETS_ROOT:?需要 DATASETS_ROOT}/datasets/private/so101_sim_gen/$SCENE"
STAGE="$WORK/staged"
PREP="$WORK/prep"
OUT="$WORK/shards"

echo "场景 $SCENE · 环境 $ENV_ID · 要 $WANT 集 · 编码 $VCODEC · 容差 $TOLERANCE 帧"

# 半截的旧产物必须先清掉，不能续着录：`--resume` 会把新集追加在旧集后面，于是一个
# 数据集里混着两次运行的产物，而 shard<N>.order 那本账只对得上其中一次。
for s in $(seq 0 $((SHARDS - 1))); do
  if [ -e "$OUT/shard${s}" ]; then
    echo "★ $OUT/shard${s} 已存在。先删掉它再录 —— 续录会让集号与 shard${s}.order 对不上。" >&2
    exit 1
  fi
done
mkdir -p "$OUT/logs"

# ① 规划
rm -rf "$STAGE"
$PY -u "$HERE/prepare_episodes.py" "$SCENE" "$STAGE" $((WANT * OVERSAMPLE / 100)) 0

# ② 成功门：一集一个进程并行复跑。并行度封顶 12 —— 每个进程一条 CPU PhysX 单线程。
N=$(find "$STAGE/plans" -name 'ep*.npy' | wc -l)
VERDICTS="$OUT/logs/screen.txt"
seq 0 $((N - 1)) | xargs -P 12 -I{} $PY -u "$HERE/check_plans.py" "$SCENE" "$STAGE" {} \
  > "$VERDICTS" 2>&1

# ③ 只留过门的，重新编号
$PY -u "$HERE/compact_prep.py" "$STAGE" "$PREP" "$VERDICTS" "$WANT"

mapfile -t EPISODES < <(find "$PREP/plans" -maxdepth 1 -name 'ep*.npy' -printf '%f\n' \
  | sed 's/^ep//;s/\.npy$//' | sort -n)
echo "待录 ${#EPISODES[@]} 集，分 $SHARDS 片"

# ④ 分片并行录制
record_shard () {
  local shard=$1
  local gpu="${GPU_LIST[$((shard % ${#GPU_LIST[@]}))]}"
  local repo="so101_sim/${SCENE}_shard${shard}"
  local root="$OUT/shard${shard}"
  local order="$OUT/logs/shard${shard}.order"
  local first=1
  : > "$order"
  for i in $(seq "$shard" "$SHARDS" $((${#EPISODES[@]} - 1))); do
    local ep="${EPISODES[$i]}"
    local want secs attempt
    want=$($PY -c "
import json; print(json.load(open('$PREP/meta/ep${ep}.json'))['n_frames'])")
    secs=$($PY -c "print(round(($want + $WARMUP) / $FPS, 4))")
    attempt=1
    while : ; do
      CUDA_VISIBLE_DEVICES="$gpu" $RECORD \
        --robot.type=so101_sim \
        --robot.discover_packages_path=so101_sim \
        --robot.task="$ENV_ID" \
        --robot.episode_length=1200 \
        --robot.initial_state_path="$PREP/states/ep${ep}.json" \
        --teleop.type=so101_dataset_player \
        --teleop.discover_packages_path=so101_sim \
        --teleop.actions_path="$PREP/plans/ep${ep}.npy" \
        --dataset.repo_id="$repo" \
        --dataset.root="$root" \
        --dataset.single_task="$TASK_TEXT" \
        --dataset.vcodec="$VCODEC" \
        --dataset.streaming_encoding=true \
        --dataset.encoder_threads=2 \
        --dataset.num_episodes=1 \
        --dataset.episode_time_s="$secs" \
        --dataset.reset_time_s=0 \
        --dataset.push_to_hub=false \
        --resume=$([ $first -eq 1 ] && echo false || echo true) \
        --play_sounds=false >>"$OUT/logs/shard${shard}.log" 2>&1
      first=0
      # 记账之前先确认**真的多出一集**：`lerobot-record` 退出码为 0 却什么也没写出来时，
      # 只对账是看不出来的 —— 账和数据会一起少，两边"自洽"地错下去。
      local recorded expect got
      recorded=$(find "$root/data" -name '*.parquet' | wc -l)
      expect=$(( $(wc -l < "$order") + 1 ))
      if [ "$recorded" -ne "$expect" ]; then
        echo "★ 片 $shard 录 ep${ep} 之后数据集里有 $recorded 集，应当是 $expect 集" >&2
        exit 1
      fi
      got=$($PY -c "
import pyarrow.parquet as pq, pathlib
files = sorted(pathlib.Path('$root').glob('data/**/*.parquet'))
print(pq.read_table(files[-1], columns=['frame_index']).num_rows)")
      if [ "$got" -le $((want + TOLERANCE)) ] && [ "$got" -ge $((want - TOLERANCE)) ]; then
        # 行号（从 0 起）就是这一集在本分片里的 episode_index。
        echo "$ep ok" >> "$order"
        break
      fi
      # 超差的那一集已经写进数据集了，删不掉也不该原地改 —— 记成 reject 占住它的集号，
      # 收口时用官方 delete_episodes 一次删干净。
      echo "$ep reject" >> "$order"
      echo "  片 $shard ep${ep} 第 $attempt 次录到 $got 帧（规划 $want，容差 $TOLERANCE）" \
        >> "$OUT/logs/shard${shard}.retry"
      attempt=$((attempt + 1))
      if [ "$attempt" -gt "$MAX_RETRIES" ]; then
        echo "  片 $shard ep${ep} 录 $MAX_RETRIES 次都超差，放弃" >> "$OUT/logs/shard${shard}.retry"
        break
      fi
    done
  done
  echo "  片 $shard（GPU $gpu）写入 $(wc -l < "$order") 集，其中合格 $(grep -c ' ok$' "$order" || true) 集"
}

PIDS=()
for s in $(seq 0 $((SHARDS - 1))); do
  record_shard "$s" &
  PIDS+=($!)
done
# 逐个 pid 等，不用不带参数的 `wait` —— 后者恒返回 0，某一片失败会被当成全部成功，
# 于是接着拿一份缺集的数据去合并，而缺了哪几集只有对账才看得出来。
for pid in "${PIDS[@]}"; do
  wait "$pid"
done

# ⑤ 保真核对：录到的动作要与规划逐位相同。
$PY -u "$HERE/check_recorded.py" "$SCENE" "$OUT" "$PREP" "$SHARDS"

echo "BUILD_DONE $SCENE -> $OUT/shard0..$((SHARDS - 1))（合并交给 close_scene.sh）"
