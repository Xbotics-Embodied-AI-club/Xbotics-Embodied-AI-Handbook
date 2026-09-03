#!/usr/bin/env bash
# 把一个场景的仿真数据集用 lerobot-record 重录一遍，产出与真机逐字同构的数据集。
#
# 已交付那四份出自已退役的手工转换路径，视频编码是 mpeg4 而真机实测是 h264；官方合并
# 逐字比 features 字典、`video.codec` 在其中 ⇒ 合不了。重录时按 `recipe.VIDEO_CODEC`
# 显式声明编码，features 就不需要事后抹平。
#
# 三步，全部走官方命令：
#
#   ① recover_scene.py  逐集反解初始场景（物体/料箱摆回原处）。一个场景一个进程 ——
#                       建环境要几秒，而反解本身是毫秒级的 CPU 正运动学。
#   ② lerobot-record    每集一个进程，`--resume` 追加进同一个分片数据集。
#   ③ check_regen.py    保真核对：录到的动作要完整包含源轨迹。
#
# ★ **本脚本到分片为止，不合并。** 合并要等跨后端门筛完、不合格的集逐片删掉之后再做，
#   由 `close_scene.sh` 收口：官方 merge 的产物 `meta/episodes` 指针会大面积悬空
#   （实测 243/344），合过之后任何 lerobot 数据编辑工具都不能再用 ⇒ 删必须在合并之前。
#
# ★ 一个进程只能录一集：`lerobot-record` 按墙钟停表，而一次调用只认一个窗口值，
#   每集的窗口又必须按各自的源集长度给。
#
# ★ 分片是为了并行，一个分片一个数据集 —— 多个进程写同一个数据集会互相踩。
#   **分片数受"循环还能不能跑满 30 Hz"限制**：跑不满就少帧，少过容差那一集就得重录。
#   实测单分片只差 1 帧（开 streaming 编码）或 2 帧（不开），8 分片时循环掉到 22.7 Hz。
#   每片用 `CUDA_VISIBLE_DEVICES` 绑一张卡。
#
# ★ 开 `--dataset.streaming_encoding` 是为了把每帧两张 640×480 的 PNG 落盘那段开销拿掉：
#   它直接编码，不再先写图再事后编。实测同一集少的帧数从 2 帧降到 1 帧。
#
# ★ 后端固定 CPU PhysX（`SO101SimRobotConfig.sim_backend` 的默认值）：GPU PhysX
#   单环境每控制步 53ms，跑不到 30 Hz，墙钟窗口会把轨迹截掉三分之一。
#
# ★ 源集号 ↔ 数据集内集号的账记在 `logs/shard<N>.order`，每录成一集追加一行。
#   `--resume` 按录制顺序发号（0,1,2,…），而分片是隔片取集（0,2,4,… / 1,3,5,…），
#   两者不相等。没有这本账，跨后端门会拿错集的动作去复跑、`delete_episodes` 会删错集，
#   而这两件事都不报错。
#
# ★ 第四个参数是**热身补偿帧数**，按机器给。窗口本该正好是 `源帧数/30`，但循环第一帧要
#   建管线、明显慢于 33.3ms，于是每集固定少那么几帧 —— 这是**开销**不是随机误差，
#   重录救不了。实测：RTX PRO 6000 少 1 帧，A800 少 6 帧（最差 11）。
#   把这几帧补回窗口，帧数就落回源集上。**判据不受它影响**：合格与否始终按
#   「与源集差 ≤ recipe.FRAME_TOLERANCE 帧且动作逐位相同」判，那条与机器无关。
#
# 用法：bash regen_dataset.sh <场景> [<分片数>] [<可用 GPU 列表>] [<热身补偿帧数>]
#       场景取 recipe.py 里的键：cube40 / cube20 / cylinder40
#       GPU 列表逗号分隔，如 `0,1,2,3`；不给则全用 0 号卡
set -euo pipefail

SCENE="${1:?用法: bash regen_dataset.sh <场景> [<分片数>] [<GPU 列表>] [<热身补偿帧数>]}"
SHARDS="${2:-2}"
# 每片绑一张卡：渲染是分片之间唯一争用的资源。
GPUS="${3:-0}"
# 热身补偿：循环第一帧建管线的固定开销，按机器实测给。见文件头。
WARMUP="${4:-0}"
IFS=',' read -r -a GPU_LIST <<< "$GPUS"
# 数据集的帧率。窗口按它算。
FPS=30
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="$HOME/.venv/Xbotics2-handbook/bin/python"
RECORD="$HOME/.venv/Xbotics2-handbook/bin/lerobot-record"

# 逐行读，不做空格↔下划线的往返编码 —— 那种编码遇到本来就带下划线的取值会静默改写它。
{
  read -r ENV_ID
  read -r SOURCE_NAME
  read -r TASK_TEXT
  read -r VCODEC
  read -r TOLERANCE
  read -r MAX_RETRIES
} < <($PY -c "
import sys; sys.path.insert(0, '$HERE')
import recipe
s = recipe.SCENES['$SCENE']
print(s['env_id']); print(s['source_name']); print(s['task_text'])
print(recipe.VIDEO_CODEC); print(recipe.FRAME_TOLERANCE); print(recipe.MAX_RECORD_RETRIES)
")

# `HF_LEROBOT_HOME` 只在交互 shell 里由 direnv 注入，ssh 过来的非交互 shell 没有它。
# 按 `.envrc` 声明的关系从 `HF_HOME` 推，不写死路径。
LEROBOT_HOME="${HF_LEROBOT_HOME:-${HF_HOME:?需要 HF_HOME 或 HF_LEROBOT_HOME}/lerobot}"
SRC="$LEROBOT_HOME/so101_sim/$SOURCE_NAME"
WORK="${DATASETS_ROOT:?需要 DATASETS_ROOT}/datasets/private/so101_sim_regen"
STATES="$WORK/states/$SCENE"
OUT="$WORK/$SCENE"

echo "场景 $SCENE · 环境 $ENV_ID · 源 $SRC · 编码 $VCODEC · 容差 $TOLERANCE 帧 · 最多录 $MAX_RETRIES 次"

# 半截的旧产物必须由人清掉，不能续着录：`--resume` 会把新集追加在旧集后面，于是
# 一个数据集里混着两次运行的产物，而 `logs/shard<N>.order` 那本账只对得上其中一次。
for s in $(seq 0 $((SHARDS - 1))); do
  if [ -e "$OUT/shard${s}" ]; then
    echo "★ $OUT/shard${s} 已存在。先删掉它再录 —— 续录会让集号与 shard${s}.order 对不上。" >&2
    exit 1
  fi
done

# ① 反解初始场景
$PY "$HERE/recover_scene.py" "$SCENE" "$SRC" "$STATES"

mapfile -t EPISODES < <(find "$STATES" -maxdepth 1 -name 'ep*.json' -printf '%f\n' \
  | sed 's/^ep//;s/\.json$//' | sort -n)
echo "待录 ${#EPISODES[@]} 集，分 $SHARDS 片"

# ② 分片并行录制。每片一个数据集，片内逐集 --resume 追加。
record_shard () {
  local shard=$1
  # 轮流分配 GPU；片数多于卡数时同一张卡上会落多片。
  local gpu="${GPU_LIST[$((shard % ${#GPU_LIST[@]}))]}"
  local repo="so101_sim_regen/${SCENE}_shard${shard}"
  local root="$OUT/shard${shard}"
  local order="$OUT/logs/shard${shard}.order"
  local first=1
  : > "$order"
  for i in $(seq "$shard" "$SHARDS" $((${#EPISODES[@]} - 1))); do
    local ep="${EPISODES[$i]}"
    # 窗口 = `(源帧数 + 热身补偿)/30`。补偿只抵消循环第一帧的固定开销，不是"留余量"——
    # 留余量会多出一串静止帧，把末尾静止段撑长（实测余量 90 帧时撑到真机的四倍）。
    local want secs
    want=$($PY -c "
import json; print(json.load(open('$STATES/meta/ep${ep}.json'))['n_frames'])")
    secs=$($PY -c "print(round(($want + $WARMUP) / $FPS, 4))")
    local attempt=1
    while : ; do
      CUDA_VISIBLE_DEVICES="$gpu" $RECORD \
        --robot.type=so101_sim \
        --robot.discover_packages_path=so101_sim \
        --robot.task="$ENV_ID" \
        --robot.episode_length=1200 \
        --robot.initial_state_path="$STATES/ep${ep}.json" \
        --robot.state_log_path="$OUT/logs/${SCENE}-ep${ep}.npz" \
        --teleop.type=so101_dataset_player \
        --teleop.discover_packages_path=so101_sim \
        --teleop.repo_id="so101_sim/$SOURCE_NAME" \
        --teleop.root="$SRC" \
        --teleop.episode="$ep" \
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
        echo "★ 片 $shard 录源集 ep${ep} 之后数据集里有 $recorded 集，应当是 $expect 集" >&2
        exit 1
      fi
      # 刚写出来的那一集就是最后一个 parquet；比它的帧数与源集差多少。
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
      # 收口时用官方 delete_episodes 一次删干净，然后重录一遍。
      echo "$ep reject" >> "$order"
      echo "  片 $shard 源集 ep${ep} 第 $attempt 次录到 $got 帧（源 $want，容差 $TOLERANCE）" \
        >> "$OUT/logs/shard${shard}.retry"
      attempt=$((attempt + 1))
      if [ "$attempt" -gt "$MAX_RETRIES" ]; then
        echo "  片 $shard 源集 ep${ep} 录 $MAX_RETRIES 次都超差，放弃" >> "$OUT/logs/shard${shard}.retry"
        break
      fi
    done
  done
  echo "  片 $shard（GPU $gpu）写入 $(wc -l < "$order") 集，其中合格 $(grep -c ' ok$' "$order" || true) 集"
}

mkdir -p "$OUT/logs"
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

# ③ 保真核对：录到的动作要完整包含源轨迹。任务成没成由 verify_cross_backend.py 判 ——
#    那件事要在两个物理后端各跑一遍，不属于录制收尾。
$PY "$HERE/check_regen.py" "$SCENE" "$OUT" "$SRC" "$SHARDS"

echo "REGEN_DONE $SCENE -> $OUT/shard0..$((SHARDS - 1))（合并交给 close_scene.sh，删在合并之前）"
