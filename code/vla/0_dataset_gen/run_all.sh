#!/usr/bin/env bash
# 开发环：装环境 → 重备一批规划并判成败 → 渲一段目审视频。
#
# 这是**改了产线之后自查用的**短链，不是产出数据集的那条 —— 那条是 `build_dataset.sh`。
# 两者的区别只有一个：这里不录制，直接拿规划在仿真里复跑，快得多，适合逐个改动验证。
#
# 用法：RESULT_DIR=<绝对路径> bash run_all.sh [<集数>] [<场景>]
set -euo pipefail
D="$(cd "$(dirname "$0")" && pwd)"
COUNT="${1:-40}"
SCENE="${2:-cube40}"
bash "$D/run_sync.sh"
bash "$D/run_prep_check.sh" "$COUNT" "$SCENE"
bash "$D/run_render.sh" "$SCENE" "$SCENE-review.mp4"
echo ALL_END
