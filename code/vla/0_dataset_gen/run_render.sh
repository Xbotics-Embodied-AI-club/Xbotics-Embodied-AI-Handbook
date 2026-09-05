#!/usr/bin/env bash
# 把准备好的规划渲成一段目审视频。落点目录由调用方给（见 run_prep_check.sh 的说明）。
#
# 用法：RESULT_DIR=<绝对路径> bash run_render.sh <场景> <输出文件名> [<集号,逗号分隔>]
set -euo pipefail
cd "$(dirname "$0")"
R="${RESULT_DIR:?缺 RESULT_DIR —— 本轮 attempt 目录由调用方从 run_id.py 取}"
SCENE="${1:?缺场景}"
NAME="${2:?缺输出文件名}"
EPS="${3:-}"
"$HOME/.venv/Xbotics2-handbook/bin/python" -u render_plans.py "$SCENE" "$R/prep-$SCENE" \
  "$R/$NAME" ${EPS:+"$EPS"}
