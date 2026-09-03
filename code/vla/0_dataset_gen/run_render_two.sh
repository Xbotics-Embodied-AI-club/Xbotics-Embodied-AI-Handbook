#!/usr/bin/env bash
# 并行渲两支：失败集与成功集抽样。渲染单进程逐帧编码，两支同时跑省一半墙钟（bd xb-nqqr）。
set -euo pipefail
cd "$(dirname "$0")"
nvidia-smi -L | head -1
R=/mnt/nas_code/Xpersonal/Xbotics2/experiment_main_v1/results/EAI-exp-002-attempt-4
PY="$HOME/.venv/Xbotics2-handbook/bin/python"
$PY -u render_plans.py cube40 "$R/prep-cube40" "$R/cube40-n40-fail.mp4" 10,0,9,12,15,18,28,34,39 &
$PY -u render_plans.py cube40 "$R/prep-cube40" "$R/cube40-n40-good.mp4" 3,1,8,6,5,11,4,7,2,13 &
wait
echo RENDER_TWO_END
