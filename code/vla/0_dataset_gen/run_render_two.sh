#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
nvidia-smi -L | head -1
R=/mnt/nas_code/Xpersonal/Xbotics2/experiment_main_v1/results/EAI-exp-002-attempt-4
PY="$HOME/.venv/Xbotics2-handbook/bin/python"
$PY -u render_plans.py cube40 "$R/prep-cube40" "$R/cube40-v2-fail.mp4" 9,10,11,12,27,37 &
$PY -u render_plans.py cube40 "$R/prep-cube40" "$R/cube40-v2-good.mp4" 3,8,4,1,5,2,6,7,0,13 &
wait
echo RENDER_TWO_END
