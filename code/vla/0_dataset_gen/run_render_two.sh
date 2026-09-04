#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
nvidia-smi -L | head -1
R=/mnt/nas_code/Xpersonal/Xbotics2/experiment_main_v1/results/EAI-exp-002-attempt-4
PY="$HOME/.venv/Xbotics2-handbook/bin/python"
$PY -u render_plans.py cube40 "$R/prep-cube40" "$R/cube40-v3-fail.mp4" 11,9,12,28,38 &
$PY -u render_plans.py cube40 "$R/prep-cube40" "$R/cube40-v3-good.mp4" 1,7,5,3,10,4,6,8,2,0 &
wait
echo RENDER_TWO_END
