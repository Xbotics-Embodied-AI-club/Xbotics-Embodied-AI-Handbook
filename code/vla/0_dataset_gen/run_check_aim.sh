#!/usr/bin/env bash
# 瞄点残差 + 瞄点相对两指尖中点的偏差 + 对齐残余 + 要跨的宽度（cube40）。
set -euo pipefail
cd "$(dirname "$0")"
nvidia-smi -L | head -1
R=/mnt/nas_code/Xpersonal/Xbotics2/experiment_main_v1/results/EAI-exp-002-attempt-4
"$HOME/.venv/Xbotics2-handbook/bin/python" -u check_aim.py cube40 "$R/prep-cube40" 1,2
