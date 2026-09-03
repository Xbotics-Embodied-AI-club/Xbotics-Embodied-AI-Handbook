#!/usr/bin/env bash
# 量夹住那一段的抖动（cube40 全 10 集），一集一个进程并行。
set -euo pipefail
cd "$(dirname "$0")"
nvidia-smi -L | head -1
R=/mnt/nas_code/Xpersonal/Xbotics2/experiment_main_v1/results/EAI-exp-002-attempt-4
"$HOME/.venv/Xbotics2-handbook/bin/python" -u measure_grip_jitter.py cube40 "$R/prep-cube40"
