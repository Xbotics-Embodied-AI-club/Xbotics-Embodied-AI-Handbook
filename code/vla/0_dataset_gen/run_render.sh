#!/usr/bin/env bash
# 渲全部 10 集（cube40），上视图 + 腕部并排。不挑集 —— 失败的也放进去。
set -euo pipefail
cd "$(dirname "$0")"
nvidia-smi -L | head -1        # 机器身份只认显卡（bd xb-kj8h）
R=/mnt/nas_code/Xpersonal/Xbotics2/experiment_main_v1/results/EAI-exp-002-attempt-4
"$HOME/.venv/Xbotics2-handbook/bin/python" -u render_plans.py cube40 "$R/prep-cube40" \
  "$R/cube40-drop10-10.mp4"
