#!/usr/bin/env bash
# 瞄点残差 + 对齐残余 + 要跨的宽度（cube40，10 集，现失败集 0/3/5）。
set -euo pipefail
cd "$(dirname "$0")"
R=/mnt/nas_code/Xpersonal/Xbotics2/experiment_main_v1/results/EAI-exp-002-attempt-4
"$HOME/.venv/Xbotics2-handbook/bin/python" -u check_aim.py cube40 "$R/prep-cube40" 0,3,5
