#!/usr/bin/env bash
# 重备 10 集 cube40 规划（两指按**方块的面**对齐）并立即判成败。
#
# 早先 `jaw_yaw_error` 对齐的是世界 x/y 轴，而方块复位时带随机自旋 —— 实测「离面对齐」
# >28° 的四集全败（指尖撞角把方块推走、从没抬起来过），<10.3° 的全成。
set -euo pipefail
cd "$(dirname "$0")"
R=/mnt/nas_code/Xpersonal/Xbotics2/experiment_main_v1/results/EAI-exp-002-attempt-4
P="$R/prep-cube40"
rm -rf "$P"
PY="$HOME/.venv/Xbotics2-handbook/bin/python"
$PY -u prepare_episodes.py cube40 "$P" 10 0
$PY -u check_plans.py cube40 "$P"
