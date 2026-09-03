#!/usr/bin/env bash
# 重备 10 集 cube40 规划并判成败。**一集一个进程并行判**（bd xb-nqqr）。
set -euo pipefail
cd "$(dirname "$0")"
nvidia-smi -L | head -1        # 机器身份只认显卡（bd xb-kj8h）
R=/mnt/nas_code/Xpersonal/Xbotics2/experiment_main_v1/results/EAI-exp-002-attempt-4
P="$R/prep-cube40"
PY="$HOME/.venv/Xbotics2-handbook/bin/python"
rm -rf "$P"
$PY -u prepare_episodes.py cube40 "$P" 10 0
N=$(find "$P/plans" -name 'ep*.npy' | wc -l)
# 并行度取集数：每个进程一条 CPU PhysX 单线程回放，本机核数远够。
seq 0 $((N - 1)) | xargs -P "$N" -I{} $PY -u check_plans.py cube40 "$P" {} > "$R/review/per-ep.txt" 2>&1
sort "$R/review/per-ep.txt" | awk '/ep[0-9]+:/'
ok=$(awk '/: 成功/' "$R/review/per-ep.txt" | wc -l)
echo
echo "  cube40：$ok/$N 成功（$((ok * 100 / N))%）"
echo CHECK_PLANS_END
