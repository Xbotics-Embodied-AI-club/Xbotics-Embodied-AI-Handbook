#!/usr/bin/env bash
# 重备若干集 cube40 规划并判成败。**一集一个进程并行判**（bd xb-nqqr）。
#
# 集数默认 40：n=10 分辨不出 8/10 与 9/10 的差别，而打滑与空抓两类失效在换场景后照常
# 出现 ⇒ 它们是分布上的、不是个别场景的，必须量占比。并行化之后这个代价付得起。
#
# 用法：bash run_prep_check.sh [<集数>]
set -euo pipefail
cd "$(dirname "$0")"
nvidia-smi -L | head -1        # 机器身份只认显卡（bd xb-kj8h）
R=/mnt/nas_code/Xpersonal/Xbotics2/experiment_main_v1/results/EAI-exp-002-attempt-4
P="$R/prep-cube40"
PY="$HOME/.venv/Xbotics2-handbook/bin/python"
rm -rf "$P"
COUNT="${1:-40}"
$PY -u prepare_episodes.py cube40 "$P" "$COUNT" 0
N=$(find "$P/plans" -name 'ep*.npy' | wc -l)
# 并行度封顶 12：每个进程一条 CPU PhysX 单线程回放，再多就抢核了。
seq 0 $((N - 1)) | xargs -P 12 -I{} $PY -u check_plans.py cube40 "$P" {} > "$R/review/per-ep.txt" 2>&1
sort "$R/review/per-ep.txt" | awk '/ep[0-9]+:/'
ok=$(awk '/: 成功/' "$R/review/per-ep.txt" | wc -l)
echo
echo "  cube40：$ok/$N 成功（$((ok * 100 / N))%）"
echo CHECK_PLANS_END
