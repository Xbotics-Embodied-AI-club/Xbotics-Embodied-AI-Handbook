#!/usr/bin/env bash
# 一条链：重装 venv（取到 so101-sim 新提交）→ 重备规划 → 并行判成败 → 渲视频。
set -euo pipefail
D="$(cd "$(dirname "$0")" && pwd)"
bash "$D/run_sync.sh"
bash "$D/run_prep_check.sh"
bash "$D/run_render.sh"
echo ALL_END
