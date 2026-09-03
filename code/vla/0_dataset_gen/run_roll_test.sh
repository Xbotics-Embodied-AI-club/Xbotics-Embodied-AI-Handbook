#!/usr/bin/env bash
# 可证伪测试：腕滚在 90° 等价解里择优后，打滑那三集是否进稳定簇。
set -euo pipefail
D="$(cd "$(dirname "$0")" && pwd)"
bash "$D/run_prep_check.sh"
bash "$D/run_jitter.sh"
echo ROLL_TEST_END
