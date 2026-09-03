#!/usr/bin/env bash
# 按提交里的 uv.lock 恢复 venv（撤销 --upgrade-package 造成的重解），并核对 so101_sim 在位。
set -euo pipefail
nvidia-smi -L | head -1
export PROJECT_NAME=Xbotics2 SUBPROJECT_NAME=handbook
cd /mnt/nas_code/Xpersonal/Xbotics2/.worktrees/handbook-EAI-exp-002b/code
/usr/local/bin/sync-env --extra gpu-x86
"$HOME/.venv/Xbotics2-handbook/bin/python" -c "
import so101_sim
from so101_sim.robots.so101_base import so101
print('  so101_sim 在位:', so101_sim.__file__)
print('  夹爪关节摩擦:', getattr(so101, 'GRIPPER_JOINT_FRICTION', '★ 没有这个常量 = 装的是旧版'))
"
echo RESTORE_END
