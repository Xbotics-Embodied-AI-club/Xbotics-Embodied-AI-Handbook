#!/usr/bin/env bash
# 把 so101-sim 的新提交装进 worker 的 venv，并当场核对装进去的是不是那一版。
#
# ★ `so101_sim @ git+…` 不带钉版，但 `uv.lock` 会把它锁在某个 SHA 上 —— 只跑 sync
#   取不到新提交，而且**不报错**。所以必须先 `uv lock --upgrade-package so101_sim`。
#   曾因此白扫一轮夹爪碰撞体内缩量：三档参数跑出逐位相同的结果（bd xb-…）。
#
# ★ `--extra gpu-x86` 不能省。本项目的三个 extras（gpu-x86 / nogpu-x86 / rdk-s100）
#   声明为互斥，**默认不装任何一个**：不带它跑 sync 只留 20 个包，会把 so101_sim 与
#   lerobot 整套剪掉（实测踩过，venv 当场不可用）。`--all-extras` 也不行 —— 互斥。
#
# ★ 装完必须核对：打印 venv 里那个常量的值。不核对就等于假定 sync 生效了。
#
# ★ 工具路径用绝对路径。`~/.local/bin/uv` 是**另一台机器**上的位置，本机在
#   `/usr/local/bin/uv`；`sync-env` 只是它的薄包装（读 PROJECT_NAME/SUBPROJECT_NAME
#   拼出 UV_PROJECT_ENVIRONMENT 后 `exec uv sync "$@"`）。
set -euo pipefail
nvidia-smi -L | head -1        # 机器身份只认显卡（bd xb-kj8h）
UV=/usr/local/bin/uv
CODE=/mnt/nas_code/Xpersonal/Xbotics2/.worktrees/handbook-EAI-exp-002b/code
export PROJECT_NAME=Xbotics2 SUBPROJECT_NAME=handbook
export UV_PROJECT_ENVIRONMENT="$HOME/.venv/Xbotics2-handbook"
cd "$CODE"
$UV lock --upgrade-package so101_sim
/usr/local/bin/sync-env --extra gpu-x86
"$UV_PROJECT_ENVIRONMENT/bin/python" - <<'PY'
from so101_sim.robots.so101_base import so101
print("  venv 里的夹爪关节摩擦:", so101.GRIPPER_JOINT_FRICTION)
print("  venv 里的夹爪力矩上限:", so101.GRIPPER_FORCE_LIMIT)
PY
echo SYNC_END
