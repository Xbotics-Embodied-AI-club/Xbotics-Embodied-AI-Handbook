#!/usr/bin/env bash
# 拉取 D-Robotics rdk_LeRobot_tools 的 s600 分支到 ./rdk_LeRobot_tools（与本脚本同目录的子目录）。
#
# 这是 RDK S600/S100 上 ACT BPU 部署的官方脚本仓库
# （export_bpu_actpolicy.py / bpu_control_robot.py / bpu_export_config_s600_calfix.yaml）。
# 纯脚本仓库（无 setup.py），不可作 package 依赖，故只 fetch 不 install；
# 配合 experiments pyproject 的 gpu_x86 extra（lerobot + onnx 等）使用。
#
# 用法：cd experiments && bash lerobot/rdk/fetch_rdk_tools.sh
# 设计：
#   - clone 落点 = code/lerobot/rdk/rdk_LeRobot_tools/（外层仓库 gitignore，不分发源树）。
#   - 保留自带 .git（可见分支 / 可重拉）。
#   - 幂等：源已存在则跳过 clone。可反复运行。
set -euo pipefail

RDK_TOOLS_REF="${RDK_TOOLS_REF:-s600}"
RDK_TOOLS_URL="${RDK_TOOLS_URL:-https://github.com/D-Robotics/rdk_LeRobot_tools.git}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${1:-$SCRIPT_DIR/rdk_LeRobot_tools}"

if [ -f "$TARGET/export_bpu_actpolicy.py" ]; then
  echo "[fetch_rdk_tools] $TARGET 已存在，跳过 clone。"
else
  echo "[fetch_rdk_tools] clone rdk_LeRobot_tools $RDK_TOOLS_REF -> $TARGET"
  git clone --depth 1 --branch "$RDK_TOOLS_REF" "$RDK_TOOLS_URL" "$TARGET"
fi

echo "[fetch_rdk_tools] 完成：$TARGET （rdk_LeRobot_tools $RDK_TOOLS_REF）"
