#!/usr/bin/env bash
# setup_env.sh —— 一键创建并注册第 4 讲 Demo 的隔离运行环境。
#
# 默认使用 venv（轻量、可靠；本 Demo 依赖均为纯 pip 包，无需 conda）。
# 若希望使用 conda，加 --conda 参数（需 conda 通道网络可达）。
#
# 用法：
#   bash setup_env.sh             # venv（默认，推荐）
#   bash setup_env.sh --conda     # conda（用 environment.yml）
#
# 之后启动 Notebook：
#   source .venv/bin/activate && jupyter notebook sensors_coordinates_demo.ipynb   # venv
#   conda activate lecture4-sensors && jupyter notebook sensors_coordinates_demo.ipynb  # conda

set -euo pipefail

KERNEL_NAME="lecture4-sensors"
DISPLAY_NAME="Python (lecture4-sensors)"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USE_CONDA=false

for arg in "$@"; do
  case "$arg" in
    --conda) USE_CONDA=true ;;
    --venv) USE_CONDA=false ;;
    *) echo "未知参数：$arg（支持 --conda / --venv）" >&2; exit 1 ;;
  esac
done

if [ "$USE_CONDA" = true ]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "错误：未检测到 conda，无法使用 --conda。请改用 venv：bash setup_env.sh" >&2
    exit 1
  fi
  echo "==> 使用 conda 创建环境 '${KERNEL_NAME}' ..."
  if conda env list | grep -qE "^${KERNEL_NAME}[[:space:]]"; then
    conda env update -f "${HERE}/environment.yml" --prune
  else
    conda env create -f "${HERE}/environment.yml"
  fi

  eval "$(conda shell.bash hook)"
  conda activate "${KERNEL_NAME}"
  python -m ipykernel install --user --name "${KERNEL_NAME}" --display-name "${DISPLAY_NAME}"

  echo ""
  echo "==> 完成。启动 Notebook："
  echo "   conda activate ${KERNEL_NAME}"
  echo "   jupyter notebook sensors_coordinates_demo.ipynb"
else
  echo "==> 使用 venv 创建隔离环境 ..."
  VENV_DIR="${HERE}/.venv"
  if [ ! -d "${VENV_DIR}" ]; then
    python3 -m venv "${VENV_DIR}"
  fi
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
  python -m pip install --upgrade pip
  pip install -r "${HERE}/requirements.txt"
  python -m ipykernel install --user --name "${KERNEL_NAME}" --display-name "${DISPLAY_NAME}"

  echo ""
  echo "==> 完成。启动 Notebook："
  echo "   source .venv/bin/activate"
  echo "   jupyter notebook sensors_coordinates_demo.ipynb"
fi
