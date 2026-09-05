#!/usr/bin/env bash
# 把训好的策略推到 HF —— **整仓删了重传**，连同验收数字与跑法。
#
# 与数据集那份同一条规矩（用户 2026-09-04 定）：不做增量。增量的失败方式很讨厌 ——
# 旧版权重残留在同一个目录里，config 说的和文件对不上，而 HF 与下载方都不报错。
#
# ★ 传之前先验 checkpoint 能不能被 lerobot 装起来。只看文件在不在挡不住
#   「存了但存坏了」—— 那种情况下载方要到推理时才发现。
#
# ★ 全参与 LoRA 两种产物都认：全参落的是 `model.safetensors`（整套权重），
#   LoRA 落的是 `adapter_model.safetensors` + `adapter_config.json`（只有旁路）。
#   装载路径也不同 —— 适配器要先读 `PeftConfig` 找到基座，再把旁路挂上去。
#   第一版只认 `model.safetensors`，拿 LoRA 产物跑会判成「不是完整目录」直接退出。
#
# 用法：bash upload_model_hf.sh <pretrained_model 目录> <HF 仓名> [<说明文件>]
set -euo pipefail

CKPT="${1:?用法: bash upload_model_hf.sh <pretrained_model 目录> <HF 仓名> [<说明文件>]}"
REPO="${2:?缺 HF 仓名}"
CARD="${3:-}"
# 指定 VENV 就用它 bin 下的命令，不指定就用 PATH 上的。
BIN="${VENV:+$VENV/bin/}"
HF="${BIN}hf"

# 代理由机器的 /etc/environment 预置，不在这里写死。非登录 shell 读不到它，
# 后台起时若连不上 HF，在调用侧导出。

echo "== ① 先确认这份权重装得起来 =="
"${BIN}python" - "$CKPT" <<'PY'
import sys
from pathlib import Path

ckpt = Path(sys.argv[1])
if not (ckpt / "config.json").is_file():
    sys.exit("★ 缺 config.json —— 这不是一个 pretrained_model 目录")

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

if (ckpt / "adapter_model.safetensors").is_file():
    from peft import PeftConfig, PeftModel
    pc = PeftConfig.from_pretrained(ckpt)
    base = SmolVLAPolicy.from_pretrained(pc.base_model_name_or_path)
    p = PeftModel.from_pretrained(base, ckpt)
    tr = sum(x.numel() for x in p.parameters() if x.requires_grad)
    tot = sum(x.numel() for x in p.parameters())
    print(f"  适配器装载成功：基座 {pc.base_model_name_or_path}，"
          f"旁路 {tr/1e6:.1f}M / 总 {tot/1e6:.1f}M")
elif (ckpt / "model.safetensors").is_file():
    p = SmolVLAPolicy.from_pretrained(ckpt)
    print(f"  整套权重装载成功：{p.name}，参数 "
          f"{sum(x.numel() for x in p.parameters())/1e6:.1f}M")
else:
    sys.exit("★ 既没有 model.safetensors 也没有 adapter_model.safetensors")
PY

echo "== ② 整仓删了重传 =="
# 删除会交互确认且没有 --yes；用 here-string 喂，不能用 `yes |`
# （`yes` 吃 SIGPIPE 退 141，在 pipefail 下会把脚本打断在「删了还没传」的半路）。
$HF repos delete "$REPO" --repo-type model --missing-ok <<< "y"
$HF repos create "$REPO" --repo-type model --exist-ok --public

echo "== ③ 传权重 =="
$HF upload "$REPO" "$CKPT" . --repo-type model \
  --commit-message "${COMMIT_MSG:-SO-101 仿真+真机混训权重}"

if [ -n "$CARD" ] && [ -f "$CARD" ]; then
  echo "== ④ 传说明 =="
  $HF upload "$REPO" "$CARD" README.md --repo-type model \
    --commit-message "模型卡：训练配置与验收数字"
fi

echo "UPLOAD_MODEL_HF_DONE  https://huggingface.co/$REPO"
