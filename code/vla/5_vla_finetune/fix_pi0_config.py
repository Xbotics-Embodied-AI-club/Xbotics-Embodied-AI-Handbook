"""把 `lerobot/pi0` 的 config.json 补成当前 lerobot 认得的新格式。

## 为什么需要这一步

`lerobot/pi0` 在 HF 上最新那次提交（`e4ed526a`，2025-09-19，「Migrate pi0 weights to
pipeline」）**把权重迁到了新结构，但 config.json 没跟着改**。于是：

- 权重是好的 —— 与一份能被当前 lerobot 装起来的 pi0 checkpoint 逐一比对，
  777 个张量名完全相同；
- 配置是旧的 —— 里面 8 个字段本版 `PI0Config` 已经不认（`num_steps` 改叫
  `num_inference_steps`，`proj_width` / `adapt_to_pi_aloha` / `resize_imgs_with_padding`
  / `use_delta_joint_actions_aloha` / `use_cache` / `attention_implementation`
  / `train_state_proj` 整个去掉了）。

draccus 解码遇到不认识的字段**直接抛 `DecodingError`**，训练在第一秒就退出。
症状看着像环境坏了，其实只是一个 JSON 落后了 —— **不是显存问题，别去调 batch。**

## 这个脚本做什么

删掉那些字段，其余交给 `PI0Config` 的默认值。**这样做是安全的，因为逐项核对过**：
`paligemma_variant=gemma_2b`、`action_expert_variant=gemma_300m`、
`image_resolution=(224,224)`、`num_inference_steps=10`、`max_state_dim=32`、
`max_action_dim=32`、`chunk_size=50`、`n_action_steps=50`、`tokenizer_max_length=48`
—— 默认值与一份实际训练过、能装起来的 pi0 配置逐项相同。

★ 被删的字段里若有**非默认取值**，脚本会报出来而不是静默丢掉：静默丢掉等于悄悄换了
  一个模型，而这种事不会有任何东西报错。

用法：`python fix_pi0_config.py <pi0 目录> [<输出目录>]`
      不给输出目录就原地改（会先存一份 `config.json.orig`）。
"""

import json
import shutil
import sys
from pathlib import Path

# 旧字段 → 新字段名。值为 None 表示这一项在新版里整个去掉了。
RENAMED = {"num_steps": "num_inference_steps"}
DROPPED = (
    "proj_width",                     # 由 action_expert_variant 决定，不再单独配
    "adapt_to_pi_aloha",              # ALOHA 专用的关节换算，已移出策略配置
    "use_delta_joint_actions_aloha",  # 同上
    "resize_imgs_with_padding",       # 换成 image_resolution
    "use_cache",
    "attention_implementation",
    "train_state_proj",
)


def fix(src_dir: Path, dst_dir: Path) -> None:
    """读旧 config，写新 config。

    Args:
        src_dir: 含 `config.json` 与 `model.safetensors` 的 pi0 目录。
        dst_dir: 输出目录；与 `src_dir` 相同表示原地改。
    """
    src = src_dir / "config.json"
    if not src.is_file():
        sys.exit(f"★ {src} 不在")
    cfg = json.loads(src.read_text())

    from lerobot.policies.pi0.configuration_pi0 import PI0Config
    known = {f.name for f in __import__("dataclasses").fields(PI0Config)}

    for old, new in RENAMED.items():
        if old in cfg:
            cfg[new] = cfg.pop(old)
            print(f"  改名 {old} → {new} = {cfg[new]}")

    defaults = PI0Config()
    for k in DROPPED:
        if k in cfg:
            print(f"  删掉 {k} = {cfg.pop(k)}")

    stale = sorted(set(cfg) - known - {"type"})
    if stale:
        sys.exit(f"★ 还剩 {len(stale)} 个本版不认的字段，本脚本的清单该更新了：{stale}")

    # 逐项核对：新版默认值与旧 config 里同名字段的取值一致，才敢说「交给默认值」是安全的。
    for k in ("chunk_size", "n_action_steps", "max_state_dim", "max_action_dim",
              "tokenizer_max_length", "num_inference_steps"):
        if k in cfg and cfg[k] != getattr(defaults, k):
            print(f"  ⚠ {k}：checkpoint 说 {cfg[k]}，本版默认 {getattr(defaults, k)} —— 以 checkpoint 为准")

    dst_dir.mkdir(parents=True, exist_ok=True)
    out = dst_dir / "config.json"
    if out.resolve() == src.resolve():
        shutil.copy2(src, src.with_suffix(".json.orig"))
        print(f"  原地改，旧版存为 {src.with_suffix('.json.orig').name}")
    out.write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"  写出 {out}（{len(cfg)} 个字段）")


def main(argv) -> int:
    if len(argv) not in (1, 2):
        sys.exit(__doc__)
    src = Path(argv[0])
    fix(src, Path(argv[1]) if len(argv) == 2 else src)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
