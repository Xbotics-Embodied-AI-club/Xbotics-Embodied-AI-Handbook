# vla0_smol —— VLA-0 的 0.5B LeRobot 原生策略（溯源说明）

本目录只做**溯源说明**。vla0_smol 的 policy 代码不在这里，而是以补丁
`code/platform/lerobot/0002-vla0-smol-policy.patch` 注入到本仓 lerobot 源树
（`code/platform/lerobot/lerobot/src/lerobot/policies/vla0_smol/`），由
`code/platform/lerobot/fetch_lerobot.sh` 自动 `git apply`。

## 模型

- **VLA-0**：把动作离散成整数、当作普通文本 token 让通用 VLM 自回归输出，VLM 本体零改装；
  推理时用 xgrammar 约束解码强制输出「恰好 N 个整数」。
- **vla0_smol**：VLA-0 的 0.5B 变体，骨干 `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`，
  被社区做成 LeRobot 原生 policy（`lerobot-train` / `lerobot-eval` 可直接跑）。

## checkpoint（直接下用）

- HuggingFace：`Robot-Learning-Collective/VLA-0-Smol`（在 LIBERO 上训练；License Apache-2.0）。
- 由 lerobot 标准 `from_pretrained` / `--policy.path` 加载，下载落 HuggingFace hub 缓存。
- 关键配置：`chunk_size=8`、动作维度 7、`n_action_bins=512`、`precision=float32`；
  官方评测用 ensemble 模式（`n_action_steps=0` + `ensemble_size=8`）。

## policy 代码上游

- 仓库：`Robot-Learning-Collective/lerobot-experiments`（fork of huggingface/lerobot）
- 取用提交：`8e71efbebbb7ea2942b579b1d21e3ec11ac2d271`（branch `dev`）
- 路径：`src/lerobot/policies/vla0_smol/{configuration,modeling,processor}_vla0_smol.py`
- 论文 / 项目页：VLA-0 — https://robot-learning-collective.github.io/vla-0-smol ；官方 3B 版 https://github.com/NVlabs/vla0

## 移植到本仓 v0.5.1 时的改动

1. **适配 lerobot v0.5.1**：import 路径与 factory 注册对齐本仓版本。
2. **去掉对 transformers 的运行时改写**：上游 `monkey_patch.py` 在运行时改
   `transformers` 的 `SmolVLMProcessor` / `SmolVLMModel.inputs_merger`（混合精度训练才需要）。
   本仓评测跑 float32，该改写为恒等操作，已删除，**不改 transformers**。
   （Phase B 若用 bf16 训练再按需以实例级覆写处理，不做全局 monkey patch。）
3. **config 兼容**：补 `amp_dtype` 字段以解码已发布 checkpoint 的 `config.json`（该字段为训练期
   AMP 记录，float32 评测不参与计算）。
