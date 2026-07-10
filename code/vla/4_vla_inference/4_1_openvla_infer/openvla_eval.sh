#!/usr/bin/env bash
# OpenVLA 官方 LIBERO-10 微调 checkpoint 标准评测。
# 与同目录 openvla_demo.py 等价的官方命令行入口：lerobot-eval 自带向量环境、多 episode
# 统计（pc_success）与自动录像，评测/复现用它最方便；demo .py 则用于课堂逐行走读。
# 在 code/ 目录运行：bash vla/4_vla_inference/4_1_openvla_infer/openvla_eval.sh
set -euo pipefail

# 想要更可信的成功率就把 N_EPISODES 调大（例如 10）；episode 依次换初始状态。
N_EPISODES=2

uv run lerobot-eval \
    --policy.type=openvla \
    --policy.pretrained_path=openvla/openvla-7b-finetuned-libero-10 \
    --policy.unnorm_key=libero_10 \
    --policy.device=cuda \
    --env.type=libero \
    --env.task=libero_10 \
    --env.task_ids="[0]" \
    --eval.n_episodes=$N_EPISODES \
    --eval.batch_size=1 \
    --output_dir=vla/4_vla_inference/4_1_openvla_infer/output/eval_openvla

# 结果：output/eval_openvla/ 下有逐 episode 视频（videos/）与 eval_info.json（含 pc_success）。
