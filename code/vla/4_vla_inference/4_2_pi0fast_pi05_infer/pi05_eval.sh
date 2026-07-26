#!/usr/bin/env bash
# π0.5 LIBERO 微调 checkpoint 标准评测。
# 与同目录 pi05_demo.py 等价的官方命令行入口：lerobot-eval 自带向量环境、多 episode
# 统计（pc_success）与自动录像，评测/复现用它最方便；demo .py 则用于课堂逐行走读。
# 在 code/ 目录运行：bash vla/4_vla_inference/4_2_pi0fast_pi05_infer/pi05_eval.sh
set -euo pipefail

# 想要更可信的成功率就把 N_EPISODES 调大（例如 10）；episode 依次换初始状态。
N_EPISODES=2

uv run lerobot-eval \
    --policy.path=lerobot/pi05_libero_finetuned_v044 \
    --policy.device=cuda \
    --env.type=libero \
    --env.task=libero_goal \
    --env.task_ids="[5]" \
    --eval.n_episodes=$N_EPISODES \
    --eval.batch_size=1 \
    --output_dir=vla/4_vla_inference/4_2_pi0fast_pi05_infer/output/eval_pi05

# 结果：output/eval_pi05/ 下有逐 episode 视频（videos/）与 eval_info.json（含 pc_success）。
