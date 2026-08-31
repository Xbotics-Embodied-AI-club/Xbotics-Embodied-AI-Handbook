#!/usr/bin/env bash
# VLA-0（讲15 GRPO 后训练产物）标准评测。checkpoint 在本地训练产物目录。
# 与同目录 vla0_demo.py 等价的官方命令行入口：lerobot-eval 自带向量环境、多 episode
# 统计（pc_success）与自动录像，评测/复现用它最方便；demo .py 则用于逐行走读。
# 在 code/ 目录运行：bash vla/4_vla_inference/4_3_vla0_infer/vla0_eval.sh
set -euo pipefail

# 想要更可信的成功率就把 N_EPISODES 调大（例如 10）；episode 依次换初始状态。
N_EPISODES=2

uv run lerobot-eval \
    --policy.path="$DATASETS_ROOT/models/trained/xbotics_rl_grpo_vla0/grpo_runs/iter002" \
    --policy.device=cuda \
    --env.type=libero \
    --env.task=libero_object \
    --env.task_ids="[0]" \
    --eval.n_episodes=$N_EPISODES \
    --eval.batch_size=1 \
    --output_dir=vla/4_vla_inference/4_3_vla0_infer/output/eval_vla0

# 结果：output/eval_vla0/ 下有逐 episode 视频（videos/）与 eval_info.json（含 pc_success）。
