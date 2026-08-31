#!/usr/bin/env bash
# SmolVLA LIBERO 微调 checkpoint 标准评测。
# 与同目录 smolvla_demo.py 等价的官方命令行入口：lerobot-eval 自带向量环境、多 episode
# 统计（pc_success）与自动录像，评测/复现用它最方便；demo .py 则用于逐行走读。
# 在 code/ 目录运行：bash vla/4_vla_inference/4_4_smolvla_infer/smolvla_eval.sh
set -euo pipefail

# 想要更可信的成功率就把 N_EPISODES 调大（例如 10）；episode 依次换初始状态。
N_EPISODES=2

# rename_map：此 checkpoint 按 camera1/2/3 命名训练；LIBERO 环境只给 image/image2
# 两路，改名映射过去即可（缺 camera3 允许——校验只要求一方是另一方的子集）。
uv run lerobot-eval \
    --policy.path=lerobot/smolvla_libero \
    --policy.device=cuda \
    --rename_map='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}' \
    --env.type=libero \
    --env.task=libero_goal \
    --env.task_ids="[5]" \
    --eval.n_episodes=$N_EPISODES \
    --eval.batch_size=1 \
    --output_dir=vla/4_vla_inference/4_4_smolvla_infer/output/eval_smolvla

# 结果：output/eval_smolvla/ 下有逐 episode 视频（videos/）与 eval_info.json（含 pc_success）。
