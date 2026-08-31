#!/usr/bin/env bash
# 微调后 SmolVLA 的仿真评测。lerobot-eval 自带向量环境、多 episode 成功率统计
# （pc_success）与自动录像，是复现/汇报最方便的官方入口。
# 在 code/ 目录运行：bash vla/5_vla_finetune/5_2_smolvla_full_sft/smolvla_eval.sh
set -euo pipefail

# 评测同一个仿真任务（so101_sim 环境由我们维护的 lerobot fork（xbotics 分支）注册）。
CKPT="$DATASETS_ROOT/models/trained/so101_sim_smolvla/SO101ReachCube-v1/checkpoints/last/pretrained_model"

# episode 越多成功率越可信；每个 episode 换一次随机初始状态。
N_EPISODES=20

# 与训练一致的相机改名：so101_sim 环境输出 base_camera，checkpoint 按 camera1 训练。
# 注意：checkpoint 是在已下线的单相机 ReachCube 任务数据上训的（见 README「数据准备」），
# 而 --env.task 只能填 so101_sim 现存的场景——两者语义不再对齐，未训练过 PickPlaceCube40
# 的策略在此评测下 pc_success≈0 是预期结果，不代表策略变差。
uv run lerobot-eval \
    --policy.path="$CKPT" \
    --policy.device=cuda \
    --rename_map='{"observation.images.base_camera": "observation.images.camera1"}' \
    --env.type=so101_sim \
    --env.task=SO101PickPlaceCube40-v1 \
    --eval.n_episodes=$N_EPISODES \
    --eval.batch_size=10 \
    --output_dir=vla/5_vla_finetune/5_2_smolvla_full_sft/result/eval_smolvla

# 结果：result/eval_smolvla/ 下有逐 episode 视频（videos/）与 eval_info.json（含 pc_success）。
