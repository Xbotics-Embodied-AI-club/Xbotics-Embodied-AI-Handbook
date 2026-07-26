#!/usr/bin/env bash
# 用 bash 解释器执行这个脚本。
set -euo pipefail
# 任何命令出错就立刻退出，未定义变量也直接报错，管道里只要有一步失败也算失败。

export MUJOCO_GL=egl
`# 让 LIBERO / MuJoCo 使用 EGL。`

# 先用官方的数据集编辑命令，把目标 episode 列表裁成一个本地子集。
lerobot-edit-dataset \
`# 官方的数据集编辑命令。` \
  --repo_id=lerobot/libero \
`# 输入数据集就是官方的 libero。` \
  --new_root=vla/3_imitation_learning/3_1_act/outputs/libero_goal_plate_subset \
`# 新数据集输出到这个本地目录。` \
  --operation.type=split \
`# 这里做的是 split 操作，也就是按 episode 切分数据集。` \
  --operation.splits='{"train": [379, 422, 426, 431, 433, 447, 448, 451, 459, 466, 481, 483, 488, 507, 511, 513, 522, 532, 537, 549, 551, 563, 568, 582, 607, 615, 620, 621, 626, 634, 639, 642, 646, 653, 655, 670, 679, 708, 716, 718, 726, 727, 749, 750, 768, 770, 801, 803, 806]}'
`# 只保留 train 里写的这些 episode 号。`

# 再用官方训练命令，直接对刚才裁出来的子数据集开训。
lerobot-train \
`# 官方训练命令。` \
  --dataset.repo_id=lerobot/libero_train \
`# 训练时把子数据集当成一个新的数据集名字来用。` \
  --dataset.root=vla/3_imitation_learning/3_1_act/outputs/libero_goal_plate_subset/train \
`# 训练数据实际位置就是 split 后生成的 train 目录。` \
  --dataset.use_imagenet_stats=false \
`# 不用 ImageNet 统计量，保持和脚本原逻辑一致。` \
  --dataset.video_backend=pyav \
`# 用视频后端读取数据。` \
  --env.type=libero \
`# 环境类型是 LIBERO。` \
  --env.task=libero_goal \
`# 只训练 libero_goal 这个任务。` \
  --env.task_ids='[8]' \
`# 只保留 task id 8。` \
  --env.obs_type=pixels_agent_pos \
`# 观测形式是图像加位姿。` \
  --env.observation_height=256 \
`# 图像高度设为 256。` \
  --env.observation_width=256 \
`# 图像宽度设为 256。` \
  --policy.type=act \
`# 使用 ACT policy。` \
  --policy.device=cuda \
`# 训练放在 CUDA 上跑。` \
  --policy.push_to_hub=false \
`# 不把模型推到 Hub。` \
  --output_dir=vla/3_imitation_learning/3_1_act/outputs/act_libero_goal_plate \
`# 输出目录写死，方便课堂演示。` \
  --job_name=libero_act \
`# 训练任务名写成 libero_act。` \
  --batch_size=256 \
`# batch size 直接写死。` \
  --num_workers=2 \
`# DataLoader worker 数量写死。` \
  --steps=100000 \
`# 训练步数写死。` \
  --eval_freq=2 \
`# 评估频率写死。` \
  --save_freq=2 \
`# 保存频率写死。` \
  --log_freq=5 \
`# 日志频率写死。` \
  --save_checkpoint=true \
`# 开启 checkpoint 保存。` \
  --wandb.enable=true \
`# 开启 wandb。` \
  --wandb.project=act-libero \
`# wandb 项目名写死。` \
  --eval.n_episodes=10 \
`# 每次评估 10 个 episode。` \
  --eval.batch_size=1 \
`# 评估 batch size 设为 1。` \
  --eval.use_async_envs=false
`# 评估环境不使用异步进程。`
