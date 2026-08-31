#!/usr/bin/env bash
# 用官方 CLI 训练 ACT 的另一条路：不写 Python，先把数据集裁成子集，再直接开训。
# 和 train_act_libero.py 等价，区别只在筛选 episode 的方式——那边用代码筛，这边把
# episode 号写进命令里。想快速改超参试一把时，这条路更顺手。
#
# 在 code/ 目录下运行：bash vla/3_imitation_learning/3_1_act/train_act_libero.sh
set -euo pipefail

# MuJoCo 走 EGL 后端，无显示器的机器也能离屏渲染出评估画面。
export MUJOCO_GL=egl

SUBSET_ROOT=vla/3_imitation_learning/3_1_act/outputs/libero_goal_plate_subset

# lerobot/libero 把十个任务套件混在一起，直接训会把十几个任务的示教混进同一个策略。
# 这里先按 episode 号切出「把碗放到盘子上」这一条任务，得到一个本地子数据集。
# 这串号码就是 train_act_libero.py 里 get_task_episodes() 筛出来的那批。
lerobot-edit-dataset \
  --repo_id=lerobot/libero \
  --new_root="${SUBSET_ROOT}" \
  --operation.type=split \
  --operation.splits='{"train": [379, 422, 426, 431, 433, 447, 448, 451, 459, 466, 481, 483, 488, 507, 511, 513, 522, 532, 537, 549, 551, 563, 568, 582, 607, 615, 620, 621, 626, 634, 639, 642, 646, 653, 655, 670, 679, 708, 716, 718, 726, 727, 749, 750, 768, 770, 801, 803, 806]}'

# 再对刚裁出来的子集开训。策略超参全用 ACTConfig 的默认值，命令行里只给数据、环境和
# 训练调度——这样命令读起来就是「训什么、在哪评、评多勤」三件事。
lerobot-train \
  --dataset.repo_id=lerobot/libero_train \
  --dataset.root="${SUBSET_ROOT}/train" \
  --dataset.use_imagenet_stats=false \
  --dataset.video_backend=pyav \
  --env.type=libero \
  --env.task=libero_goal \
  --env.task_ids='[8]' \
  --env.obs_type=pixels_agent_pos \
  --env.observation_height=256 \
  --env.observation_width=256 \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir=vla/3_imitation_learning/3_1_act/outputs/act_libero_goal_plate \
  --job_name=libero_act \
  --batch_size=256 \
  --num_workers=2 \
  --steps=100000 \
  --eval_freq=2 \
  --save_freq=2 \
  --log_freq=5 \
  --save_checkpoint=true \
  --wandb.enable=true \
  --wandb.project=act-libero \
  --eval.n_episodes=10 \
  --eval.batch_size=1 \
  --eval.use_async_envs=false
