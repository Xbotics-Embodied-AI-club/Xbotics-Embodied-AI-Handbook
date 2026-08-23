#!/usr/bin/env bash
# 把数据集里某一条 episode 的动作，逐帧原样回放到真机上。
#
# 用途：验标定和硬件。回放不经过任何模型 —— 只是把录制时记下的关节角，
# 一帧一帧发给舵机。所以：
#   臂能把录制的轨迹走出来  → 标定 / 串口 / 舵机都正常，问题在策略或相机视角
#   臂走歪 / 抽搐 / 跑飞     → 标定不对（这条臂的零位或行程标错了）
#
# 不需要相机、不需要 policy server、不需要隧道，纯板上本地。
#
#   bash vla/2_data_collection/2_2_so101_setup/replay_episode.sh
#
# ⚠️ 一开跑，臂会从当前姿态直接跳到该 episode 第 0 帧的姿态，可能是一次较快的移动。
#    先把臂摆到大致差不多的低位、手放急停，再跑。Ctrl-C 随时停（停后臂失力，扶住）。
set -euo pipefail

# 数据集：发布者原始的单任务集之一。root 指到共享数据根下的目录，repo_id 只是个标签。
DATASET_ROOT="$DATASETS_ROOT/datasets/public/so101-pick-place-tasks/pick_up_a_plush_toy_and_place_in_the_bin"
REPO_ID=so101/pick_up_a_plush_toy_and_place_in_the_bin
EPISODE=0

cd "$(dirname "$0")/../../.."

uv run python -m lerobot.scripts.lerobot_replay \
  --robot.type=so101_follower \
  --robot.port=/dev/follower \
  --robot.id=so101_01 \
`# 必须和标定时的 ARM_ID 一致，才能加载这条臂的标定文件。` \
  --dataset.repo_id="$REPO_ID" \
  --dataset.root="$DATASET_ROOT" \
  --dataset.episode="$EPISODE" \
  --dataset.fps=30 \
  --play_sounds=false
