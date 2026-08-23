#!/usr/bin/env bash
# 在板子上跑 SmolVLA 远程推理。用法与 π0 那个一样。
#
#   bash vla/5_vla_finetune/5_4_so101_real_sft/deploy/run_smolvla_client.sh

set -euo pipefail

SERVER=127.0.0.1:8080
# SERVER 填 GPU 机的 IP:8080 即可。若两机不在同一网段互通，可从 GPU 机侧建一条 SSH 反向
# 隧道：`ssh -R 8080:localhost:8080 robot@<板子IP>`，板上就把 SERVER 写 127.0.0.1:8080 走隧道。

# 与 π0 同一批 9 任务数据训出来的，用来做对照。采用点是末点 030000。
PRETRAINED=Harrysunshine/so101-smolvla-9task

# SmolVLA 看语言，任务文本同样要用训练数据里的说法。
TASK="Pick up a battery and place in the bin"

CAMERAS='{ top:   {type: opencv, index_or_path: /dev/topcam,   width: 640, height: 480, fps: 30, fourcc: "MJPG"},
           wrist: {type: opencv, index_or_path: /dev/wristcam, width: 640, height: 480, fps: 30, fourcc: "MJPG"} }'

uv run python -m lerobot.async_inference.robot_client \
  --server_address="$SERVER" \
  --robot.type=so101_follower \
  --robot.port=/dev/follower \
  --robot.id=so101_01 \
  --robot.cameras="$CAMERAS" \
  --task="$TASK" \
  --policy_type=smolvla \
  --pretrained_name_or_path="$PRETRAINED" \
  --policy_device=cuda \
  --client_device=cpu \
  --actions_per_chunk=50 \
  --chunk_size_threshold=0.5 \
  --aggregate_fn_name=weighted_average \
  --fps=30
