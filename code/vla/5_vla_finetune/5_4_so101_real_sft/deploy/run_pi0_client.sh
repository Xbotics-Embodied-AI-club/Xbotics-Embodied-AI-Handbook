#!/usr/bin/env bash
# 在 Orin Nano 上跑 π0：板子只负责读相机、读关节、下发动作，推理在 GPU 机上做。
set -euo pipefail

# GPU 机的地址。策略在那边加载和推理。
# ⚠️ 下面这个 IP 要填成真正跑 policy server 的 GPU 机地址，再跑本脚本。
SERVER=127.0.0.1:8080
# SERVER 填 GPU 机的 IP:8080 即可。若两机不在同一网段互通，可从 GPU 机侧建一条 SSH 反向
# 隧道：`ssh -R 8080:localhost:8080 robot@<板子IP>`，板上就把 SERVER 写 127.0.0.1:8080 走隧道。

# checkpoint 路径——这个字符串是发给 GPU 机的，由 **GPU 机** 去读，不是板子读。
# 它在共享数据根上，同网段的机器都读得到。
CKPT="$DATASETS_ROOT/models/trained/so101/pi0/pretrained_model"

# 任务指令。π0 是语言条件策略，这句话必须和训练数据里的指令**逐字一致**。
# 这句抄自数据集 meta/tasks.parquet：是 "a plush toy"，且 "place in the bin" 没有 it。
TASK="Pick up a plush toy and place in the bin"

cd "$(dirname "$0")/.."

uv run python \
  -m lerobot.async_inference.robot_client \
  --server_address="$SERVER" \
  --robot.type=so101_follower \
  --robot.port=/dev/follower \
  --robot.id=so101_01 \
`# 必须和标定时用的 ARM_ID 一致，否则读不到这条臂的标定文件。` \
  --robot.cameras='{ top: {type: opencv, index_or_path: /dev/topcam, width: 640, height: 480, fps: 30, fourcc: "MJPG"}, wrist: {type: opencv, index_or_path: /dev/wristcam, width: 640, height: 480, fps: 30, fourcc: "MJPG"} }' \
`# 相机名 top / wrist 对应策略的 observation.images.top / .wrist。` \
  --task="$TASK" \
  --policy_type=pi0 \
  --pretrained_name_or_path="$CKPT" \
  --policy_device=cuda \
  --client_device=cpu \
  --actions_per_chunk=50 \
`# 和 π0 的 n_action_steps=50 对齐；要多了模型也吐不出来。` \
  --chunk_size_threshold=0.5 \
  --aggregate_fn_name=weighted_average \
  --debug_visualize_queue_size=false
