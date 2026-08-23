#!/usr/bin/env bash
# 在板子上跑 ACT 远程推理。用法与 π0 那个一样，差别都写在下面注释里。
#
#   bash vla/5_vla_finetune/5_4_so101_real_sft/deploy/run_act_client.sh

set -euo pipefail

SERVER=127.0.0.1:8080
# SERVER 填 GPU 机的 IP:8080 即可。若两机不在同一网段互通，可从 GPU 机侧建一条 SSH 反向
# 隧道：`ssh -R 8080:localhost:8080 robot@<板子IP>`，板上就把 SERVER 写 127.0.0.1:8080 走隧道。

# ACT 是单任务模型（只训了方块那一个任务），不是 9 任务混训的。
PRETRAINED=Harrysunshine/so101-act-cube

# 这个 --task 对 ACT 其实不起作用：lerobot 的 ACT 不接语言输入，它只看图像和关节角。
# 参数是 client 必填的，所以照实写上，但换成别的字它的行为不会变。
# 换句话说，这个 checkpoint 只会做方块那一件事。
TASK="Pick up a cube and place in the bin"

CAMERAS='{ top:   {type: opencv, index_or_path: /dev/topcam,   width: 640, height: 480, fps: 30, fourcc: "MJPG"},
           wrist: {type: opencv, index_or_path: /dev/wristcam, width: 640, height: 480, fps: 30, fourcc: "MJPG"} }'

uv run python -m lerobot.async_inference.robot_client \
  --server_address="$SERVER" \
  --robot.type=so101_follower \
  --robot.port=/dev/follower \
  --robot.id=so101_01 \
  --robot.cameras="$CAMERAS" \
  --task="$TASK" \
  --policy_type=act \
  --pretrained_name_or_path="$PRETRAINED" \
  --policy_device=cuda \
  --client_device=cpu \
  --actions_per_chunk=100 \
`# ACT 的 chunk_size 是 100（π0/SmolVLA 是 50），拉满省得频繁往返。` \
  --chunk_size_threshold=0.5 \
  --aggregate_fn_name=weighted_average \
  --fps=30
