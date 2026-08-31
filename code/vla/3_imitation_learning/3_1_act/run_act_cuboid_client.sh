#!/usr/bin/env bash
# 异步推理的真机版 client：SO-101 在本地跑，策略在远端 GPU 机器上跑。
# 和 infer_act_libero_client.py 是同一套机制，只是把仿真环境换成了真的机械臂。
#
# 先在 GPU 机器上起 server，再在接着机械臂的这台机器上运行：
#   bash vla/3_imitation_learning/3_1_act/run_act_cuboid_client.sh
set -euo pipefail

uv run python -m lerobot.async_inference.robot_client \
  `# server_address 填 GPU 机器的 IP:端口；本机测试用 127.0.0.1。` \
  --server_address=127.0.0.1:8080 \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyFollower \
  --robot.id=my_awesome_follower_arm \
  `# 相机名要和训练数据集里的 image key 一致，否则策略认不出这两路画面。` \
  --robot.cameras='{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: "MJPG"}, side: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: "MJPG"} }' \
  `# 任务描述要和采数据时写的那句一致。` \
  --task="Put the cuboid into the basket" \
  --policy_type=act \
  `# 这个路径是在远端 server 那台机器上解析的，不是本机路径。` \
  --pretrained_name_or_path=outputs/act_cuboid_local/checkpoints/last/pretrained_model \
  --policy_device=cuda \
  `# client 端不做推理，动作留在 CPU 上直接发给舵机。` \
  --client_device=cpu \
  --actions_per_chunk=100 \
  `# 队列剩不到一半就提前请求下一段：补货要早于断货，否则机械臂会停顿。` \
  --chunk_size_threshold=0.5 \
  `# 多段 chunk 覆盖同一时刻时，用加权平均把它们合成一个动作。` \
  --aggregate_fn_name=weighted_average \
  --debug_visualize_queue_size=false
