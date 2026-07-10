#!/usr/bin/env bash
# 用 bash 运行这个脚本。
set -euo pipefail
# 任一命令失败就退出。

uv run python \
`# 统一 uv 环境启动（在 code/ 下运行）。` \
  -m lerobot.async_inference.robot_client \
`# 启动 LeRobot 的 async inference robot client。` \
  --server_address=127.0.0.1:8080 \
`# 这里写 GPU 机的真实 IP:端口。` \
  --robot.type=so101_follower \
`# 指定 SO101 follower 机器人配置。` \
  --robot.port=/dev/ttyFollower \
`# 机器人串口写死为 follower 的 udev 名称。` \
  --robot.id=my_awesome_follower_arm \
`# 机器人 ID 写死，方便日志里识别。` \
  --robot.cameras='{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: "MJPG"}, side: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: "MJPG"} }' \
`# 两路相机配置写死为 front 和 side。` \
  --task="Put the cuboid into the basket" \
`# 任务文本写死成“把 cuboid 放进篮子里”。` \
  --policy_type=act \
`# 远程推理用 ACT。` \
  --pretrained_name_or_path=outputs/act_cuboid_local/checkpoints/last/pretrained_model \
`# 这里填训练出来的 checkpoint 路径，实际是发给远程 server 去加载。` \
  --policy_device=cuda \
`# policy 在 GPU 机上加载和推理。` \
  --client_device=cpu \
`# client 端不做推理，只把远程回来的动作留在 CPU 上给机器人执行。` \
  --actions_per_chunk=100 \
`# 每次从 server 拉一段动作。` \
  --chunk_size_threshold=0.5 \
`# 队列低于一半时就提前补下一段动作。` \
  --aggregate_fn_name=weighted_average \
`# 用加权平均做动作聚合。` \
  --debug_visualize_queue_size=false
`# 是否可视化队列长度；调 buffer 时可以改成 true。`
