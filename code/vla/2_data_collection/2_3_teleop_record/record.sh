#!/usr/bin/env bash
# SO101 leader -> follower teleoperation + dataset recording.
# Task: put the cuboid into the basket.
#
# 设备名来自 udev 绑定，不同机架绑出来的名字不一样，跑之前先确认自己这台绑的是哪一套：
#   platform/so101_real/setup/bind_camera_s100.sh      -> /dev/top_camera, /dev/wrist_camera
#   platform/so101_real/setup/bind_devices.sh          -> /dev/topcam, /dev/wristcam
#   platform/so101_real/setup/bind_uarm_serial_port.sh -> 串口名由第二个参数指定
# 下面这条按 bind_camera_s100.sh 那套写；换机架就改这里的四个设备名。

set -euo pipefail

export PATH="$UV_PROJECT_ENVIRONMENT/bin:$PATH"
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyFollower \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras='{ top: {type: opencv, index_or_path: "/dev/top_camera", width: 640, height: 480, fps: 30, fourcc: "MJPG"}, wrist: {type: opencv, index_or_path: "/dev/wrist_camera", width: 640, height: 480, fps: 30, fourcc: "MJPG"} }' \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyLeader \
    --teleop.id=my_awesome_leader_arm \
    --display_data=true \
    --dataset.repo_id=local/cuboid \
    --dataset.root="$HF_LEROBOT_HOME/so101/cuboid" \
    --dataset.num_episodes=50 \
    --dataset.single_task="Put the cuboid into the basket" \
    --dataset.push_to_hub=false \
    --dataset.episode_time_s=30 \
    --dataset.reset_time_s=30 
#    --resume=true
