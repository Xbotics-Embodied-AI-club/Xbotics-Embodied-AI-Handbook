#!/usr/bin/env bash
# SO101 leader -> follower teleoperation + dataset recording.
# Task: put the cuboid into the basket.

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
