#!/usr/bin/env bash
# SO101 leader -> follower teleoperation with two cameras + rerun.
# Leader on /dev/ttyLeader, follower on /dev/ttyFollower (already udev-renamed).
# 相机键名与设备名和 record.sh 保持一致（top / wrist），否则这里验通的相机
# 到了录制那一步会落成另一套字段名。设备名同样来自 udev 绑定，换机架就改这一行。

set -euo pipefail

export PATH="$UV_PROJECT_ENVIRONMENT/bin:$PATH"

lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyFollower \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras='{ top: {type: opencv, index_or_path: "/dev/top_camera", width: 640, height: 480, fps: 30, fourcc: "MJPG"}, wrist: {type: opencv, index_or_path: "/dev/wrist_camera", width: 640, height: 480, fps: 30, fourcc: "MJPG"} }' \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyLeader \
    --teleop.id=my_awesome_leader_arm \
    --display_data=true \
    --fps=30
