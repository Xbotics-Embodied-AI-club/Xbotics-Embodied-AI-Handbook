#!/usr/bin/env bash
# SO101 leader -> follower teleoperation with two cameras + rerun.
# Leader on /dev/ttyLeader, follower on /dev/ttyFollower (already udev-renamed).

set -euo pipefail

export PATH="$UV_PROJECT_ENVIRONMENT/bin:$PATH"

lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyFollower \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras='{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: "MJPG"}, side: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: "MJPG"} }' \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyLeader \
    --teleop.id=my_awesome_leader_arm \
    --display_data=true \
    --fps=30
