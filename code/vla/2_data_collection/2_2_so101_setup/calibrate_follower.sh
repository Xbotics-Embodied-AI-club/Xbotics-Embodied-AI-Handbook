#!/usr/bin/env bash
# 标定 SO101 从臂。新到手的臂、换过舵机、或者拆装过关节之后都要重标一次。
#
# 标定测的是每个舵机的零位偏移和活动范围。这些数是**这一条物理臂专属**的，
# 不是代码：别的臂的标定文件搬过来，关节角就是错的，策略输出的动作会打偏。
#
# 标定结果落 $HF_LEROBOT_HOME/calibration/robots/so_follower/<id>.json。
# 走这个共享位置而不是 ~/.cache，是为了标一次所有机器都读得到。
#
#   bash calibrate_follower.sh
#
# 跑起来之后按屏幕提示做两步：先把臂摆到中位（各关节大致居中）按回车，
# 然后把每个关节从一端慢慢推到另一端、走满行程，再按回车结束。

set -euo pipefail

PORT=/dev/follower
ARM_ID=so101_01
# ARM_ID 就是这条物理臂的名字。标定数据是**单臂专属**的（每个舵机的零位偏移和行程），
# 换臂就得换名字重标，不能共用一份。

echo "标定 $ARM_ID（$PORT）"
echo "标定文件将落到：$HF_LEROBOT_HOME/calibration/robots/so_follower/$ARM_ID.json"
echo

uv run lerobot-calibrate \
  --robot.type=so101_follower \
  --robot.port="$PORT" \
  --robot.id="$ARM_ID"
