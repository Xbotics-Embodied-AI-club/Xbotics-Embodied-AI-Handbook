#!/usr/bin/env bash
# 打印当前插着的相机和串口分别落在哪个 USB 口位置（ID_PATH）。
#
# 用途：换了 USB 口、或者第一次在新板子上装机时，用它读出真实的 ID_PATH，
# 抄进 bind_devices.sh 顶部那三个常量。
#
#   bash find_devices.sh

set -uo pipefail

echo "===== 相机（video4linux）====="
for dev in /dev/video*; do
  [ -e "$dev" ] || continue
  node=$(basename "$dev")
  # index 1 是 metadata 节点，不取图，列出来只会干扰判断
  idx=$(cat "/sys/class/video4linux/$node/index" 2>/dev/null || echo "?")
  [ "$idx" = "0" ] || continue
  path=$(udevadm info -q property -n "$dev" 2>/dev/null | sed -n 's/^ID_PATH=//p')
  name=$(cat "/sys/class/video4linux/$node/name" 2>/dev/null)
  vid=$(udevadm info -q property -n "$dev" 2>/dev/null | sed -n 's/^ID_VENDOR_ID=//p')
  pid=$(udevadm info -q property -n "$dev" 2>/dev/null | sed -n 's/^ID_MODEL_ID=//p')
  echo "$dev  ID_PATH=$path  ($vid:$pid  $name)"
done

echo
echo "===== 串口（tty）====="
for dev in /dev/ttyACM* /dev/ttyUSB*; do
  [ -e "$dev" ] || continue
  path=$(udevadm info -q property -n "$dev" 2>/dev/null | sed -n 's/^ID_PATH=//p')
  serial=$(udevadm info -q property -n "$dev" 2>/dev/null | sed -n 's/^ID_SERIAL=//p')
  echo "$dev  ID_PATH=$path  ($serial)"
done

echo
echo "===== 已绑定的固定名 ====="
ls -l /dev/topcam /dev/wristcam /dev/follower 2>&1

echo
echo "两个相机是同型号时，只能靠口位置区分。分不清哪个是顶部哪个是腕部就拔一个再跑一次。"
