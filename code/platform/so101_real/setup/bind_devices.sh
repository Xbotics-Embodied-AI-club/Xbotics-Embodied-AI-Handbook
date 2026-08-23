#!/usr/bin/env bash
# Jetson Orin Nano 固定设备名：按 USB 物理口位置绑定，不认设备型号。
#
# 位置定死，以后都按这个插：
#   口 2.4 -> /dev/topcam      顶部相机（俯视工作区）
#   口 2.3 -> /dev/wristcam    腕部相机（跟着夹爪动）
#   口 2.2 -> /dev/follower    SO101 从臂串口
#
# 为什么按口位置而不按 vendor/product：手上两个相机是同型号同序列号（1e45:8022），
# 型号规则根本分不开它们；而且换个牌子的相机插同一个口，还应该拿到同一个名字。
# 所以规则只匹配 ID_PATH（USB 口位置），换设备不用改脚本，换口才要改。
#
# 相机那两条额外卡 ATTR{index}=="0"：一个 UVC 相机会枚举出两个 video 节点，
# index 0 是取图的，index 1 是 metadata。不卡这个会有一半概率绑到不能取图的那个。
#
#   sudo bash bind_devices.sh              写规则并立即生效
#   bash bind_devices.sh --dry-run         只打印规则，不写盘

set -euo pipefail

TOPCAM_PORT="platform-3610000.usb-usb-0:2.4:1.0"
WRISTCAM_PORT="platform-3610000.usb-usb-0:2.3:1.0"
FOLLOWER_PORT="platform-3610000.usb-usb-0:2.2:1.0"

# 口位置怎么来的：把设备插好，跑 find_devices.sh，把它打印的 ID_PATH 抄到上面三行。
# 哪个口是顶部哪个是腕部，光看 ID_PATH 看不出来（同型号相机）——绑完跑
# vla/2_data_collection/2_2_so101_setup/check_cameras.py 看两张样张，装反了就把这两行的值对调。
# 上面这组值是实测对好的。

camera_rules="\
SUBSYSTEM==\"video4linux\", KERNEL==\"video*\", ENV{ID_PATH}==\"$TOPCAM_PORT\", ATTR{index}==\"0\", SYMLINK+=\"topcam\", MODE=\"0666\"
SUBSYSTEM==\"video4linux\", KERNEL==\"video*\", ENV{ID_PATH}==\"$WRISTCAM_PORT\", ATTR{index}==\"0\", SYMLINK+=\"wristcam\", MODE=\"0666\""

# ID_MM_DEVICE_IGNORE 让 ModemManager 别去探这个串口 —— 它一探就会往舵机总线发 AT 指令，
# 抢占几秒串口，表现是连臂时随机报 timeout。
serial_rules="\
SUBSYSTEM==\"tty\", ENV{ID_PATH}==\"$FOLLOWER_PORT\", SYMLINK+=\"follower\", MODE=\"0666\", ENV{ID_MM_DEVICE_IGNORE}=\"1\""

camera_rule_file="/etc/udev/rules.d/99-so101-nano-camera.rules"
serial_rule_file="/etc/udev/rules.d/99-so101-nano-serial.rules"

echo "相机绑定（按 USB 口位置）:"
echo "  $TOPCAM_PORT   -> /dev/topcam"
echo "  $WRISTCAM_PORT -> /dev/wristcam"
echo "从臂串口绑定:"
echo "  $FOLLOWER_PORT -> /dev/follower"
echo

if [[ "${1:-}" == "--dry-run" ]]; then
  echo "--- $camera_rule_file ---"; printf '%s\n' "$camera_rules"
  echo "--- $serial_rule_file ---"; printf '%s\n' "$serial_rules"
  exit 0
fi

printf '%s\n' "$camera_rules" | sudo tee "$camera_rule_file" >/dev/null
printf '%s\n' "$serial_rules" | sudo tee "$serial_rule_file" >/dev/null

sudo udevadm control --reload-rules
sudo udevadm trigger --subsystem-match=video4linux
sudo udevadm trigger --subsystem-match=tty
udevadm settle || true

echo "已写入规则。当前链接："
ls -l /dev/topcam /dev/wristcam /dev/follower 2>&1
