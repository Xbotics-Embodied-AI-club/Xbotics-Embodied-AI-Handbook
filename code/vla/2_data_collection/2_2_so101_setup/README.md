# SO-101 硬件绑定

给主从臂串口和相机装 udev 固定名，之后遥操/录制脚本里的设备名才稳定：

- `bind_uarm_serial_port.sh` — 读 USB 序列号，绑 `/dev/uarmLeft`、`/dev/uarmRight`（或 ttyLeader/ttyFollower）
- `bind_uarm_serial_port_s100.sh` / `bind_camera_s100.sh` — RDK S100 板端的固定 USB 位版本（`/dev/top_camera`、`/dev/wrist_camera`）

均支持 `--dry-run` 先看规则再写入。
