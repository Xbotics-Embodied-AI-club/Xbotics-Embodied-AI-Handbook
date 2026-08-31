"""确认两路相机都能按 640×480@30 取图，并各存一张样张。

对应第9讲《操作数据闭环》4.2 节的备注——正式录之前先确认相机这一路是好的。

跑这个的目的不是"能打开吗"，而是**两路同时开的时候还能不能保住 30fps**。
两个相机插在同一个 USB 2.0 hub 上，裸 YUYV 的带宽是塞不下两路的，必须走 MJPG 让相机
自己先压一遍。所以下面固定用 MJPG——真掉到 20fps 以下，先看是不是插到了同一个 hub。

    uv run python vla/2_data_collection/2_2_so101_setup/check_cameras.py
"""

from pathlib import Path
import time

import cv2

# 设备名来自 udev 绑定，不同机架绑出来的名字不一样，跑之前先确认自己这台绑的是哪一套：
#   platform/so101_real/setup/bind_devices.sh     -> /dev/topcam, /dev/wristcam
#   platform/so101_real/setup/bind_camera_s100.sh -> /dev/top_camera, /dev/wrist_camera
# 下面这条按 bind_devices.sh 那套写；换机架就改这两个设备名（键名 top / wrist 不要改，
# 它们要和 2_3_teleop_record/ 的录制命令写成同一套，否则落盘字段名对不上）。
CAMERAS = {"top": "/dev/topcam", "wrist": "/dev/wristcam"}
WIDTH, HEIGHT, FPS = 640, 480, 30
WARMUP_FRAMES = 10
MEASURE_FRAMES = 90

OUT_DIR = Path(__file__).resolve().parents[2] / "result"
OUT_DIR.mkdir(parents=True, exist_ok=True)

caps = {}
for name, dev in CAMERAS.items():
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    if not cap.isOpened():
        raise SystemExit(
            f"打不开 {name} ({dev})，先跑 platform/so101_real/setup/ 下对应本机架的绑定脚本，"
            "或把上面 CAMERAS 里的设备名改成自己这套"
        )
    caps[name] = cap
    print(f"{name}: {dev} 已打开")

# 前几帧相机还在自动曝光收敛，不计入帧率
for _ in range(WARMUP_FRAMES):
    for cap in caps.values():
        cap.read()

print(f"\n两路同时取 {MEASURE_FRAMES} 帧，测实际帧率……")
t0 = time.perf_counter()
last = {}
for _ in range(MEASURE_FRAMES):
    for name, cap in caps.items():
        ok, frame = cap.read()
        if not ok:
            raise SystemExit(f"{name} 取帧失败")
        last[name] = frame
elapsed = time.perf_counter() - t0

fps = MEASURE_FRAMES / elapsed
print(f"耗时 {elapsed:.2f}s，每路 {fps:.1f} fps")

for name, frame in last.items():
    h, w = frame.shape[:2]
    path = OUT_DIR / f"{name}.jpg"
    cv2.imwrite(str(path), frame)
    print(f"{name}: {w}×{h}，样张 {path}")

for cap in caps.values():
    cap.release()

if fps < 25:
    print("\n帧率不足 25：两个相机很可能插在同一个 USB hub 上，换一个到另一组口。")
else:
    print("\n两路都到 30fps 附近，可以部署。")
print("顶部/腕部装反了的话，看 result/ 里两张图哪张是俯视，然后对调 USB 插口。")
