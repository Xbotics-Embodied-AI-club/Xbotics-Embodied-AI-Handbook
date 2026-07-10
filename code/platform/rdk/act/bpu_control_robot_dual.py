"""S100 实物测试：双相机 + 修正版模型 跑 ACT 实机控制。

- 复用官方 rdk_LeRobot_tools 的 BPUACTPolicy / build_policy_batch / sanity_check_policy（不另写推理）。
- 用**修正版模型**（top/wrist 归一化已对调）→ 按**正常物理映射**接相机：
  模型 `top` ← 物理俯视相机 /dev/video0；模型 `wrist` ← 物理夹爪相机 /dev/video2。
- DRY_RUN=True：只预测、不发指令（机械臂只被 connect 上电保持，不主动运动）——先确认相机+臂+模型联通。
  确认无误后把 DRY_RUN 改 False 跑真动控制环（MAX_RELATIVE_TARGET 钳制每步步长保安全）。
"""

import logging
import os
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
TOOLS_DIR = os.path.join(BASE, "rdk_tools_s600")
os.environ.setdefault("HF_LEROBOT_HOME", os.path.join(BASE, "hf_lerobot"))  # 标定 so100_follower.json 在此
sys.path.insert(0, TOOLS_DIR)

from bpu_control_robot import (
    BPUACTPolicy,
    build_policy_batch,
    detect_cameras_from_model,
    detect_n_action_steps,
    sanity_check_policy,
)
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig

# —— 配置 ——
BPU_ACT_PATH = os.path.join(BASE, "bpu_output_fixed")     # 修正版模型（正常物理映射）
ROBOT_PORT = "/dev/ttyACM0"
CAM_DEV = {"top": "/dev/video0", "wrist": "/dev/video2"}  # 模型top←俯视(Realtek) / 模型wrist←夹爪(Sonix)
FPS = 30
INFERENCE_TIME = 20                # 秒
MAX_RELATIVE_TARGET = 8.0          # 每步步长钳制(度)，首测保守；None=不钳制
DRY_RUN = os.environ.get("DRY_RUN", "1") != "0"   # 默认干跑(安全)；DRY_RUN=0 才真发指令

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
log = logging.getLogger("dual")

camera_names = detect_cameras_from_model(BPU_ACT_PATH)
n_action_steps = detect_n_action_steps(BPU_ACT_PATH, None)
log.info("model=%s cameras=%s n_action_steps=%d DRY_RUN=%s", BPU_ACT_PATH, camera_names, n_action_steps, DRY_RUN)

policy = BPUACTPolicy(BPU_ACT_PATH, n_action_steps, camera_names)
sanity_check_policy(policy, camera_names)
policy.reset()

cameras = {
    name: OpenCVCameraConfig(index_or_path=CAM_DEV[name], width=640, height=480, fps=FPS, warmup_s=10, fourcc="MJPG")
    for name in camera_names
}
robot = SO100Follower(
    SO100FollowerConfig(
        port=ROBOT_PORT,
        id="so100_follower",
        max_relative_target=(None if DRY_RUN else MAX_RELATIVE_TARGET),
        cameras=cameras,
    )
)
robot.connect()
motor_names = list(robot.bus.motors.keys())
log.info("connected. motors=%s", motor_names)

try:
    for tick in range(INFERENCE_TIME * FPS):
        t0 = time.perf_counter()
        obs = robot.get_observation()
        batch = build_policy_batch(obs, policy, motor_names)
        action_values = policy.select_action(batch)
        if tick == 0 or (tick + 1) % FPS == 0:
            cur = [round(float(obs[f"{m}.pos"]), 1) for m in motor_names]
            cmd = [round(float(action_values[i].item()), 1) for i in range(len(motor_names))]
            log.info("tick=%d  当前臂位=%s  模型指令=%s", tick, cur, cmd)
        if not DRY_RUN:
            robot.send_action({f"{m}.pos": action_values[i].item() for i, m in enumerate(motor_names)})
        time.sleep(max(0.0, 1.0 / FPS - (time.perf_counter() - t0)))
finally:
    robot.disconnect()
    log.info("disconnected.")
