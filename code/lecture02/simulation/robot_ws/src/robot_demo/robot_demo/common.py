"""Shared constants for robot_demo nodes.

集中定义关节数量、名称顺序、关节限位、默认目标、频率与各阈值，
保证 6 个节点对「六关节、顺序固定、维度一致」的理解完全一致
（对应本讲 README §6 约束 2：动作维度一致，顺序固定）。

接真实硬件时，只需修改本文件的关节名称 / 限位，其余节点无需改动。
"""

import numpy as np

# 关节数量（六关节机械臂）
NUM_JOINTS = 6

# 关节名称与顺序（顺序固定，所有节点共用，绝不能各自硬编码）
JOINT_NAMES = [f"joint_{i}" for i in range(1, NUM_JOINTS + 1)]

# 关节限位（弧度），对每个关节统一 (lower, upper)
JOINT_LIMITS = (-np.pi, np.pi)

# 默认目标关节位置（绝对位置，弧度），顺序与 JOINT_NAMES 一致
DEFAULT_TARGET = [0.0, -0.4, 0.8, 0.0, 0.4, 0.0]

# robot_state_node 状态发布频率（Hz）
STATE_RATE = 20.0

# target_publisher 目标发布周期（秒）
TARGET_PERIOD = 1.0

# 策略增益与单步限幅（弧度）
GAIN = 0.5
MAX_STEP = 0.05

# 到达判定阈值：最大关节误差（弧度）小于该值判 reached
REACH_TOLERANCE = 0.02

# 状态新鲜度阈值（纳秒）：超过 200ms 判 stale_state
STALE_STATE_NS = 200_000_000

# 任务超时（秒）：超过该时长未 reached 判 timeout
TASK_TIMEOUT = 15.0
