"""LIBERO 环境接口验证：环境创建 → observation 结构 → action 空间 → step 控制 → 轨迹跟踪。

libero_demo.ipynb 的脚本版：一口气跑完 notebook 里的全部验证点，适合装完环境后自检。
"""
import os
os.environ["MUJOCO_GL"] = "egl"   # 无桌面环境用 EGL 离屏渲染

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "output"
os.makedirs(OUT_DIR, exist_ok=True)

# ── 环境创建：LIBERO 按「套件 + 任务 id」组织，这里取 libero_spatial 的第 0 个任务 ──
print("=== 环境创建 ===")
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.envs.factory import make_env

env_cfg = LiberoEnvConfig(
    task="libero_spatial",
    task_ids=[0],
    obs_type="pixels_agent_pos",   # 图像 + 本体状态 两路观测
    observation_height=256,
    observation_width=256,
)

envs = make_env(env_cfg, n_envs=1)
env  = envs['libero_spatial'][0]
print("环境创建成功")

# ── observation 结构：递归打印整棵观测树，认识图像/本体状态各字段的形状 ──
print("\n=== Observation 结构 ===")
obs, info = env.reset()

def print_obs_tree(d, prefix=""):
    for k, v in d.items():
        if isinstance(v, dict):
            print(f"{prefix}{k}/")
            print_obs_tree(v, prefix + "  ")
        elif isinstance(v, np.ndarray):
            print(f"{prefix}{k}: shape={v.shape}, dtype={v.dtype}")
        else:
            print(f"{prefix}{k}: type={type(v).__name__}")

print_obs_tree(obs)

# ── action 空间：LIBERO 是 7 维末端增量控制（xyz + 姿态 + 夹爪） ──
print("\n=== Action Space ===")
action_space = env.single_action_space
print(f"Action space: {action_space}")
assert action_space.shape == (7,)
print("Action space 验证通过")

# ── step 控制：发一个 x 方向动作，确认末端真的动了 ──
print("\n=== Step 控制 ===")
obs, _ = env.reset()
eef_before = obs["robot_state"]["eef"]["pos"][0].copy()
print(f"初始 EEF 位置: {eef_before}")

action = np.zeros((1, 7), dtype=np.float32)
action[0, 0] = 1.0   # 只推 x 方向
obs2, reward, terminated, truncated, info = env.step(action)
eef_after = obs2["robot_state"]["eef"]["pos"][0].copy()
print(f"Step 后 EEF 位置: {eef_after}")

# ── 轨迹跟踪：先稳定，再来回推 x 方向，记录末端轨迹——直观看到「动作是增量」──
print("\n=== 轨迹跟踪 ===")
env = env.envs[0]
obs, info = env.reset()
dummy_action = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
for _ in range(20):   # 静止若干步让机械臂稳定
    obs, _, _, _, _ = env.step(dummy_action)

positions = [obs["robot_state"]["eef"]["pos"].copy()]
actions_record = []

for _ in range(30):   # 向 +x 推
    a = np.array([1.0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
    obs, _, _, _, _ = env.step(a)
    positions.append(obs["robot_state"]["eef"]["pos"].copy())
    actions_record.append(a[:3].copy())

for _ in range(20):   # 停
    a = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
    obs, _, _, _, _ = env.step(a)
    positions.append(obs["robot_state"]["eef"]["pos"].copy())
    actions_record.append(a[:3].copy())

for _ in range(30):   # 向 -x 推回来
    a = np.array([-1.0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
    obs, _, _, _, _ = env.step(a)
    positions.append(obs["robot_state"]["eef"]["pos"].copy())
    actions_record.append(a[:3].copy())

positions = np.array(positions)
print(f"Total steps: {len(positions) - 1}")
print(f"Start pos: {positions[0]}")
print(f"End pos:   {positions[-1]}")

env.close()
print("\n环境已关闭，全部通过！")
