"""生成重构前的物理基线：reset(seed=0) 后走 10 步固定动作，记末态关节角与物体位姿。

在重构前的代码树（git worktree @ 锚点 commit）上跑，产出交给重构后的回归测试逐位比对。
动作序列 0.02*(i%3-1) 是确定性的，不含随机数，所以两边可以严格对齐。
"""

import json
import pathlib

import numpy as np
import torch
import gymnasium as gym

import so101_sim  # noqa: F401  导入即注册

ENVS = ["SO101PickPlaceCube40-v1", "SO101PickPlaceCube20-v1", "SO101PickPlaceCylinder40-v1"]
OUT = pathlib.Path(__file__).resolve().parent / "physics_baseline.json"

base = {}
for env_id in ENVS:
    env = gym.make(env_id, num_envs=1, obs_mode="state", sim_backend="gpu",
                   render_mode="all", domain_randomization=False, max_episode_steps=100)
    env.reset(seed=0)
    u = env.unwrapped
    action = torch.zeros((1, u.single_action_space.shape[-1]), device=u.device)
    for i in range(10):
        action[:] = 0.02 * (i % 3 - 1)
        env.step(action)
    base[env_id] = {
        "qpos": u.agent.robot.get_qpos()[0].cpu().numpy().round(6).tolist(),
        "item_p": u.item.pose.p[0].cpu().numpy().round(6).tolist(),
        "item_q": u.item.pose.q[0].cpu().numpy().round(6).tolist(),
    }
    env.close()
    print(env_id, "ok")

OUT.write_text(json.dumps(base, indent=2))
print("baseline written:", OUT)
