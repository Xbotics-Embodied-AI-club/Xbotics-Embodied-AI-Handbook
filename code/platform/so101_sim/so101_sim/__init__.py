"""so101_sim —— SO101 机械臂的 ManiSkill3 仿真环境包。

对外两个入口：

1. **原生**（数据生产、RL 都走这个）——ManiSkill 的批量环境，观测是 GPU 上的
   torch tensor，首维是 num_envs::

       import so101_sim
       env = gym.make("SO101PickPlaceCube40-v1", num_envs=64, sim_backend="gpu")

   RL 训练要用的降采样 + 颜色抖动 + 向量化，走便利函数 `visual_rl_env`；只要关节状态、
   跳过渲染管线的场景走 `state_rl_env`::

       from so101_sim import visual_rl_env, state_rl_env
       env = visual_rl_env("SO101PickPlaceCube40-v1", num_envs=64)
       env = state_rl_env("SO101PickPlaceCube40-v1", num_envs=64)

2. **lerobot 评测**——标准单环境 gym.Env，观测转成 numpy 与 lerobot 的
   ``{"agent_pos": ..., "pixels": {...}}`` 约定::

       lerobot-eval --env.type=so101_sim --env.task=SO101PickPlaceCube40-v1

   （`--env.type=so101_sim` 这个选项由 `platform/lerobot/0004-so101-sim-env.patch`
   注册；不打该补丁时入口 1 仍可独立使用。）

import 本包即完成注册：三个分发任务 + KIT 机器人 + lerobot 评测口。
"""

from gymnasium.envs.registration import register

# 导入即向 ManiSkill 注册 KIT 机器人（KIT 版与真机速度包线版）
from so101_sim.robots import so101_kit as _so101_kit  # noqa: F401
from so101_sim.robots import so101_kit_slow as _so101_kit_slow  # noqa: F401

# 导入即向 ManiSkill 注册三个分发环境
from so101_sim import envs as _envs  # noqa: F401

from so101_sim.lerobot_env import So101SimEnv
from so101_sim.wrappers import visual_rl_env, state_rl_env

register(id="SO101Sim-v1", entry_point="so101_sim.lerobot_env:So101SimEnv")

__all__ = ["So101SimEnv", "visual_rl_env", "state_rl_env"]
