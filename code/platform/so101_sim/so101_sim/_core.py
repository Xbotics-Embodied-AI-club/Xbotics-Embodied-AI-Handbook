"""共享的 ManiSkill 批量环境构造核。

`lerobot_env.So101SimEnv`（lerobot 单环境评测口）和 `wrappers.visual_rl_env`（RL 批量训练口）
都从这一句 `gym.make` 出发，只是 num_envs / obs_mode / 渲染参数不同——保证两条消费路径拿到的
是同一个底层仿真定义（同一批任务、同一份机器人资产），不会各自漂移出两套环境。
"""

from __future__ import annotations

import gymnasium as gym


def _make_maniskill(
    task: str,
    num_envs: int,
    obs_mode: str,
    sensor_size: int,
    render_mode: str,
    domain_randomization: bool = False,
):
    """构造一个原始（未包装）的批量 ManiSkill 环境，首维恒为 num_envs。"""
    return gym.make(
        task,
        num_envs=num_envs,
        obs_mode=obs_mode,
        sim_backend="gpu",
        render_mode=render_mode,
        domain_randomization=domain_randomization,
        sensor_configs=dict(width=sensor_size, height=sensor_size),
    )
