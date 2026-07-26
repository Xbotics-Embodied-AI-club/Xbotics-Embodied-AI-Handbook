from __future__ import annotations

from typing import Any

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg


def split_actor_critic_obs(obs):
    """mjlab 把观测按组返回；actor 用普通观测，critic 用特权观测。"""
    actor_obs = obs["actor"] if "actor" in obs else obs["policy"]
    return actor_obs, obs["critic"]


class G1WalkEnv:
    """G1 平地速度跟随（基础行走）环境，三个算法版本共用。

    指令是一个目标速度（前进 / 转向），奖励让机器人跟上这个速度并保持站立。
    actor 看普通观测，critic 额外看到脚部接触等特权信息（训练时用，部署不需要）。
    """

    def __init__(
        self,
        num_envs=4096,
        device="cuda:0",
        episode_length_s=20.0,
        seed=None,
        render_mode=None,
    ):
        cfg = unitree_g1_flat_env_cfg()
        cfg.scene.num_envs = num_envs
        cfg.episode_length_s = episode_length_s
        cfg.seed = seed
        cfg.viewer.max_extra_envs = 0

        resolved_device = device if torch.cuda.is_available() or device == "cpu" else "cpu"
        self._env = ManagerBasedRlEnv(cfg=cfg, device=resolved_device, render_mode=render_mode)
        self.device = torch.device(self._env.device)
        self.num_envs = self._env.num_envs
        self.action_dim = int(self._env.single_action_space.shape[0])
        self.metadata = self._env.metadata

        obs, critic_obs = self.get_observations()
        self.obs_dim = int(obs.shape[1])
        self.critic_obs_dim = int(critic_obs.shape[1])

    @property
    def unwrapped(self) -> Any:
        return self._env

    def reset(self):
        obs, _ = self._env.reset()
        return split_actor_critic_obs(obs)

    def get_observations(self):
        obs = self._env.observation_manager.compute()
        return split_actor_critic_obs(obs)

    def step(self, actions):
        obs, rewards, terminated, time_outs, extras = self._env.step(actions)
        dones = torch.logical_or(terminated, time_outs)
        actor_obs, critic_obs = split_actor_critic_obs(obs)
        return actor_obs, critic_obs, rewards, dones, {"time_outs": time_outs, "mjlab_extras": extras}

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()
