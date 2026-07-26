from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from mjlab.tasks.tracking.mdp import MotionCommand, MotionCommandCfg


def split_actor_critic_obs(obs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    actor_obs = obs["actor"] if "actor" in obs else obs["policy"]
    return actor_obs, obs["critic"]


class MjlabTrackingEnv:
    def __init__(
        self,
        motion_file: str | Path,
        num_envs: int = 4096,
        device: str = "cuda:0",
        episode_length_s: float = 10.0,
        seed: int | None = None,
        render_mode: str | None = None,
        show_reference_ghost: bool = False,
    ) -> None:
        cfg = unitree_g1_flat_tracking_env_cfg(has_state_estimation=True)
        cfg.scene.num_envs = num_envs
        cfg.episode_length_s = episode_length_s
        cfg.seed = seed
        cfg.viewer.max_extra_envs = 0
        motion_cfg = cfg.commands["motion"]
        assert isinstance(motion_cfg, MotionCommandCfg)
        motion_cfg.motion_file = str(Path(motion_file))
        motion_cfg.debug_vis = show_reference_ghost

        resolved_device = device if torch.cuda.is_available() or device == "cpu" else "cpu"
        self._env = ManagerBasedRlEnv(cfg=cfg, device=resolved_device, render_mode=render_mode)
        self._motion_command = cast(MotionCommand, self._env.command_manager.get_term("motion"))
        self.device = torch.device(self._env.device)
        self.num_envs = self._env.num_envs
        self.action_dim = int(self._env.single_action_space.shape[0])
        self.motion = self._motion_command.motion
        self.metadata = self._env.metadata

    @property
    def unwrapped(self) -> Any:
        return self._env

    def reset(self):
        obs, _ = self._env.reset()
        return split_actor_critic_obs(obs)

    def get_observations(self):
        obs = self._env.observation_manager.compute()
        return split_actor_critic_obs(obs)

    def step(self, actions: torch.Tensor):
        obs, rewards, terminated, time_outs, extras = self._env.step(actions)
        dones = torch.logical_or(terminated, time_outs)
        actor_obs, critic_obs = split_actor_critic_obs(obs)
        return actor_obs, critic_obs, rewards, dones, {"time_outs": time_outs, "mjlab_extras": extras}

    def render(self):
        return self._env.render()

    def close(self) -> None:
        self._env.close()


BeyondMimicEnv = MjlabTrackingEnv
