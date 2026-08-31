"""把 mjlab 的宇树 G1 动作跟踪任务包成与行走线同形状的接口。

和 `1_1_g1_walk_rl/env.py` 唯一的区别是任务目标：这里每一帧都要贴住一段给定的
参考动作，而不是跟上一个速度指令。接口刻意保持一致，三个训练脚本才能原样搬过来，
把「任务变难」这个变量单独隔离出来。

讲义对应：第14讲 6.6 节。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from mjlab.tasks.tracking.mdp import MotionCommand, MotionCommandCfg


def split_actor_critic_obs(obs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """把 mjlab 按组返回的观测拆成 actor 那份和 critic 那份。

    Args:
        obs: mjlab 返回的观测字典。

    Returns:
        (actor 观测, critic 观测)。
    """
    actor_obs = obs["actor"] if "actor" in obs else obs["policy"]
    return actor_obs, obs["critic"]


class MjlabTrackingEnv:
    """mjlab 的 G1 动作跟踪任务，包成与行走线同形状的接口。

    与行走环境的唯一区别是任务目标：这里每一帧都要贴住一段给定的参考动作，
    而不是跟上一个速度指令。接口保持一致，三个训练脚本才能原样搬过来。
    """
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
        """露出底层的 mjlab 环境对象。

        Returns:
            未经包装的 mjlab 环境。
        """
        return self._env

    def reset(self):
        """重置全部并行环境。

        Returns:
            (actor 观测, critic 观测)。
        """
        obs, _ = self._env.reset()
        return split_actor_critic_obs(obs)

    def get_observations(self):
        """取当前观测，不推进仿真。

        Returns:
            (actor 观测, critic 观测)。
        """
        obs = self._env.observation_manager.compute()
        return split_actor_critic_obs(obs)

    def step(self, actions: torch.Tensor):
        """把动作送进仿真，推进一个控制周期。

        Args:
            actions: 形状 (N, action_dim) 的关节目标量。

        Returns:
            (actor 观测, critic 观测, 奖励, done, 附加信息)。
        """
        obs, rewards, terminated, time_outs, extras = self._env.step(actions)
        dones = torch.logical_or(terminated, time_outs)
        actor_obs, critic_obs = split_actor_critic_obs(obs)
        return actor_obs, critic_obs, rewards, dones, {"time_outs": time_outs, "mjlab_extras": extras}

    def render(self):
        """渲染一帧画面，录视频时用。

        Returns:
            一帧 RGB 图像；未开启渲染时为 None。
        """
        return self._env.render()

    def close(self) -> None:
        """关闭仿真、释放显存。"""
        self._env.close()


BeyondMimicEnv = MjlabTrackingEnv
