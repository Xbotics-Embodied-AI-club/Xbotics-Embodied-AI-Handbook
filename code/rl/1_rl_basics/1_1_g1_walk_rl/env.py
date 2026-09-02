"""把 mjlab 的宇树 G1 平地速度跟随任务包成课程用的薄接口。

真正的物理、观测、奖励、终止条件全在 mjlab 里，这里只做两件事：把配置里的
并行环境数、回合长度暴露出来，以及把 mjlab 按组返回的观测拆成 actor 和 critic
各自那一份。三个训练版本共用本文件，保证对照时环境完全一致。

讲义对应：第14讲 4.1 节（认识 mjlab）与 5.4 节（代码走读）。
"""
from __future__ import annotations

from typing import Any

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg


def split_actor_critic_obs(obs):
    """把 mjlab 按组返回的观测拆成 actor 那份和 critic 那份。

    两份不一样是有意的：critic 只在训练时存在，可以多看仿真器里的"天眼"信息
    （脚部接触之类），把分打得更准；actor 部署时要单独带走，只能看真机拿得到的。

    Args:
        obs: mjlab 返回的观测字典，键为观测组名。

    Returns:
        (actor 观测, critic 观测) 两个张量。
    """
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
        """露出底层的 mjlab 环境对象，供录像等需要原始接口的场合使用。

        Returns:
            未经包装的 mjlab 环境。
        """
        return self._env

    def reset(self):
        """重置全部并行环境，开始新一轮回合。

        Returns:
            (actor 观测, critic 观测)。
        """
        obs, _ = self._env.reset()
        return split_actor_critic_obs(obs)

    def get_observations(self):
        """取当前时刻的观测，不推进仿真。

        采样循环开头要先拿到"这一步看到什么"才能出动作，所以需要一个只读的入口。

        Returns:
            (actor 观测, critic 观测)。
        """
        obs = self._env.observation_manager.compute()
        return split_actor_critic_obs(obs)

    def step(self, actions):
        """把动作送进仿真，推进一个控制周期。

        mjlab 把"摔倒终止"和"超时截断"分成两个信号返回，这里合并成一个 done 交给
        采样循环——两者对回报截断的作用是一样的。

        Args:
            actions: 形状 (N, action_dim) 的关节目标量。

        Returns:
            (actor 观测, critic 观测, 奖励, done, 附加信息)。附加信息里保留了
            未合并的 `time_outs`，PPO 版本要用它给超时的回合补一段自举回报。
        """
        obs, rewards, terminated, time_outs, extras = self._env.step(actions)
        dones = torch.logical_or(terminated, time_outs)
        actor_obs, critic_obs = split_actor_critic_obs(obs)
        return actor_obs, critic_obs, rewards, dones, {"time_outs": time_outs, "mjlab_extras": extras}

    def render(self):
        """渲染一帧画面，录 rollout 视频时用。

        Returns:
            一帧 RGB 图像；未开启渲染时为 None。
        """
        return self._env.render()

    def close(self):
        """关闭仿真、释放显存。"""
        self._env.close()
