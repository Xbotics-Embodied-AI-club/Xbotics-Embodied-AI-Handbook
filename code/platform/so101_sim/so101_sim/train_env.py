"""so101_sim 的 RL 训练入口：千级并行、GPU 常驻的 `TrainEnv`。

任务定义、奖励、成功判据、机器人资产全部复用 `_core._make_maniskill`（squint 的 8 个
ManiSkill 任务）——和 lerobot 评测口 `So101SimEnv` 共享同一个底层仿真定义，只是这里批量
跑千级环境、给策略吃降采样后的 RGB + 关节状态。

`obs_mode="visual"`：观测降采样 RGB + 开域随机化，供视觉策略（如 SAC）训练；
`obs_mode="state"`：只给关节状态向量，跳过渲染，供状态策略/调试用。

`TrainEnv` 的方法/属性签名与旧版 `rl/3_offpolicy/3_2_so101_offpolicy/env.py::SO101VisualEnv`
逐一致，训练脚本只需把 env 构造换成 `so101_sim.make_train_env(...)`，其余零改。
"""

from __future__ import annotations

import numpy as np  # noqa: F401  C 扩展 ABI 顺序：numpy 在 torch 前
import torch

from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from so101_sim._core import _make_maniskill
from utils import DownsampleObsWrapper, ColorJitterWrapper  # vendored squint（sys.path 由 __init__ 挂好）


class TrainEnv:
    """千级并行的 SO101 训练环境。

    visual: obs = {rgb (N,image_size,image_size,3) uint8, state (N,state_dim)}，action = (N,action_dim)。
    state:  obs = {state (N,state_dim)}（无 rgb），跳过渲染管线，供状态策略/调试用。
    """

    def __init__(self, task: str, num_envs: int, obs_mode: str = "visual", image_size: int = 16,
                 render_size: int = 128, domain_randomization: bool = True, device: str = "cuda"):
        self.device = device
        self.obs_mode = obs_mode

        if obs_mode == "visual":
            # 原始环境：开域随机化（robot/背景/物体扰动，sim2real 关键）；相机分辨率 render_size。
            raw = _make_maniskill(task, num_envs=num_envs, obs_mode="rgb+segmentation",
                                   sensor_size=render_size, render_mode="all",
                                   domain_randomization=domain_randomization)
            # 展平 RGB+state → 降采样到 image_size → 颜色抖动（进一步抗 sim2real 分布差）。
            env = FlattenRGBDObservationWrapper(raw, rgb=True, depth=False, state=True)
            if render_size != image_size:
                env = DownsampleObsWrapper(env, target_size=image_size)
            env = ColorJitterWrapper(env)
        elif obs_mode == "state":
            env = _make_maniskill(task, num_envs=num_envs, obs_mode="state",
                                   sensor_size=render_size, render_mode="all",
                                   domain_randomization=domain_randomization)
        else:
            raise NotImplementedError(f"obs_mode '{obs_mode}' 不支持；用 'visual' 或 'state'。")

        self.vec = ManiSkillVectorEnv(env, num_envs, ignore_terminations=True, record_metrics=True)

        self.num_envs = num_envs
        self.single_action_space = self.vec.unwrapped.single_action_space
        if obs_mode == "visual":
            self.state_dim = int(np.prod(self.vec.unwrapped.single_observation_space["state"].shape))
        else:
            self.state_dim = int(np.prod(self.vec.unwrapped.single_observation_space.shape))
        self.action_dim = int(np.prod(self.single_action_space.shape))
        self.obs = None

    def _format(self, raw_obs):
        if self.obs_mode == "visual":
            return raw_obs
        return {"state": raw_obs}

    def reset(self):
        raw_obs, _ = self.vec.reset()
        self.obs = self._format(raw_obs)
        return self.obs

    def step(self, action: torch.Tensor):
        raw_obs, reward, terminated, truncated, info = self.vec.step(action)
        success = info["success"].bool()
        self.obs = self._format(raw_obs)
        return self.obs, reward, terminated, truncated, success

    def close(self):
        self.vec.close()


def make_train_env(task: str, num_envs: int, obs_mode: str = "visual", image_size: int = 16,
                    domain_randomization: bool = True, device: str = "cuda") -> TrainEnv:
    """构造 RL 训练用的批量 SO101 环境（visual 或 state 观测），接口与旧版 SO101VisualEnv 一致。"""
    return TrainEnv(task, num_envs=num_envs, obs_mode=obs_mode, image_size=image_size,
                     domain_randomization=domain_randomization, device=device)
