"""把 ManiSkill3 SO101 任务包装成 lerobot 评测用的单环境 gym.Env。

lerobot 的 `make_env` 对 `so101_sim` 走通用分支：先 `import so101_sim` 触发注册
（见 `__init__.py`），再用 `SyncVectorEnv` 把若干份本环境包成向量环境。ManiSkill 天生
是批量张量（首维 = num_envs），这里默认 num_envs=1、对首维取 `[0]`，对上层呈现成普通单
环境接口——和 lerobot 自带的 `LiberoEnv` 一样，评测代码无需知道底层是 ManiSkill。

`num_envs` 参数保留了泛化到批量的能力（obs 格式化 / 动作 reshape / reward / success 都按
batch 维处理），但 `num_envs=1` 时的行为与之前完全一致，不破坏 lerobot 评测契约。

这是本模块与仿真器之间**唯一的耦合点**：只有本文件依赖 `so101_sim.tasks` / ManiSkill，
其它一切（评测、rollout、换色、机器人伪装）都只依赖 `So101SimEnv`。
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from so101_sim._core import _make_maniskill


def _to_numpy(x: Any) -> np.ndarray:
    """ManiSkill 返回 GPU torch 张量；统一搬到 CPU numpy。"""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


class So101SimEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        task: str = "SO101PickPlaceCube40-v1",
        obs_type: str = "pixels_agent_pos",
        obs_mode: str = "rgb",
        render_mode: str = "rgb_array",
        observation_width: int = 128,
        observation_height: int = 128,
        episode_length: int = 400,
        num_envs: int = 1,
        **kwargs,
    ):
        super().__init__()
        self.task = task
        self.obs_type = obs_type
        self.render_mode = render_mode
        self.observation_width = observation_width
        self.observation_height = observation_height
        self.num_envs = num_envs
        # lerobot 的 rollout 用 env.call("_max_episode_steps") 界定单集步数上限。
        # 三个分发任务本身已注册 max_episode_steps=400，这里与之对齐。
        self._max_episode_steps = episode_length

        # 每个包装实例内部持有一个 num_envs 份的 ManiSkill 环境（GPU 后端、无头 RGB 渲染）。
        # 三个分发场景已在 import so101_sim 时注册；task 即其 id。
        self._env = _make_maniskill(
            task,
            num_envs=num_envs,
            obs_mode=obs_mode,
            sensor_size=observation_width,
            render_mode=render_mode,
        )

        # 相机名不写死：三个场景各有 top 与 wrist 两路，和真机数据集一致，
        # 直接取环境自己声明的传感器名，环境加相机或改名都不用回来改这里。
        self._camera_names = sorted(self._env.unwrapped._sensor_configs)

        # 观测空间：每路相机一张 RGB（+ 可选关节位置）。关节维度取自单环境动作/状态。
        image_space = spaces.Box(
            low=0, high=255, shape=(observation_height, observation_width, 3), dtype=np.uint8
        )
        single_act = getattr(self._env, "single_action_space", self._env.action_space)
        self._action_dim = int(np.prod(single_act.shape[-1:]))
        pixels_space = spaces.Dict({name: image_space for name in self._camera_names})
        if obs_type == "pixels":
            self.observation_space = spaces.Dict({"pixels": pixels_space})
        elif obs_type == "pixels_agent_pos":
            self.observation_space = spaces.Dict(
                {
                    "pixels": pixels_space,
                    "agent_pos": spaces.Box(
                        low=-np.inf, high=np.inf, shape=(self._action_dim,), dtype=np.float32
                    ),
                }
            )
        else:
            raise NotImplementedError(
                f"obs_type '{obs_type}' 不支持；用 'pixels' 或 'pixels_agent_pos'。"
            )

        self.action_space = spaces.Box(
            low=np.asarray(single_act.low, dtype=np.float32).reshape(-1),
            high=np.asarray(single_act.high, dtype=np.float32).reshape(-1),
            shape=(self._action_dim,),
            dtype=np.float32,
        )

    def _format_raw_obs(self, raw_obs: dict) -> dict:
        # ManiSkill 观测：sensor_data.<相机名>.rgb (N,H,W,3) / agent.noisy_qpos (N,dof)。
        # num_envs=1 时对首维取 [0]，还原成之前的单环境形状；否则整批返回。
        images = {}
        for name in self._camera_names:
            rgb = _to_numpy(raw_obs["sensor_data"][name]["rgb"]).astype(np.uint8)
            images[name] = rgb[0] if self.num_envs == 1 else rgb
        if self.obs_type == "pixels":
            return {"pixels": images}
        agent_pos = _to_numpy(raw_obs["agent"]["noisy_qpos"]).astype(np.float32)
        agent_pos = agent_pos[0] if self.num_envs == 1 else agent_pos
        return {"pixels": images, "agent_pos": agent_pos}

    def reset(self, seed: int | None = None, **kwargs) -> tuple[dict, dict]:
        super().reset(seed=seed)
        raw_obs, _ = self._env.reset(seed=seed)
        info = {"is_success": False} if self.num_envs == 1 else {"is_success": np.zeros(self.num_envs, dtype=bool)}
        return self._format_raw_obs(raw_obs), info

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict[str, Any]]:
        # 上层给单环境动作 (dof,)（或批量 (num_envs,dof)）；ManiSkill 需要批量 (num_envs,dof)。
        act = np.asarray(action, dtype=np.float32).reshape(self.num_envs, self._action_dim)
        raw_obs, reward, terminated, truncated, info = self._env.step(act)

        success = _to_numpy(info["success"]).reshape(self.num_envs).astype(bool)
        reward_batch = _to_numpy(reward).reshape(self.num_envs).astype(np.float32)
        terminated_batch = _to_numpy(terminated).reshape(self.num_envs).astype(bool) | success
        truncated_batch = _to_numpy(truncated).reshape(self.num_envs).astype(bool)
        observation = self._format_raw_obs(raw_obs)

        if self.num_envs != 1:
            out_info: dict[str, Any] = {"task": self.task, "is_success": success, "done": terminated_batch}
            return observation, reward_batch, terminated_batch, truncated_batch, out_info

        is_success = bool(success[0])
        reward_out = float(reward_batch[0])
        terminated_out = bool(terminated_batch[0])
        truncated_out = bool(truncated_batch[0])
        out_info = {"task": self.task, "is_success": is_success, "done": terminated_out}
        if terminated_out or truncated_out:
            # 与 LiberoEnv 一致：给出 final_info，并就地 reset 让下一集从头开始。
            out_info["final_info"] = {"task": self.task, "is_success": is_success, "done": True}
            self.reset()
        return observation, reward_out, terminated_out, truncated_out, out_info

    def render(self) -> np.ndarray:
        frame = _to_numpy(self._env.render())
        return frame[0].astype(np.uint8) if frame.ndim == 4 else frame.astype(np.uint8)

    def close(self):
        self._env.close()
