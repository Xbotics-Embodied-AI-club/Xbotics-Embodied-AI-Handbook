"""RL 训练用的观测包装与便利构造器。

这些包装不是格式转换，是 sim-to-real 的实质手段：

- **降采样到 16px**：低分辨率反而让视觉策略训得快、迁移得好（源自 squint 的做法）。
- **颜色抖动**：让策略不去死记仿真的配色，抗真实相机的色差。
- **``ignore_terminations=True``**：RL 要按固定步长稳定收集，不要一成功就断——
  这和评测时"成功即结束"的语义正好相反，所以两个入口不能共用同一套包装。

`visual_rl_env` 返回的是 ManiSkill 的标准 `ManiSkillVectorEnv`，不是自定义类型：
用的人拿到的还是原生生态里的对象，`reset`/`step` 就是 ManiSkill 的语义。
"""

from __future__ import annotations

import numpy as np  # noqa: F401  C 扩展 ABI 顺序：numpy 必须在 torch 之前
import torch
import torch.nn.functional as F
import torchvision
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
import gymnasium as gym

from so101_sim._core import _make_maniskill

# `DownsampleObsWrapper` 与 `ColorJitterWrapper` 取自 squint 的 vendored utils.py（MIT），
# 已转为本项目自维护。`DownsampleObsWrapper` 与原文件逐字等价；`ColorJitterWrapper` 在
# 「多相机拼接后按每 3 通道一组分别抖动」这一点上偏离了原文件（原文件只处理单相机 3 通道
# 输入），其余逻辑（batch 维处理、归一化/反归一化往返、默认抖动幅度）保持一致，改动原因见
# 该类的类文档。


class DownsampleObsWrapper(gym.ObservationWrapper):
    """把 RGB 观测从 render_size 降采样到 target_size（area 插值）。

    输入约定为 (B, H, W, C) 格式。
    """

    def __init__(self, env, target_size):
        super().__init__(env)
        self.target_size = target_size
        # 更新观测空间
        old_rgb_space = self.observation_space['rgb']
        C = old_rgb_space.shape[-1]
        self.observation_space['rgb'] = gym.spaces.Box(
            low=0, high=255, shape=(target_size, target_size, C), dtype=old_rgb_space.dtype
        )

    def observation(self, obs):
        rgb = obs['rgb']  # (B, H, W, C) 或 (H, W, C)
        if rgb.shape[-2] == self.target_size:
            return obs  # 已经是目标尺寸

        # 兼容有无 batch 维两种情况
        squeeze = rgb.dim() == 3
        if squeeze:
            rgb = rgb.unsqueeze(0)

        # (B, H, W, C) -> (B, C, H, W)，供 interpolate 使用
        rgb = rgb.permute(0, 3, 1, 2)
        rgb = F.interpolate(rgb.float(), size=(self.target_size, self.target_size), mode='area').to(torch.uint8)
        # (B, C, H, W) -> (B, H, W, C)
        rgb = rgb.permute(0, 2, 3, 1)

        if squeeze:
            rgb = rgb.squeeze(0)

        obs['rgb'] = rgb
        return obs


class ColorJitterWrapper(gym.ObservationWrapper):
    """对 RGB 观测做随机颜色抖动，增强 sim2real 鲁棒性。

    输入约定为 (B, H, W, C) 格式，通道数 C 必须是 3 的整数倍——每 3 通道对应一路相机
    （`FlattenRGBDObservationWrapper` 把多路相机的 RGB 沿通道维拼接后就是这种排布）。
    top 和 wrist 是两个物理上独立的相机，各自的成色偏差、曝光噪声互不相关，所以每一路
    单独抽一次随机抖动参数才是与物理对应的增强；如果两路共用同一次抖动，等于假设两个
    传感器会同步产生完全相同的色差，这在真实标定里并不成立。因此这里对每一组 3 通道
    分别调用一次 `self.jitter`（每次调用都会重新采样亮度/对比度/饱和度/色调），而不是
    采样一次参数后复用。
    """

    def __init__(self, env, brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05):
        super().__init__(env)
        self.jitter = torchvision.transforms.ColorJitter(brightness, contrast, saturation, hue)

    def observation(self, obs):
        rgb = obs['rgb']  # (B, H, W, C) 或 (H, W, C)，uint8，C 是 3 的整数倍

        # 兼容有无 batch 维两种情况
        squeeze = rgb.dim() == 3
        if squeeze:
            rgb = rgb.unsqueeze(0)

        # (B, H, W, C) -> (B, C, H, W)，供 ColorJitter 使用
        rgb = rgb.permute(0, 3, 1, 2)
        rgb = rgb.float() / 255.0

        num_channels = rgb.shape[1]
        if num_channels % 3 != 0:
            raise ValueError(
                f"ColorJitterWrapper 只接受通道数是 3 的整数倍的输入（每 3 通道对应一路"
                f"相机），但收到了 {num_channels} 通道。"
            )
        # 每 3 通道一组（对应一路相机），各自独立抽样抖动参数，互不共享。
        groups = [self.jitter(rgb[:, i:i + 3]) for i in range(0, num_channels, 3)]
        rgb = torch.cat(groups, dim=1)

        # (B, C, H, W) -> (B, H, W, C)
        rgb = rgb.permute(0, 2, 3, 1)

        # 转回 uint8
        rgb = (rgb.clamp(0, 1) * 255).to(torch.uint8)

        if squeeze:
            rgb = rgb.squeeze(0)

        obs['rgb'] = rgb
        return obs


def visual_rl_env(task: str, num_envs: int, image_size: int = 16, render_size: int = 128,
                   domain_randomization: bool = True) -> ManiSkillVectorEnv:
    """给视觉 RL 用的批量环境：原生环境 + 展平 + 降采样 + 颜色抖动 + 向量化。"""
    raw = _make_maniskill(task, num_envs=num_envs, obs_mode="rgb+segmentation",
                          sensor_size=render_size, render_mode="all",
                          domain_randomization=domain_randomization)
    env = FlattenRGBDObservationWrapper(raw, rgb=True, depth=False, state=True)
    if render_size != image_size:
        env = DownsampleObsWrapper(env, target_size=image_size)
    env = ColorJitterWrapper(env)
    return ManiSkillVectorEnv(env, num_envs, ignore_terminations=True, record_metrics=True)


def state_rl_env(task: str, num_envs: int, domain_randomization: bool = True) -> ManiSkillVectorEnv:
    """只给关节状态向量的批量环境（跳过渲染管线），供状态策略与调试用。"""
    raw = _make_maniskill(task, num_envs=num_envs, obs_mode="state", sensor_size=128,
                          render_mode="all", domain_randomization=domain_randomization)
    return ManiSkillVectorEnv(raw, num_envs, ignore_terminations=True, record_metrics=True)
