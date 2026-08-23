"""用训好的 v6 策略跑轨迹，录成 ManiSkill 的 h5。

数据生产线第一步：把同目录上一级 `train_v6_squint.py` 训好的策略在仿真里 rollout 出
一批**成功轨迹**，用 ManiSkill 的 `RecordEpisode` 录下来。关键是录的时候套在**原始
128px 环境**外面（h5 里存的是全分辨率图 + 完整 `env_states`），而策略自己看的是降
采样到 16px 的视图（训练时也是这个分辨率）——两者靠包装器叠出来。

`env_states` 是后面换外观的命门：有了它，replay 就能把同一套动作重渲成任意相机/外观。
"""

import math
import os
import sys
from pathlib import Path

import numpy as np  # noqa: F401  C 扩展 ABI 顺序：numpy 在 torch 前
import torch
import gymnasium as gym

import so101_sim  # 注册任务
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from so101_sim.wrappers import DownsampleObsWrapper

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_v6_squint import CNNEncoder, Actor  # noqa: E402  数据生产线加载教学 v6 训好的策略

NUM_ENVS = 16
IMAGE_SIZE = 16      # 策略输入分辨率（与训练一致）
RENDER_SIZE = 128    # h5 里存的全分辨率
EPISODE_STEPS = 400  # 与三个分发场景注册的 max_episode_steps 一致：抓起来再放进料盒要这么多步


def rollout(task: str, ckpt_path: Path, n_episodes: int, out_dir: Path, device: str = "cuda") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    n_batches = math.ceil(n_episodes / NUM_ENVS)

    # 原始环境（默认外观、关域随机化，让每集稳定）；RecordEpisode 录它的 128px + env_states。
    raw = gym.make(task, num_envs=NUM_ENVS, obs_mode="rgb+segmentation", render_mode="all",
                   sim_backend="gpu", domain_randomization=False, reconfiguration_freq=1,
                   sensor_configs=dict(width=RENDER_SIZE, height=RENDER_SIZE))
    rec = RecordEpisode(raw, output_dir=str(out_dir), save_trajectory=True, save_video=False,
                        trajectory_name="rollout", max_steps_per_video=EPISODE_STEPS)
    # 策略看到的视图：flatten RGB+state 再降采样到 16px。
    pol = FlattenRGBDObservationWrapper(rec, rgb=True, depth=False, state=True)
    pol = DownsampleObsWrapper(pol, target_size=IMAGE_SIZE)
    vec = ManiSkillVectorEnv(pol, NUM_ENVS, ignore_terminations=True, record_metrics=True)

    obs, _ = vec.reset()
    n_state = math.prod(vec.unwrapped.single_observation_space["state"].shape)
    n_act = math.prod(vec.unwrapped.single_action_space.shape)
    low = torch.as_tensor(vec.unwrapped.single_action_space.low, device=device)
    high = torch.as_tensor(vec.unwrapped.single_action_space.high, device=device)
    enc = CNNEncoder().to(device)
    act = Actor(enc.repr_dim, n_state, n_act, low, high).to(device)
    ck = torch.load(ckpt_path, map_location=device)
    enc.load_state_dict(ck["encoder"]); act.load_state_dict(ck["actor"])
    enc.eval(); act.eval()

    once_total, ep_total = 0.0, 0
    for b in range(n_batches):
        obs, _ = vec.reset()
        ever = torch.zeros(NUM_ENVS, dtype=torch.bool, device=device)
        for _ in range(EPISODE_STEPS):
            with torch.no_grad():
                _, _, a = act.get_action(enc(obs["rgb"]), obs["state"])  # 确定性均值动作，采集更稳
            obs, _, _, _, info = vec.step(a)
            ever |= info["success"].bool()
        once = ever.float().mean().item()
        once_total += once * NUM_ENVS; ep_total += NUM_ENVS
        print(f"  batch {b + 1}/{n_batches}: success_once={once:.2f}")

    vec.close()  # flush h5
    h5_path = out_dir / "rollout.h5"
    print(f"rollout done: {ep_total} 集，mean success_once={once_total / max(ep_total, 1):.3f} -> {h5_path}")
    return h5_path


# 独立跑：改这里的任务与集数。
TASK = "SO101PickPlaceCube40-v1"
N_EPISODES = 64

if __name__ == "__main__":
    datasets_root = Path(os.environ["DATASETS_ROOT"])
    ckpt = datasets_root / "models" / "trained" / "so101_sim_sac" / TASK / "sac_ckpt.pt"  # train_v6_squint.py 的产物
    work = datasets_root / "so101_sim" / "_gen" / TASK
    rollout(TASK, ckpt, N_EPISODES, work)
