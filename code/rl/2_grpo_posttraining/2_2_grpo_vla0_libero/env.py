"""LIBERO 同状态组环境：GRPO 的采样前提是「同一初始状态下比较一组 rollout 的相对好坏」。

这里把 LIBERO 包装成向量化环境：一组 n 个子环境被强制设成同一个初始状态，
策略在上面采样 n 条轨迹，任务成败（0/1）就是每条轨迹的稀疏奖励。
组内奖励有成有败，才有相对优势信号——这是训练脚本里前沿状态池存在的原因。
"""

import os

# LIBERO 走 MuJoCo 离屏渲染，必须用 EGL 后端（无显示器的训练机上 glfw 会崩）。
os.environ["MUJOCO_GL"] = "egl"
os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"

import numpy as np
import torch

from lerobot.envs.configs import LiberoEnv
from lerobot.envs.factory import make_env
from lerobot.scripts.lerobot_eval import add_envs_task, preprocess_observation
from lerobot.utils.constants import ACTION

from model import sample_chunk


def build_group_env(task_id, n):
    """建一个 n 路向量化 LIBERO 环境（libero_object 套件的第 task_id 个任务）。"""
    cfg = LiberoEnv(task="libero_object", task_ids=[task_id])
    d = make_env(cfg, n_envs=n, use_async_envs=False)
    return next(iter(next(iter(d.values())).values())), cfg


def set_same_init_state(env, idx):
    """把向量环境里所有子环境的初始状态锁成同一个 idx——同状态组是 GRPO 的前提。"""
    n = 0
    for sub in env.envs:
        s = getattr(sub, "unwrapped", sub)
        if hasattr(s, "init_state_id"):
            n_init = len(s._init_states) if getattr(s, "_init_states", None) is not None else 1
            s.init_state_id = idx % max(n_init, 1)
            s._reset_stride = 0
            n += 1
    return n


def slice_record(rec, g):
    """从一批生成记录里切出第 g 个环境自己的那一条。"""
    return {k: (v[g:g + 1] if torch.is_tensor(v) else v) for k, v in rec.items()}


def rollout_group(policy, env, pre, env_pre, post, env_post, seed, group_size,
                  init_idx, temperature, max_steps_cap, collect_records=True):
    """在同一初始状态上采一组 rollout。

    返回 (records, rewards)：records[g] 是第 g 条轨迹每次生成动作块时的输入/输出记录
    （训练时要用它复算 logprob），rewards[g] 是该条轨迹的 0/1 成败。
    """
    m = policy.model
    policy.reset()
    set_same_init_state(env, init_idx)
    obs, _ = env.reset(seed=seed)
    max_steps = env.call("_max_episode_steps")[0]
    if max_steps_cap > 0:
        max_steps = min(max_steps, max_steps_cap)
    records = [[] for _ in range(group_size)]
    success = np.zeros(group_size, dtype=bool)
    done = np.zeros(group_size, dtype=bool)
    queue = []
    step = 0
    while step < max_steps and not done.all():
        if not queue:
            # 动作块用完了：把观测走一遍预处理管线，再让 VLA-0 生成下一个动作块。
            o = preprocess_observation(obs)
            o = add_envs_task(env, o)
            o = env_pre(o)
            o = pre(o)
            chunk, rec = sample_chunk(m, o, temperature)
            if collect_records:
                for g in range(group_size):
                    if not done[g]:
                        records[g].append(slice_record(rec, g))
            queue = list(chunk.transpose(0, 1))
        a = queue.pop(0)
        at = post(a)
        a_np = env_post({ACTION: at})[ACTION].to("cpu").numpy()
        obs, reward, term, trunc, info = env.step(a_np)
        reward = np.ravel(reward)
        term = np.ravel(term)
        trunc = np.ravel(trunc)
        success = success | (reward > 0)
        done = done | term.astype(bool) | trunc.astype(bool)
        step += 1
    return records, success.astype(float).tolist()
