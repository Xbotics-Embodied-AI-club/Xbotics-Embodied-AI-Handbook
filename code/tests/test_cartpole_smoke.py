"""CartPole 值学习两级（表格 Q-learning / DQN）的冒烟测试。

只验证核心组件在小规模下能跑通、不报错，不追求训出可用策略（那是
`rl/3_offpolicy/3_1_cartpole_value_rl/train_v*.py` 直接运行的事）。
跑法：`python -m pytest tests/test_cartpole_smoke.py -v`。
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import gymnasium as gym
import lightning as L
import numpy as np
import torch

MODULE_DIR = Path(__file__).resolve().parents[1] / "rl" / "3_offpolicy" / "3_1_cartpole_value_rl"


def _load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, MODULE_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


qlearning = _load_module("cartpole_train_v1_qlearning", "train_v1_qlearning.py")
dqn = _load_module("cartpole_train_v2_dqn", "train_v2_dqn.py")


def test_discretize_returns_correct_bucket_tuple():
    obs = np.array([0.1, -0.2, 0.05, 0.3], dtype=np.float32)
    bucket = qlearning.discretize(obs)
    assert len(bucket) == 4
    assert all(0 <= index < qlearning.NUM_BINS for index in bucket)


def test_tabular_q_has_no_trainable_parameters():
    """第一级的教学点：Q 表放在 nn.Module 里，但一个可训练参数都没有。"""
    model = qlearning.TabularQ(n_actions=2)
    assert list(model.parameters()) == []
    assert model.table.shape == (qlearning.NUM_BINS,) * 4 + (2,)


def test_tabular_qlearning_training_step_runs(tmp_path):
    env = gym.make("CartPole-v1")
    n_actions = env.action_space.n

    stats = qlearning.CollectorState(env, seed=0)
    model = qlearning.TabularQLearning(
        n_actions=n_actions, alpha=0.2, gamma=0.99, stats=stats,
        qtable_path=tmp_path / "qtable.npy", result_path=tmp_path / "r.json", print_every=1000,
    )
    data = qlearning.CartPoleTabularData(
        env, model.model, stats, steps_per_epoch=4,
        epsilon_start=1.0, epsilon_end=1.0, epsilon_decay_steps=100,
    )
    trainer = L.Trainer(
        accelerator="cpu", devices=1, max_epochs=1, logger=False, enable_checkpointing=False,
        enable_progress_bar=False, enable_model_summary=False,
    )
    trainer.fit(model, data)

    # 表格更新确实发生了：至少有一格被改动过，且全表有限。
    assert torch.isfinite(model.model.table).all()
    assert (model.model.table != 0).any()
    env.close()


def test_qnetwork_forward_shape():
    net = dqn.QNetwork(state_dim=4, hidden_dim=8, n_actions=2)
    obs_batch = torch.randn(5, 4)
    q_values = net(obs_batch)
    assert q_values.shape == (5, 2)


def test_dqn_training_step_runs(tmp_path):
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    buffer = dqn.ReplayBuffer(capacity=200, state_dim=state_dim)
    stats = dqn.CollectorState(env, seed=0)
    dqn.warmup_buffer(env, buffer, stats, warmup_steps=50)

    model = dqn.DQN(
        state_dim=state_dim, n_actions=n_actions, hidden_dim=8, learning_rate=1.0e-3, gamma=0.99,
        target_sync_every=10, checkpoint_path=tmp_path / "dqn.pt", save_interval=1000, max_epochs=1,
        stats=stats, result_path=tmp_path / "r.json",
    )
    data = dqn.CartPoleDQNData(
        env, model, buffer, stats, steps_per_epoch=2, batches_per_epoch=1, batch_size=8,
        epsilon_start=1.0, epsilon_end=1.0, epsilon_decay_steps=100,
    )
    trainer = L.Trainer(
        accelerator="cpu", devices=1, max_epochs=1, logger=False, enable_checkpointing=False,
        enable_progress_bar=False, enable_model_summary=False,
    )
    trainer.fit(model, data)
    env.close()
