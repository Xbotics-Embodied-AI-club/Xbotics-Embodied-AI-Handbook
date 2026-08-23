"""3_2_so101_offpolicy 各阶梯网络/更新的冒烟测试：形状对不对、一步更新报不报错。

覆盖 v3 DDPG、v4 TD3、v5 SAC：确定性 `DeterministicActor`/`QCritic` 前向 shape，
以及在真实 `so101_sim.state_rl_env(...)` 上跑一个 minibatch 的
`training_step`（小 num_envs，loss 有限）。v4 额外核验双 Q 的两个 critic 都能前向、
`critic_loss`/`actor_loss` 有限（含延迟更新那一步 `batch_idx=0` 触发 actor 更新的
分支）。v5 额外核验随机策略 `SquashedGaussianActor.get_action` 返回
`(action, log_prob, det_action)` 三元组 shape，以及一步 SAC `training_step`
（三个优化器：actor/critic/log_alpha）不报错、loss 有限。三个文件各自都有一份
`RunningNorm`（在线观测归一化，治 actor tanh 饱和冻结）：额外核验它的
`update`/`normalize` 数值正确，以及三个 `LightningModule` 都带 `obs_norm` 且
`sample_action`/`eval_action`/`training_step` 在有归一化的情况下照常跑通。
v6 额外核验：`C51TwinQ` 的 logits/expected_q shape（分布式 Critic，不是
标量），视觉 `Actor.get_action` 三元组 shape；v6 的 `Projection` 自带 LayerNorm，
不需要 `RunningNorm`。这两项只测独立的网络前向，不依赖仿真环境——`CNNEncoder`/
`ReplayBuffer` 仍按单相机 3 通道写的，在 `platform/so101_sim` 现有的 KIT 双相机
（6 通道）场景上跑不动，这是留给 RL 模块重新整训时一并解决的架构缺口，见同目录
README「待重新整训」一节。
需要 CUDA（ManiSkill GPU 后端）。
运行：`uv run python -m pytest tests/test_so101_offpolicy_smoke.py -v`
（so101_sim 已 editable 安装，无需 PYTHONPATH）。
"""

import os
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mani_skill")

if not torch.cuda.is_available():
    pytest.skip("so101_sim 训练环境需要 CUDA（ManiSkill GPU 后端）", allow_module_level=True)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "..", "rl", "3_offpolicy", "3_2_so101_offpolicy"))
from train_v3_ddpg import DeterministicActor, QCritic, DDPG, RunningNorm as RunningNormV3  # noqa: E402
from train_v4_td3 import TD3, RunningNorm as RunningNormV4  # noqa: E402
from train_v5_sac import SquashedGaussianActor, SAC, RunningNorm as RunningNormV5  # noqa: E402
from train_v6_squint import CNNEncoder, Actor as VisualActor, C51TwinQ, VisualSAC  # noqa: E402


@pytest.mark.parametrize("RunningNorm", [RunningNormV3, RunningNormV4, RunningNormV5])
def test_running_norm_update_and_normalize(RunningNorm):
    """三份 RunningNorm 各自独立定义但逻辑该一致：喂一批已知均值/方差的数据，
    `update` 后 `mean`/`var` 该逼近真值，`normalize` 该把数据搓成零均值单位方差。"""
    device = "cuda"
    dim = 5
    norm = RunningNorm(dim).to(device)

    # 值域故意拉到 ~200，复现 bug 现场：不归一化直接喂 tanh 会饱和
    raw_mean, raw_std = 100.0, 50.0
    x = raw_mean + raw_std * torch.randn(4096, dim, device=device)
    norm.update(x)

    assert torch.allclose(norm.mean, torch.full((dim,), raw_mean, device=device), atol=2.0)
    assert torch.allclose(norm.var.sqrt(), torch.full((dim,), raw_std, device=device), atol=2.0)

    y = norm.normalize(x)
    assert y.mean().abs().item() < 0.1
    assert abs(y.std().item() - 1.0) < 0.1
    # tanh 饱和的直接验证：原始值几乎全饱和，归一化后基本不饱和
    assert (torch.tanh(x).abs() > 0.999).float().mean().item() > 0.9
    assert (torch.tanh(y).abs() > 0.999).float().mean().item() < 0.1

    # 增量更新：分两批喂等价于一次性喂全部数据（并行 Welford 更新正确性）
    norm2 = RunningNorm(dim).to(device)
    norm2.update(x[:2048]); norm2.update(x[2048:])
    assert torch.allclose(norm.mean, norm2.mean, atol=1e-3)
    assert torch.allclose(norm.var, norm2.var, atol=1e-3)


def test_deterministic_actor_and_qcritic_shapes():
    device = "cuda"
    state_dim, action_dim, batch = 20, 6, 8
    low = torch.full((action_dim,), -1.0, device=device)
    high = torch.full((action_dim,), 1.0, device=device)

    actor = DeterministicActor(state_dim, action_dim, low, high).to(device)
    critic = QCritic(state_dim, action_dim).to(device)

    state = torch.randn(batch, state_dim, device=device)
    action = actor(state)
    assert action.shape == (batch, action_dim)
    assert torch.all(action >= low - 1e-4) and torch.all(action <= high + 1e-4)

    q = critic(state, action)
    assert q.shape == (batch,)
    assert torch.isfinite(q).all()


def test_ddpg_training_step_on_real_env():
    import so101_sim

    device = "cuda"
    num_envs = 4
    env = so101_sim.state_rl_env("SO101PickPlaceCube40-v1", num_envs=num_envs)
    try:
        state, _ = env.reset()
        action_dim = env.single_action_space.shape[-1]
        low = torch.as_tensor(env.single_action_space.low, device=device)
        high = torch.as_tensor(env.single_action_space.high, device=device)
        model = DDPG(state.shape[-1], action_dim, low, high).to(device)
        assert hasattr(model, "obs_norm")

        model.obs_norm.update(state)  # 采集时更新归一化统计，与 SO101DDPGData._collect 同一接线
        action = model.sample_action(state)
        assert action.shape == (num_envs, action_dim)

        next_state, reward, _, _, info = env.step(action)
        batch = (state, action, reward, next_state)

        # 直接搭一个最小 Trainer 跑一步，确认 manual optimization 全链路
        # （评论家更新 → 演员更新 → 目标网络软更新）不报错、loss 有限。
        import lightning as L

        class _RecordLoss(L.Callback):
            def __init__(self):
                self.metrics = None

            def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
                self.metrics = dict(trainer.callback_metrics)

        recorder = _RecordLoss()
        trainer = L.Trainer(accelerator="gpu", devices=1, max_epochs=1, limit_train_batches=1,
                            enable_checkpointing=False, logger=False, enable_model_summary=False,
                            enable_progress_bar=False, callbacks=[recorder])

        class _OneBatch(torch.utils.data.IterableDataset):
            def __iter__(self):
                yield batch

        loader = torch.utils.data.DataLoader(_OneBatch(), batch_size=None)
        trainer.fit(model, train_dataloaders=loader)

        assert recorder.metrics is not None
        assert torch.isfinite(recorder.metrics["critic_loss"])
        assert torch.isfinite(recorder.metrics["actor_loss"])
    finally:
        env.close()


def test_td3_twin_critic_forward_shapes():
    """v4 相对 DDPG 新增双 Q：两个独立 QCritic 都要能前向、输出有限。"""
    device = "cuda"
    state_dim, action_dim, batch = 20, 6, 8
    low = torch.full((action_dim,), -1.0, device=device)
    high = torch.full((action_dim,), 1.0, device=device)

    model = TD3(state_dim, action_dim, low, high).to(device)
    state = torch.randn(batch, state_dim, device=device)
    action = model.actor(state)

    q1 = model.critic1(state, action)
    q2 = model.critic2(state, action)
    assert q1.shape == (batch,) and q2.shape == (batch,)
    assert torch.isfinite(q1).all() and torch.isfinite(q2).all()
    # 两个 critic 权重独立初始化，不应该恰好相等
    assert not torch.allclose(q1, q2)


def test_td3_training_step_on_real_env():
    """一步 TD3 training_step（双 Q 取 min → 延迟更新触发 → 目标策略平滑）不报错、loss 有限。"""
    import so101_sim

    device = "cuda"
    num_envs = 4
    env = so101_sim.state_rl_env("SO101PickPlaceCube40-v1", num_envs=num_envs)
    try:
        state, _ = env.reset()
        action_dim = env.single_action_space.shape[-1]
        low = torch.as_tensor(env.single_action_space.low, device=device)
        high = torch.as_tensor(env.single_action_space.high, device=device)
        model = TD3(state.shape[-1], action_dim, low, high).to(device)
        assert hasattr(model, "obs_norm")

        model.obs_norm.update(state)  # 采集时更新归一化统计，与 SO101DDPGData._collect 同一接线
        action = model.sample_action(state)
        assert action.shape == (num_envs, action_dim)

        next_state, reward, _, _, info = env.step(action)
        batch = (state, action, reward, next_state)

        import lightning as L

        class _RecordLoss(L.Callback):
            def __init__(self):
                self.metrics = None

            def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
                self.metrics = dict(trainer.callback_metrics)

        recorder = _RecordLoss()
        # batch_idx 从 0 开始，policy_frequency=2 时 0 % 2 == 0 会触发 actor 更新，
        # 顺带核验延迟更新的两个分支中"更新"这一支不报错。
        trainer = L.Trainer(accelerator="gpu", devices=1, max_epochs=1, limit_train_batches=1,
                            enable_checkpointing=False, logger=False, enable_model_summary=False,
                            enable_progress_bar=False, callbacks=[recorder])

        class _OneBatch(torch.utils.data.IterableDataset):
            def __iter__(self):
                yield batch

        loader = torch.utils.data.DataLoader(_OneBatch(), batch_size=None)
        trainer.fit(model, train_dataloaders=loader)

        assert recorder.metrics is not None
        assert torch.isfinite(recorder.metrics["critic_loss"])
        assert torch.isfinite(recorder.metrics["actor_loss"])
    finally:
        env.close()


def test_squashed_gaussian_actor_get_action_shapes():
    """v5 相对 v3/v4 最大变化：随机策略。`get_action` 要吐三元组，且不是每次都同一个动作。"""
    device = "cuda"
    state_dim, action_dim, batch = 20, 6, 8
    low = torch.full((action_dim,), -1.0, device=device)
    high = torch.full((action_dim,), 1.0, device=device)

    actor = SquashedGaussianActor(state_dim, action_dim, low, high).to(device)
    state = torch.randn(batch, state_dim, device=device)

    action, log_prob, det_action = actor.get_action(state)
    assert action.shape == (batch, action_dim)
    assert log_prob.shape == (batch, 1)
    assert det_action.shape == (batch, action_dim)
    assert torch.all(action >= low - 1e-4) and torch.all(action <= high + 1e-4)
    assert torch.all(det_action >= low - 1e-4) and torch.all(det_action <= high + 1e-4)
    assert torch.isfinite(log_prob).all()

    # 随机策略：同一状态两次采样不应该恰好相等（确定性动作 det_action 才应该稳定）
    action2, _, det_action2 = actor.get_action(state)
    assert not torch.allclose(action, action2)
    assert torch.allclose(det_action, det_action2)


def test_sac_training_step_on_real_env():
    """一步 SAC training_step（双 Q 熵折进目标 → 自动温度 α → 熵正则的演员更新 → 软更新）不报错、loss 有限。"""
    import so101_sim

    device = "cuda"
    num_envs = 4
    env = so101_sim.state_rl_env("SO101PickPlaceCube40-v1", num_envs=num_envs)
    try:
        state, _ = env.reset()
        action_dim = env.single_action_space.shape[-1]
        low = torch.as_tensor(env.single_action_space.low, device=device)
        high = torch.as_tensor(env.single_action_space.high, device=device)
        model = SAC(state.shape[-1], action_dim, low, high).to(device)
        assert hasattr(model, "obs_norm")

        model.obs_norm.update(state)  # 采集时更新归一化统计，与 SO101SACData._collect 同一接线
        action = model.sample_action(state)
        assert action.shape == (num_envs, action_dim)

        next_state, reward, _, _, info = env.step(action)
        batch = (state, action, reward, next_state)

        import lightning as L

        class _RecordLoss(L.Callback):
            def __init__(self):
                self.metrics = None

            def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
                self.metrics = dict(trainer.callback_metrics)

        recorder = _RecordLoss()
        trainer = L.Trainer(accelerator="gpu", devices=1, max_epochs=1, limit_train_batches=1,
                            enable_checkpointing=False, logger=False, enable_model_summary=False,
                            enable_progress_bar=False, callbacks=[recorder])

        class _OneBatch(torch.utils.data.IterableDataset):
            def __iter__(self):
                yield batch

        loader = torch.utils.data.DataLoader(_OneBatch(), batch_size=None)
        trainer.fit(model, train_dataloaders=loader)

        assert recorder.metrics is not None
        assert torch.isfinite(recorder.metrics["critic_loss"])
        assert torch.isfinite(recorder.metrics["actor_loss"])
        assert torch.isfinite(recorder.metrics["alpha"])
    finally:
        env.close()


def test_c51_twinq_forward_shapes():
    """v6 相对 v3/v4/v5 的标量 `QCritic` 最大变化：Critic 输出一条分布，不是一个数。"""
    device = "cuda"
    feat_dim, state_dim, action_dim, batch, num_atoms = 1024, 6, 6, 8, 101

    critic = C51TwinQ(feat_dim, state_dim, action_dim, num_atoms=num_atoms).to(device)
    feat = torch.randn(batch, feat_dim, device=device)
    state = torch.randn(batch, state_dim, device=device)
    action = torch.randn(batch, action_dim, device=device)

    logits = critic.logits(feat, state, action)
    assert logits.shape == (2, batch, num_atoms)
    assert torch.isfinite(logits).all()

    q = critic.expected_q(feat, state, action)
    assert q.shape == (2, batch)
    assert torch.isfinite(q).all()

    reward = torch.randn(batch, device=device)
    target = critic.categorical_target(feat, state, action, reward, discount=0.9)
    assert target.shape == (2, batch, num_atoms)
    # 是概率分布：每个 (Q, batch) 切片求和该是 1
    assert torch.allclose(target.sum(-1), torch.ones(2, batch, device=device), atol=1e-4)


def test_visual_actor_get_action_shapes():
    """v6 的视觉 `Actor.get_action`：和 v5 `SquashedGaussianActor` 同一套契约，
    只是先吃 CNN 特征 + 关节状态两路输入（经 `Projection` 拼接）。"""
    device = "cuda"
    feat_dim, state_dim, action_dim, batch = 1024, 6, 6, 8
    low = torch.full((action_dim,), -1.0, device=device)
    high = torch.full((action_dim,), 1.0, device=device)

    actor = VisualActor(feat_dim, state_dim, action_dim, low, high).to(device)
    feat = torch.randn(batch, feat_dim, device=device)
    state = torch.randn(batch, state_dim, device=device)

    action, log_prob, det_action = actor.get_action(feat, state)
    assert action.shape == (batch, action_dim)
    assert log_prob.shape == (batch, 1)
    assert det_action.shape == (batch, action_dim)
    assert torch.all(action >= low - 1e-4) and torch.all(action <= high + 1e-4)
    assert torch.isfinite(log_prob).all()


@pytest.mark.xfail(
    reason="CNNEncoder/ReplayBuffer 按单相机 3 通道写的；platform/so101_sim 现有的 KIT "
    "分发场景是双相机 6 通道输出，这是留给 RL 模块重新整训时一并解决的架构缺口，不是本次"
    "接口收敛能机械修的（见 3_2_so101_offpolicy/README.md「待重新整训」一节）。",
    strict=True,
)
def test_visual_sac_training_step_on_real_env():
    """一步 v6 `VisualSAC.training_step`（C51 交叉熵评论家 → 自动温度 α → 均值 Q 演员 →
    软更新）在真实 16px 视觉环境上不报错、loss 有限。v6 不需要 `RunningNorm`：
    `Projection` 内的 LayerNorm 已经把视觉特征/状态都归一化了。"""
    import so101_sim

    device = "cuda"
    num_envs = 4
    env = so101_sim.visual_rl_env("SO101PickPlaceCube40-v1", num_envs=num_envs, image_size=16)
    try:
        obs, _ = env.reset()
        rgb, state = obs["rgb"], obs["state"]
        action_dim = env.single_action_space.shape[-1]
        low = torch.as_tensor(env.single_action_space.low, device=device)
        high = torch.as_tensor(env.single_action_space.high, device=device)
        model = VisualSAC(state.shape[-1], action_dim, low, high).to(device)
        assert not hasattr(model, "obs_norm")  # 明确没有外挂观测归一化

        action = model.sample_action(rgb, state)
        assert action.shape == (num_envs, action_dim)

        next_obs, reward, _, _, info = env.step(action)
        batch = (rgb, state, action, reward, next_obs["rgb"], next_obs["state"])

        import lightning as L

        class _RecordLoss(L.Callback):
            def __init__(self):
                self.metrics = None

            def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
                self.metrics = dict(trainer.callback_metrics)

        recorder = _RecordLoss()
        trainer = L.Trainer(accelerator="gpu", devices=1, max_epochs=1, limit_train_batches=1,
                            enable_checkpointing=False, logger=False, enable_model_summary=False,
                            enable_progress_bar=False, callbacks=[recorder])

        class _OneBatch(torch.utils.data.IterableDataset):
            def __iter__(self):
                yield batch

        loader = torch.utils.data.DataLoader(_OneBatch(), batch_size=None)
        trainer.fit(model, train_dataloaders=loader)

        assert recorder.metrics is not None
        assert torch.isfinite(recorder.metrics["critic_loss"])
        assert torch.isfinite(recorder.metrics["alpha"])
    finally:
        env.close()
