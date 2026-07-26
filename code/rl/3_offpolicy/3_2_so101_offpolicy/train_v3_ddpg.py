"""教学版状态 DDPG 训练入口：SO101 连续控制阶梯的第一级。

state → 确定性动作，直接用确定性策略梯度（Deterministic Policy Gradient）训练：
Actor 不再像 REINFORCE/PPO 那样输出一个分布再采样，而是直接吐出一个动作；
Critic 只需要学一个标量 Q(s,a)，不用像上一包 3_1 的 DQN 那样对每个动作分别打分——
这正是从离散动作迈进连续动作的关键一步：DQN 靠对 Q 值取 argmax 选动作，动作一旦
连续就没法穷举了；DDPG 用"直接输出动作 + 沿着 Q 的梯度往上爬"绕开了这个问题。

复用 3_1 DQN 已经验证过的两根稳定支柱——经验回放 + 目标网络——只是这里目标网络
既要滞后化 critic，也要滞后化 actor：算 TD 目标时，下一状态的动作也得由一份滞后
更新的目标 actor 给出，而不是当前正在训练的 actor（避免"自己追自己"）。确定性策略
本身不探索，训练时靠给动作叠加一点高斯噪声来探索环境。

四件套：环境 `so101_sim.make_train_env(obs_mode="state")`（只给关节状态，跳过渲染，
比视觉观测快得多、也更容易看出"策略到底学没学会"）＋ 网络/更新 `DDPG`（LightningModule）
＋ 数据 `SO101DDPGData`（LightningDataModule，持有环境和回放池）＋ 本文件的 `trainer.fit`。

缺什么：整条链路只有一个 Q，容易被过高估计——噪声或离群样本一旦把某个动作的 Q 值
估高了，确定性策略梯度就会把 actor 一路推向这个虚高的动作，训练容易发散或者学出
一个自欺欺人的局部解；下一级 TD3 用双 Q 取 min（Clipped Double Q）来治这个问题。

**观测归一化是这一级能学起来的前提**：SO101 的 55 维 state 里混着关节角、速度、物体
位姿等尺度完全不同的量，某些维度原始值能到 ~200。这样的观测直接喂进 `DeterministicActor`
的末层 `Tanh`，输入一大，tanh 早早饱和到 ±1 附近、梯度趋于 0——actor 的权重根本更新不动，
不是算法学不动，是网络输入没做基本的预处理就先把自己饱和死了。跟 3_1 DQN 不需要这一步
不同：DQN 的 Q 网络输出不经过 tanh，对输入尺度没那么敏感；DDPG/TD3/SAC 的 actor 末层
都是 tanh，这一步不能省。修法是 `RunningNorm`：在线维护 state 每一维的 running mean/var
（采集时用原始 state 更新统计），真正喂进网络前先归一化成零均值单位方差，回放池里仍然存
原始 state（这样早期数据依然能参与统计更新，不因为归一化本身丢信息）。

诚实的教学结论（公平预算 500 iter、三级同预算、加观测归一化后的真实结果）：在 SO101
ReachCube 上，这版 DDPG 加了归一化后 actor **确实解冻了**——权重不再是一动不动，奖励也
不再是死平的一条直线；但把预算从 200 轮拉到 500 轮，"不稳"并没有随时间自愈：曲线全程在
正负之间反复起落，500 轮下来 mean_reward（靠近度）均值只有约 0.28，success_once 从头到
尾贴着 0，一次没跨过。这正是"确定性策略 + 一份容易高估的单 Q + 只靠固定高斯噪声探索"在
操作任务上的真实弱点：Q 一旦被高估，确定性策略梯度就把 actor 整个推向那个虚高的动作，
reward 应声跳水，直到 Q 估计慢慢被纠正回来，进入下一轮同样的震荡——不是学不动，是学得
不稳，而且始终探索不到稀疏成功判定需要的精确位置，给多少轮预算都一样。放进三级同预算的
阶梯里看：mean_reward（靠近度）是从这一级的 0.28 起步的最低点，success_once 也是三级里
唯一"从未离开过 0"的一档。缺什么：整条链路只有一个 Q，容易被过高估计——下一级 TD3 用双
Q 取 min（Clipped Double Q）先把"稳"找回来，看能不能借此摸到成功。
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch.utils.data import DataLoader, IterableDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import so101_sim  # noqa: E402  统一环境：lerobot 评测与 RL 训练共用同一份定义


class RunningNorm(nn.Module):
    """在线观测归一化：跟踪 state 每一维的 running mean/std，把原始观测（值域可到 ~200）
    归一化到零均值单位方差，避免大数值让 actor 的 tanh 饱和、梯度冻结。"""

    def __init__(self, dim):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(1e-4))

    @torch.no_grad()
    def update(self, x):
        bm, bv, bc = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        delta = bm - self.mean; tot = self.count + bc
        self.mean += delta * bc / tot
        M2 = self.var * self.count + bv * bc + delta**2 * self.count * bc / tot
        self.var = M2 / tot; self.count = tot

    def normalize(self, x):
        return (x - self.mean) / (self.var.sqrt() + 1e-8)


class DeterministicActor(nn.Module):
    """state → 确定性动作（DPG，没有分布也不采样）：MLP 到 [-1,1] 再线性缩放进 [low, high]。"""

    def __init__(self, state_dim, action_dim, action_low, action_high):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim), nn.Tanh(),
        )
        self.register_buffer("action_scale", (action_high - action_low) / 2.0)
        self.register_buffer("action_bias", (action_high + action_low) / 2.0)

    def forward(self, state):
        return self.net(state) * self.action_scale + self.action_bias


class QCritic(nn.Module):
    """(state, action) → 标量 Q：拼接后过 MLP，只输出一个数值（不是分布，也不是逐动作打分表）。"""

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1)).squeeze(-1)


class DDPG(L.LightningModule):
    """一个 minibatch 的 DDPG 更新：评论家(MSE 回归 Q) → 演员(确定性策略梯度) → 软更新目标网络。"""

    def __init__(self, state_dim, action_dim, action_low, action_high,
                 gamma=0.99, tau=0.005, lr=3e-4, exploration_noise=0.1):
        super().__init__()
        self.automatic_optimization = False
        self.gamma, self.tau, self.lr = gamma, tau, lr
        self.exploration_noise = exploration_noise
        self.action_low, self.action_high = action_low, action_high

        self.actor = DeterministicActor(state_dim, action_dim, action_low, action_high)
        self.critic = QCritic(state_dim, action_dim)
        self.actor_target = DeterministicActor(state_dim, action_dim, action_low, action_high)
        self.critic_target = QCritic(state_dim, action_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        # 观测归一化：buffer 里存的仍是原始 state，这里只在喂进网络前做归一化
        self.obs_norm = RunningNorm(state_dim)

    def configure_optimizers(self):
        return (torch.optim.Adam(self.actor.parameters(), lr=self.lr),
                torch.optim.Adam(self.critic.parameters(), lr=self.lr))

    @torch.no_grad()
    def sample_action(self, state):
        """确定性动作 + 高斯探索噪声，clamp 回合法区间（确定性策略本身不探索，全靠这份噪声）。"""
        action = self.actor(self.obs_norm.normalize(state))
        noise = torch.randn_like(action) * self.exploration_noise
        return (action + noise).clamp(self.action_low, self.action_high)

    @torch.no_grad()
    def eval_action(self, state):
        return self.actor(self.obs_norm.normalize(state))

    def training_step(self, batch, batch_idx):
        actor_opt, critic_opt = self.optimizers()
        state, action, reward, next_state = batch
        # buffer 里是原始 state，喂进网络前统一归一化（统计量只在采集时更新，这里只读）
        state, next_state = self.obs_norm.normalize(state), self.obs_norm.normalize(next_state)

        # —— 评论家（MSE 回归 Q）：目标 Q 用目标网络算，动作来自目标 actor ——（always bootstrap）
        with torch.no_grad():
            next_action = self.actor_target(next_state)
            target_q = reward + self.gamma * self.critic_target(next_state, next_action)
        critic_loss = F.mse_loss(self.critic(state, action), target_q)
        critic_opt.zero_grad(); self.manual_backward(critic_loss); critic_opt.step()

        # —— 演员（确定性策略梯度）：直接最大化评论家给"actor 当前会选的动作"打的分 ——
        actor_loss = -self.critic(state, self.actor(state)).mean()
        actor_opt.zero_grad(); self.manual_backward(actor_loss); actor_opt.step()

        # —— 目标网络软更新（actor / critic 各一遍）——
        with torch.no_grad():
            for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
                tp.mul_(1 - self.tau).add_(self.tau * p)
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.mul_(1 - self.tau).add_(self.tau * p)

        self.log_dict({"critic_loss": critic_loss.detach(), "actor_loss": actor_loss.detach()},
                      prog_bar=True, on_step=True, on_epoch=False)


class ReplayBuffer:
    """定容经验回放池，整块开在 GPU 上（state 版：无 rgb，只存关节状态向量）。"""

    def __init__(self, capacity, state_dim, action_dim, device):
        z = lambda *s: torch.zeros(*s, device=device)  # noqa: E731
        self.state = z(capacity, state_dim)
        self.next_state = z(capacity, state_dim)
        self.action = z(capacity, action_dim)
        self.reward = z(capacity)
        self.capacity, self.device = capacity, device
        self.pos, self.full = 0, False

    def __len__(self):
        return self.capacity if self.full else self.pos

    def add(self, state, action, reward, next_state):
        n = state.shape[0]
        idx = (torch.arange(n, device=self.device) + self.pos) % self.capacity
        self.state[idx] = state; self.next_state[idx] = next_state
        self.action[idx] = action; self.reward[idx] = reward.float()
        self.pos = (self.pos + n) % self.capacity
        self.full = self.full or self.pos < n

    def sample(self, batch_size):
        i = torch.randint(0, len(self), (batch_size,), device=self.device)
        return self.state[i], self.action[i], self.reward[i], self.next_state[i]


class SO101DDPGData(L.LightningDataModule):
    """持有环境和回放池；每轮先采样、再把 minibatch 交给 Trainer（state 版，无 rgb）。"""

    def __init__(self, env, model, buffer, steps_per_iter, updates_per_iter, batch_size, learning_starts):
        super().__init__()
        self.env, self.model, self.buffer = env, model, buffer
        self.steps_per_iter, self.updates_per_iter = steps_per_iter, updates_per_iter
        self.batch_size, self.learning_starts = batch_size, learning_starts
        self.last_success = 0.0
        self.last_reward = 0.0

    def _collect(self, use_policy):
        state = self.env.obs["state"]
        self.model.obs_norm.update(state)  # 用原始 state 更新归一化统计，每步一次
        if use_policy:
            action = self.model.sample_action(state)
        else:  # 预热：均匀随机动作把池子填起来
            low, high = self.env.single_action_space.low, self.env.single_action_space.high
            low = torch.as_tensor(low, device=self.env.device)
            high = torch.as_tensor(high, device=self.env.device)
            action = low + (high - low) * torch.rand(self.env.num_envs, self.env.action_dim, device=self.env.device)
        next_obs, reward, _, _, success = self.env.step(action)
        self.buffer.add(state, action, reward, next_obs["state"])
        return success.float().mean().item(), reward.float().mean().item()

    def train_dataloader(self):
        def gen():
            while len(self.buffer) < self.learning_starts:
                self._collect(use_policy=False)
            stats = [self._collect(use_policy=True) for _ in range(self.steps_per_iter)]
            succ, rew = zip(*stats)
            self.last_success = float(np.mean(succ))
            self.last_reward = float(np.mean(rew))
            for _ in range(self.updates_per_iter):
                yield self.buffer.sample(self.batch_size)

        class _DS(IterableDataset):
            def __iter__(self_inner):
                return gen()

        return DataLoader(_DS(), batch_size=None)


class SuccessLogger(L.Callback):
    """每轮打印采集成功率 + 平均奖励（靠近度，success 之外的连续信号）+ 定期存 ckpt。"""

    def __init__(self, ckpt_dir, save_interval, max_iterations):
        self.ckpt_dir, self.save_interval, self.max_iterations = ckpt_dir, save_interval, max_iterations

    def on_train_epoch_end(self, trainer, pl_module):
        it = trainer.current_epoch + 1
        succ = trainer.datamodule.last_success
        rew = trainer.datamodule.last_reward
        print(f"  iter {it}: success_once={succ:.2f}  mean_reward={rew:.3f}", flush=True)
        if it % self.save_interval == 0 or it == self.max_iterations:
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"actor": pl_module.actor.state_dict(),
                       "obs_norm": pl_module.obs_norm.state_dict()}, self.ckpt_dir / "ddpg.pt")


def run_training(task, num_envs, max_iterations, updates_per_iter, batch_size,
                 buffer_capacity, learning_starts, device, seed=1):
    torch.manual_seed(seed)
    env = so101_sim.make_train_env(task, num_envs=num_envs, obs_mode="state", device=device)
    env.reset()
    low = torch.as_tensor(env.single_action_space.low, device=device)
    high = torch.as_tensor(env.single_action_space.high, device=device)
    model = DDPG(env.state_dim, env.action_dim, low, high).to(device)
    buffer = ReplayBuffer(buffer_capacity, env.state_dim, env.action_dim, device)
    data = SO101DDPGData(env, model, buffer, steps_per_iter=1, updates_per_iter=updates_per_iter,
                        batch_size=batch_size, learning_starts=learning_starts)

    ckpt_dir = Path(os.environ["DATASETS_ROOT"]) / "models" / "trained" / "so101_sim_offpolicy" / task
    trainer = L.Trainer(
        accelerator="gpu", devices=1, max_epochs=max_iterations,
        reload_dataloaders_every_n_epochs=1, enable_checkpointing=False, logger=False,
        enable_model_summary=False, enable_progress_bar=False, log_every_n_steps=10,
        callbacks=[SuccessLogger(ckpt_dir, save_interval=25, max_iterations=max_iterations)],
    )
    trainer.fit(model, datamodule=data)
    env.close()
    return ckpt_dir / "ddpg.pt"


# 改这里选任务与训练时长，然后 `python train_v3_ddpg.py`。
TASK = "SO101ReachCube-v1"

if __name__ == "__main__":
    # 与 v4 TD3 / v5 SAC 同一套共享预算，三级只差算法、可公平对照：
    # 每步 256 次更新（UTD），批 512，回放池 50 万；SAC 要到 iter~480 才稳定跨过"解决"，
    # 三级统一给够 500 iter 预算才公平——预算给短了，压根看不出 SAC 真正的优势在哪。
    run_training(
        task=TASK, num_envs=1024, max_iterations=500, updates_per_iter=256, batch_size=512,
        buffer_capacity=500_000, learning_starts=5_000, device="cuda",
    )
