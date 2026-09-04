"""DQN：用神经网络取代查表，CartPole 值学习的第二级。

相对 `train_v1_qlearning.py` 的表格 Q，这一级换掉的是**状态的表示方式**：不再手工分桶、
查一张离散表，而是用一个小 MLP `QNetwork` 直接把连续观测映射成每个动作的 Q 值——网络能
在相近状态之间泛化，不用穷举组合，天然绕开了表格方法的维度爆炸。

但网络不能像表格那样"即采即更新"：训练样本前后高度相关（同一条轨迹上的相邻步），
且更新目标自己也在变（bootstrap 用的是同一个正在训练的网络），直接套用表格 Q-learning
的更新方式会很不稳定。所以这一级新增两根稳定支柱：
  1. **经验回放（Replay Buffer）**——把交互经验存起来，训练时随机小批量采样，打破样本间
     的时序相关性，也让每条经验能被多次复用；
  2. **目标网络（Target Network）**——算 TD 目标时用一份滞后更新的网络参数，避免"自己追自己"
     导致的目标漂移。

这一级依然只能处理离散动作：`act` 靠对 Q 值取 argmax 选动作，动作空间稍微一变成连续
（比如机械臂的关节力矩），argmax 就没法做了——这是
DDPG 引入确定性策略网络要解决的问题。
"""
from __future__ import annotations

import json
import os
from collections import deque
from pathlib import Path

import gymnasium as gym
import lightning as L
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset


class QNetwork(nn.Module):
    """状态 -> 每个动作的 Q 值，state_dim(4) -> [128, 128] -> n_actions(2) 的普通 MLP。"""

    def __init__(self, state_dim, hidden_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, obs):
        """一次前向：一批观测进，每个动作的 Q 值出。

        和 v1 查表最大的区别在这里——相近的观测会得到相近的输出，一次更新能惠及
        一整片邻域，不必像表格那样每一格都自己撞够次数。

        Args:
            obs: 形状 (batch, state_dim) 的观测张量。

        Returns:
            形状 (batch, n_actions) 的 Q 值张量。
        """
        return self.net(obs)


class ReplayBuffer:
    """环形缓冲区，存 (s, a, r, s', done)；容量满后覆盖最旧的经验。"""

    def __init__(self, capacity, state_dim):
        self.capacity = capacity
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0

    def push(self, state, action, reward, next_state, done):
        """把一条转移写进环形缓冲区；池满之后覆盖最旧的那条。

        Args:
            state: 当前观测。
            action: 执行的动作下标。
            reward: 即时奖励。
            next_state: 下一观测。
            done: 是否真正失败（terminated），超时截断不算——超时时后面还有回报，
                不该把 bootstrap 掐掉。
        """
        idx = self.position % self.capacity
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = done
        self.position += 1
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """均匀随机抽一个 minibatch。

        随机抽样本身就是稳定器：同一条轨迹里相邻的转移高度相关，按时间顺序喂给网络
        等于反复用同一批相关样本冲刷梯度；打散之后一个 batch 里混着不同回合、
        不同阶段的经验，梯度估计干净得多。

        Args:
            batch_size: 这一批要抽多少条转移。

        Returns:
            `(states, actions, rewards, next_states, dones)` 五个 numpy 数组。
        """
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (
            self.states[idxs], self.actions[idxs], self.rewards[idxs],
            self.next_states[idxs], self.dones[idxs],
        )

    def __len__(self):
        return self.size


PRINT_EVERY = 20  # 每 N 回合打印一次「最近 N 回合回报的滑动均值」——与 v1 同一套统计口径


class CollectorState:
    """在线采集要跨越多个 Lightning epoch 持续存在的状态：当前观测、回合计数、ε 衰减进度。"""

    def __init__(self, env, seed):
        self.obs, _ = env.reset(seed=seed)
        self.episode_return = 0.0
        self.episode_count = 0
        self.global_env_step = 0
        self.recent_returns = deque(maxlen=PRINT_EVERY)
        # 每个回合结束时把回报追加进来，训练完整条曲线存成 json，用来画对照图。
        self.all_returns = []


def warmup_buffer(env, buffer, stats, warmup_steps):
    """训练开始前先用纯随机策略把回放池灌到 warmup_steps 条。

    没有这一步，最早那几个 batch 会从几乎空的池子里反复抽同几条样本，网络先被这几条
    带偏，后面很难拉回来。

    Args:
        env: 已 reset 过的 CartPole 环境。
        buffer: 待灌注的回放池。
        stats: 采集状态（当前观测、回合回报、计数器），会被就地更新。
        warmup_steps: 灌多少步随机经验。
    """
    for _ in range(warmup_steps):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buffer.push(stats.obs, action, reward, next_obs, float(terminated))
        stats.obs = next_obs
        stats.episode_return += reward
        if done:
            stats.episode_count += 1
            stats.recent_returns.append(stats.episode_return)
            stats.all_returns.append(stats.episode_return)
            stats.obs, _ = env.reset()
            stats.episode_return = 0.0


class CartPoleReplayDataset(IterableDataset):
    """在线采集：每次迭代先用当前策略（ε-greedy）采若干步进回放池，再 yield 若干条训练 minibatch。"""

    def __init__(self, env, model, buffer, stats, steps_per_epoch, batches_per_epoch, batch_size,
                 epsilon_start, epsilon_end, epsilon_decay_steps):
        super().__init__()
        self.env = env
        self.model = model
        self.buffer = buffer
        self.stats = stats
        self.steps_per_epoch = steps_per_epoch
        self.batches_per_epoch = batches_per_epoch
        self.batch_size = batch_size
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps

    def collect(self):
        """用当前策略在环境里采 steps_per_epoch 步，全部写进回放池。

        采样和训练在这一级是分开的两段：先采一段，再从池子里抽很多批来学——
        "采一次、学很多次"正是 off-policy 能把每条经验榨干的地方。
        """
        stats = self.stats
        for _ in range(self.steps_per_epoch):
            stats.global_env_step += 1
            epsilon = max(
                self.epsilon_end,
                self.epsilon_start - (self.epsilon_start - self.epsilon_end)
                * stats.global_env_step / self.epsilon_decay_steps,
            )
            action = self.model.act(stats.obs, epsilon)
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            self.buffer.push(stats.obs, action, reward, next_obs, float(terminated))
            stats.obs = next_obs
            stats.episode_return += reward
            if done:
                stats.episode_count += 1
                stats.recent_returns.append(stats.episode_return)
                stats.all_returns.append(stats.episode_return)
                if stats.episode_count % PRINT_EVERY == 0:
                    print(
                        f"episode {stats.episode_count:5d} | epsilon {epsilon:.3f} | "
                        f"return (avg over last {PRINT_EVERY}) {np.mean(stats.recent_returns):6.1f}"
                    )
                stats.obs, _ = self.env.reset()
                stats.episode_return = 0.0

    def __iter__(self):
        # 采样放在 DataModule 里做，这里只负责吐训练用的小批量。
        for _ in range(self.batches_per_epoch):
            states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
            yield (
                torch.as_tensor(states, dtype=torch.float32),
                torch.as_tensor(actions, dtype=torch.long),
                torch.as_tensor(rewards, dtype=torch.float32),
                torch.as_tensor(next_states, dtype=torch.float32),
                torch.as_tensor(dones, dtype=torch.float32),
            )


class CartPoleDQNData(L.LightningDataModule):
    """持有环境、回放池与统计量。

    对照 v1 的 `CartPoleTabularData`：多出来的成员只有一个 `buffer`——
    on-policy 到 off-policy 的差别，在数据这一层就是"多了个池子"。
    """

    def __init__(self, env, model, buffer, stats, steps_per_epoch, batches_per_epoch, batch_size,
                 epsilon_start, epsilon_end, epsilon_decay_steps):
        super().__init__()
        self.env = env
        self.model = model
        self.buffer = buffer
        self.stats = stats
        self.steps_per_epoch = steps_per_epoch
        self.batches_per_epoch = batches_per_epoch
        self.batch_size = batch_size
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps

    def train_dataloader(self):
        """先采一段新经验进池，再交给 DataLoader 从池里抽小批量。

        Returns:
            每次迭代吐一个 minibatch 的 DataLoader（`batch_size=None`，数据集自己成批）。
        """
        dataset = CartPoleReplayDataset(
            self.env, self.model, self.buffer, self.stats,
            self.steps_per_epoch, self.batches_per_epoch, self.batch_size,
            self.epsilon_start, self.epsilon_end, self.epsilon_decay_steps,
        )
        # 每个 epoch 先用当前策略采一段新经验进池，再交给 DataLoader 抽小批量训练。
        dataset.collect()
        return DataLoader(dataset, batch_size=None)


class DQN(L.LightningModule):
    """DQN：Q(s,a) 回归 r + γ·max_a' Q_target(s',a')，`automatic_optimization=False` 手动优化。"""

    def __init__(self, state_dim, n_actions, hidden_dim, learning_rate, gamma,
                 target_sync_every, checkpoint_path, save_interval, max_epochs, stats, result_path):
        super().__init__()
        self.online = QNetwork(state_dim, hidden_dim, n_actions)
        self.target = QNetwork(state_dim, hidden_dim, n_actions)
        self.target.load_state_dict(self.online.state_dict())
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.target_sync_every = target_sync_every
        self.checkpoint_path = checkpoint_path
        self.save_interval = save_interval
        self.stats = stats
        self.result_path = result_path
        self.max_epochs = max_epochs
        self.automatic_optimization = False
        self.grad_steps = 0

    def act(self, obs, epsilon):
        """ε-greedy：以 ε 概率随机探索，否则对在线网络的 Q 值取 argmax。

        这里的 `argmax` 正是 DQN 迈不进连续控制的原因——动作是有限几个才枚举得过来。

        Args:
            obs: 环境给的原始连续观测。
            epsilon: 这一步的探索概率。

        Returns:
            动作下标（0 或 1）。
        """
        if np.random.rand() < epsilon:
            return int(np.random.randint(self.n_actions))
        with torch.no_grad():
            q_values = self.online(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
        return int(torch.argmax(q_values, dim=1).item())

    def configure_optimizers(self):
        """只优化在线网络。

        目标网络是它的延迟拷贝，靠定期整体复制参数更新，不参与梯度下降——
        真让它跟着一起训，回归目标就又开始跟着网络晃了。

        Returns:
            在线网络参数上的 Adam 优化器。
        """
        return torch.optim.Adam(self.online.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        """一个 minibatch 的 DQN 更新：算 TD 目标、回归、按节奏同步目标网络。

        Args:
            batch: 从回放池抽出的 `(states, actions, rewards, next_states, dones)`。
            batch_idx: Lightning 传入的批序号，这里用不到。

        Returns:
            这一批的均方误差损失。
        """
        states, actions, rewards, next_states, dones = batch

        q_values = self.online(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            # 采集时 dones 写的是 terminated（不含 truncated），
            # 所以真正失败才不 bootstrap，这里直接乘 (1 - dones)。
            next_q = self.target(next_states).max(dim=1).values
            td_target = rewards + self.gamma * next_q * (1.0 - dones)
        loss = nn.functional.mse_loss(q_values, td_target)

        optimizer = self.optimizers()
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()

        self.grad_steps += 1
        if self.grad_steps % self.target_sync_every == 0:
            self.target.load_state_dict(self.online.state_dict())

        self.log("loss", loss, prog_bar=True, on_step=True, on_epoch=False)

        epoch = self.current_epoch + 1
        if epoch % self.save_interval == 0 or epoch == self.max_epochs:
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.online.state_dict(), self.checkpoint_path)
        return loss

    def on_train_end(self):
        """训练结束时把整条回报曲线落盘，供画对照图和回代核对数字用。"""
        self.result_path.parent.mkdir(parents=True, exist_ok=True)
        self.result_path.write_text(json.dumps({
            "algo": "dqn",
            "env_steps": self.stats.global_env_step,
            "episodes": self.stats.episode_count,
            "returns": self.stats.all_returns,
        }))
        print(f"回报曲线已保存到 {self.result_path}")


def main():
    """跑完整条 DQN 训练：建环境与回放池、灌 warmup、`trainer.fit`、落盘。"""
    # 固定随机种子：同一台机器上重跑能得到完全一样的曲线，书里报的数字才可复现。
    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = gym.make("CartPole-v1")
    env.action_space.seed(seed)
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    hidden_dim = 128
    buffer_capacity = 50000
    batch_size = 64
    warmup_steps = 1000
    learning_rate = 1.0e-3
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_end = 0.02
    epsilon_decay_steps = 15000
    target_sync_every = 200  # 每多少次梯度更新同步一次目标网络

    steps_per_epoch = 20  # 每个 epoch 先采这么多环境步
    batches_per_epoch = 20  # 再用这么多个小批量做梯度更新（采样:更新 = 1:1）
    # 5 万步时策略已经练到位（末段平均回报 400 上下）；再往后是收益递减，训练就停在这里。
    total_env_steps = 50000
    max_epochs = total_env_steps // steps_per_epoch
    save_interval = 500

    buffer = ReplayBuffer(buffer_capacity, state_dim)
    stats = CollectorState(env, seed)
    warmup_buffer(env, buffer, stats, warmup_steps)

    checkpoint_path = Path(os.environ["DATASETS_ROOT"]) / "models" / "trained" / "cartpole" / "dqn.pt"
    result_path = Path(__file__).resolve().parents[1] / "result" / "cartpole-dqn.json"
    model = DQN(state_dim, n_actions, hidden_dim, learning_rate, gamma,
                target_sync_every, checkpoint_path, save_interval, max_epochs, stats, result_path)
    data = CartPoleDQNData(env, model, buffer, stats, steps_per_epoch, batches_per_epoch, batch_size,
                           epsilon_start, epsilon_end, epsilon_decay_steps)

    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=max_epochs,
        reload_dataloaders_every_n_epochs=1,
        enable_checkpointing=False,
        logger=False,
        enable_model_summary=False,
        enable_progress_bar=True,
        log_every_n_steps=1,
    )
    trainer.fit(model, data)
    env.close()


if __name__ == "__main__":
    main()
