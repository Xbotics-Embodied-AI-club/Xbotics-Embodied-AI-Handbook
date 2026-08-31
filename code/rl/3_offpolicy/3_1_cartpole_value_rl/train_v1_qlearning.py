"""表格 Q-learning：CartPole 值学习的第一级，**没有神经网络**。

这一级刻意不引入任何网络层：Q 就是一张查得到、看得见的表。它和下一级
`train_v2_dqn.py` 的组织方式完全一样（Dataset → LightningDataModule →
nn.Module → LightningModule → trainer.fit），**版本之间的差异就是这一讲的知识点**——
唯一变的是"Q 从哪来"：这里是查表，下一级是网络算。

CartPole 的观测是 4 维连续量（小车位置、小车速度、杆的角度、杆的角速度），而表格方法
要求状态是离散索引，所以第一步永远是**分桶**：把每一维切成有限段，四维分桶组合起来
就是 Q 表的"状态"维度。分桶越细、Q 表越准，但表的大小是每维桶数的**指数级乘积**——
只有 4 维观测就已经要精心挑选桶数和裁剪范围才能学好，换成图像或更高维的机器人状态就
直接组合爆炸、根本存不下。这正是下一级用神经网络代替查表（拿泛化换精确、用函数近似
绕开维度诅咒）要解决的问题。

更新规则是最朴素的表格 Q-learning（TD(0)、off-policy）：
    Q[s, a] += ALPHA * (r + GAMMA * max_a' Q[s', a'] - Q[s, a])
行为策略是 ε-greedy，ε 随环境步数线性衰减，训练前期多探索、后期多利用当前 Q 表。

注意这里没有优化器、也没有反向传播：查表更新是直接改表里的一个数，不是梯度下降。
`configure_optimizers` 因此返回 None——"没有网络"这件事在代码里是看得见的。
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

# 每一维切成的桶数：状态空间大小 = NUM_BINS ** 4，四维已经是表格方法的极限量级。
NUM_BINS = 8
# 每一维参与分桶的裁剪范围（超出范围的观测会被夹到边界桶）。位置/角度用物理终止边界，
# 速度/角速度理论上无界，用比理论范围更紧的经验区间——真正学到东西的样本大多落在这个
# 区间里，切得太宽会把桶都浪费在几乎不出现的极端速度上。
OBS_LOW = np.array([-2.4, -2.0, -0.21, -2.5])
OBS_HIGH = np.array([2.4, 2.0, 0.21, 2.5])
BIN_EDGES = [np.linspace(low, high, NUM_BINS - 1) for low, high in zip(OBS_LOW, OBS_HIGH)]


def discretize(obs):
    """把 4 维连续观测映射成离散桶索引 tuple，作为 Q 表的行索引。

    表格方法只能用整数下标寻址，连续观测必须先落到桶里；桶边界一旦定死，
    落进同一个桶的状态在这一级里就是"同一个状态"，精度损失也从这里来。

    Args:
        obs: 长度 4 的连续观测（小车位置、小车速度、杆角度、杆角速度）。

    Returns:
        长度 4 的整数 tuple，每一维是该维落在哪个桶里。
    """
    return tuple(int(np.digitize(value, edges)) for value, edges in zip(obs, BIN_EDGES))


class TabularQ(nn.Module):
    """Q 表。是个 nn.Module，但**一层网络都没有**——整张表就是一个 buffer。

    放进 nn.Module 只是为了和 v2 的 `QNetwork` 保持同一个位置、同一种存取方式
    （`state_dict()` / `load_state_dict()` 都能直接用）；它没有参数、不参与求导。
    """

    def __init__(self, n_actions):
        super().__init__()
        # register_buffer：随模型保存/加载，但不是可训练参数。
        self.register_buffer("table", torch.zeros((NUM_BINS,) * 4 + (n_actions,)))
        self.n_actions = n_actions

    def forward(self, state_index):
        """查表：给一个状态的桶索引，返回该状态下每个动作的 Q 值。

        写成 `forward` 而不是普通方法，是为了让这一级和 v2 的 `QNetwork` 用同一种调用
        写法——外面那句 `self(discretize(obs))` 换到 v2 就是真的网络前向。

        Args:
            state_index: `discretize` 出来的桶索引 tuple。

        Returns:
            形状 (n_actions,) 的张量，该状态下每个动作的 Q 值。
        """
        return self.table[tuple(state_index)]

    def act(self, obs, epsilon):
        """ε-greedy：大概率按当前 Q 表挑最优动作，小概率随机试。

        Q-learning 是 off-policy 的：这里用来采数据的是带随机性的 ε-greedy，
        而更新目标里取的是 `max`（贪心策略）——采的和学的本来就不是同一个策略。

        Args:
            obs: 环境给的原始连续观测。
            epsilon: 这一步的探索概率，训练中随环境步数线性衰减。

        Returns:
            动作下标（0 或 1）。
        """
        if np.random.rand() < epsilon:
            return int(np.random.randint(self.n_actions))
        return int(torch.argmax(self(discretize(obs))))


class CollectorState:
    """跨 epoch 存活的采集状态：当前环境、当前回合回报、最近若干回合的回报。"""

    def __init__(self, env, seed):
        self.env = env
        self.obs, _ = env.reset(seed=seed)
        self.episode_return = 0.0
        self.recent_returns = deque(maxlen=50)
        # 每个回合结束时把回报追加进来，训练完整条曲线存成 json，用来画对照图。
        self.all_returns = []
        self.global_env_step = 0
        self.finished_episodes = 0
        # 回合结束时只置标志，真正的 reset 放到下一次取样前做——这样即使采集提前中断，
        # 环境也不会停在"已经结束却还要继续 step"的状态上。
        self.needs_reset = False


class CartPoleTabularDataset(IterableDataset):
    """在线采集：用当前 Q 表（ε-greedy）与环境交互，每走一步就产出一条转移用于更新。

    表格 Q-learning 是"即采即更新"的：没有回放池，一条经验用完就丢。这一点正是 v2 要改的
    ——网络扛不住这种前后高度相关的样本流。
    """

    def __init__(self, env, model, stats, steps_per_epoch,
                 epsilon_start, epsilon_end, epsilon_decay_steps):
        super().__init__()
        self.env = env
        self.model = model
        self.stats = stats
        self.steps_per_epoch = steps_per_epoch
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps

    def __iter__(self):
        stats = self.stats
        for _ in range(self.steps_per_epoch):
            if stats.needs_reset:
                stats.obs, _ = self.env.reset()
                stats.needs_reset = False
            stats.global_env_step += 1
            epsilon = max(
                self.epsilon_end,
                self.epsilon_start
                - (self.epsilon_start - self.epsilon_end)
                * stats.global_env_step / self.epsilon_decay_steps,
            )
            state = discretize(stats.obs)
            action = self.model.act(stats.obs, epsilon)
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            next_state = discretize(next_obs)
            stats.episode_return += reward

            # 真正的失败（杆倒/出界）不 bootstrap 下一状态；到时间上限的 truncated 只是
            # 环境计时器切断，不代表回报到此为止，仍要接上下一状态的估计。
            # 所以写进 done 的是 terminated，不是 terminated or truncated。
            yield (
                torch.tensor(state, dtype=torch.long),
                torch.tensor(action, dtype=torch.long),
                torch.tensor(reward, dtype=torch.float32),
                torch.tensor(next_state, dtype=torch.long),
                torch.tensor(float(terminated), dtype=torch.float32),
            )

            if terminated or truncated:
                stats.recent_returns.append(stats.episode_return)
                stats.all_returns.append(stats.episode_return)
                stats.finished_episodes += 1
                stats.episode_return = 0.0
                stats.needs_reset = True
            else:
                stats.obs = next_obs


class CartPoleTabularData(L.LightningDataModule):
    """持有环境与统计量，每个 epoch 现采一段新经验交给训练循环。

    强化学习没有事先存好的数据集，样本要靠当前策略去环境里换回来，所以这个
    DataModule 持有的是 `env` 和 `model` 本身，而不是一个文件路径。
    """

    def __init__(self, env, model, stats, steps_per_epoch,
                 epsilon_start, epsilon_end, epsilon_decay_steps):
        super().__init__()
        self.env = env
        self.model = model
        self.stats = stats
        self.steps_per_epoch = steps_per_epoch
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps

    def train_dataloader(self):
        """每个 epoch 重建一次数据集，让它拿到此刻最新的 Q 表去采样。

        Returns:
            逐条吐出转移的 DataLoader（`batch_size=None`，不做批处理）。
        """
        dataset = CartPoleTabularDataset(
            self.env, self.model, self.stats, self.steps_per_epoch,
            self.epsilon_start, self.epsilon_end, self.epsilon_decay_steps,
        )
        # batch_size=None：一条转移就是一次表格更新，不做批处理。
        return DataLoader(dataset, batch_size=None)


class TabularQLearning(L.LightningModule):
    """表格 Q-learning：直接改表里的一个数，没有优化器、没有反向传播。"""

    def __init__(self, n_actions, alpha, gamma, stats, qtable_path, result_path, print_every):
        super().__init__()
        self.model = TabularQ(n_actions)
        self.alpha = alpha
        self.gamma = gamma
        self.stats = stats
        self.qtable_path = qtable_path
        self.result_path = result_path
        self.print_every = print_every
        # 没有梯度可求，关掉 Lightning 的自动优化。
        self.automatic_optimization = False

    def configure_optimizers(self):
        """返回 None——这一级没有任何可训练参数，也就没有优化器。

        Returns:
            None。Lightning 允许这样，配合 `automatic_optimization = False` 使用。
        """
        # 查表更新不是梯度下降，这里确实没有优化器可配。
        return None

    def training_step(self, batch, batch_idx):
        """一条转移做一次表格 Q-learning 更新。

        和 v2 的对照点：这里"学"的全部动作就是把表里的一格往 TD 目标挪一点，
        既不前向也不反传；v2 把同一件事换成对一整批样本做回归。

        Args:
            batch: 一条转移 `(state, action, reward, next_state, done)`，
                其中 state / next_state 已是桶索引。
            batch_idx: Lightning 传入的批序号，这里用不到。

        Returns:
            None——没有 loss 要交给 Lightning 反传。
        """
        state, action, reward, next_state, done = batch

        cell = tuple(state.tolist()) + (int(action),)
        best_next = self.model.table[tuple(next_state.tolist())].max()
        target = reward + self.gamma * best_next * (1.0 - done)
        td_error = target - self.model.table[cell]

        # 这一行就是表格 Q-learning 的全部：把这一格往目标挪 ALPHA 那么多。
        self.model.table[cell] += self.alpha * td_error

        self.log("td_error", td_error.abs(), prog_bar=True, on_step=True, on_epoch=False)
        return None

    def on_train_epoch_end(self):
        """每隔 print_every 个 epoch 打印一次近期回报，训练时肉眼看得到进展。"""
        stats = self.stats
        if stats.recent_returns and (self.current_epoch + 1) % self.print_every == 0:
            print(
                f"env_step {stats.global_env_step:6d} | "
                f"episodes {stats.finished_episodes:5d} | "
                f"return (avg over last {len(stats.recent_returns)}) "
                f"{np.mean(stats.recent_returns):6.1f}"
            )

    def on_train_end(self):
        """训练结束时落盘：Q 表本身，以及整条回报曲线。

        回报曲线存成 json 而不是只打印，是为了让讲义里那张对照图可以从原始记录重画，
        报出来的数字也能被回代核对。
        """
        self.qtable_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(self.qtable_path, self.model.table.numpy())
        print(f"Q 表已保存到 {self.qtable_path}")

        self.result_path.parent.mkdir(parents=True, exist_ok=True)
        self.result_path.write_text(json.dumps({
            "algo": "tabular-qlearning",
            "env_steps": self.stats.global_env_step,
            "episodes": self.stats.finished_episodes,
            "returns": self.stats.all_returns,
        }))
        print(f"回报曲线已保存到 {self.result_path}")


def main():
    """跑完整条表格 Q-learning 训练：建环境、配超参、`trainer.fit`、落盘。"""
    # 固定随机种子：同一台机器上重跑能得到完全一样的曲线，书里报的数字才可复现。
    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = gym.make("CartPole-v1")
    env.action_space.seed(seed)
    n_actions = env.action_space.n

    alpha = 0.2  # 学习率：每次把这一格往 TD 目标挪多少
    gamma = 0.99  # 折扣因子
    epsilon_start = 1.0
    epsilon_end = 0.02
    epsilon_decay_steps = 60000  # ε 线性衰减到 EPSILON_END 所用的环境步数

    steps_per_epoch = 200  # 每个 epoch 采（并更新）这么多环境步
    total_env_steps = 200000
    max_epochs = total_env_steps // steps_per_epoch
    print_every = 100  # 每多少个 epoch 打印一次回报滑动均值

    qtable_path = Path(os.environ["DATASETS_ROOT"]) / "models" / "trained" / "cartpole" / "qtable.npy"
    result_path = Path(__file__).resolve().parents[1] / "result" / "cartpole-qlearning.json"

    stats = CollectorState(env, seed)
    model = TabularQLearning(n_actions, alpha, gamma, stats, qtable_path, result_path, print_every)
    data = CartPoleTabularData(env, model.model, stats, steps_per_epoch,
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
