"""表格 Q-learning：CartPole 值学习的第一级，故意不用神经网络、不用 Lightning。

CartPole 的观测是 4 维连续量（小车位置、小车速度、杆的角度、杆的角速度），而表格方法
要求状态是离散索引，所以第一步永远是**分桶**：把每一维切成有限段，四维分桶组合起来
就是 Q 表的“状态”维度。分桶越细、Q 表越准，但表的大小是每维桶数的**指数级乘积**——
只有 4 维观测就已经要精心挑选桶数和裁剪范围才能学好，换成图像或更高维的机器人状态就
直接组合爆炸、根本存不下。这正是下一级 `train_v2_dqn.py` 用神经网络代替查表
（拿泛化换精确、用函数近似绕开维度诅咒）要解决的问题。

更新规则是最朴素的表格 Q-learning（TD(0)、off-policy）：
    Q[s, a] += ALPHA * (r + GAMMA * max_a' Q[s', a'] - Q[s, a])
行为策略是 ε-greedy，ε 随回合数线性衰减，训练前期多探索、后期多利用当前 Q 表。
"""
from __future__ import annotations

import os
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np

# 每一维切成的桶数：状态空间大小 = NUM_BINS ** 4，四维已经是表格方法的极限量级。
NUM_BINS = 8
# 每一维参与分桶的裁剪范围（超出范围的观测会被夹到边界桶）。位置/角度用物理终止边界，
# 速度/角速度理论上无界，用比理论范围更紧的经验区间——真正学到东西的样本大多落在这个
# 区间里，切得太宽会把桶都浪费在几乎不出现的极端速度上。
OBS_LOW = np.array([-2.4, -2.0, -0.21, -2.5])
OBS_HIGH = np.array([2.4, 2.0, 0.21, 2.5])
BIN_EDGES = [np.linspace(low, high, NUM_BINS - 1) for low, high in zip(OBS_LOW, OBS_HIGH)]


def discretize(obs):
    """把 4 维连续观测映射成离散桶索引 tuple，作为 Q 表的行索引。"""
    return tuple(int(np.digitize(value, edges)) for value, edges in zip(obs, BIN_EDGES))


ALPHA = 0.2  # 学习率
GAMMA = 0.99  # 折扣因子

EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY_EPISODES = 2500  # 到第几回合线性衰减到 EPSILON_END，之后保持不变

EPISODES = 4000
PRINT_EVERY = 200  # 每 N 回合打印一次「最近 N 回合回报的滑动均值」——两级统一的统计口径


def main():
    env = gym.make("CartPole-v1")
    n_actions = env.action_space.n
    Q = np.zeros((NUM_BINS,) * 4 + (n_actions,))

    recent_returns = deque(maxlen=PRINT_EVERY)
    for episode in range(1, EPISODES + 1):
        epsilon = max(
            EPSILON_END,
            EPSILON_START - (EPSILON_START - EPSILON_END) * episode / EPSILON_DECAY_EPISODES,
        )
        obs, _ = env.reset()
        state = discretize(obs)
        episode_return = 0.0
        done = False
        while not done:
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(Q[state]))

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = discretize(next_obs)

            # 真正的失败（杆倒/出界）不 bootstrap 下一状态；到时间上限的 truncated 只是
            # 环境计时器切断，不代表回报到此为止，仍要接上下一状态的估计。
            target = reward if terminated else reward + GAMMA * np.max(Q[next_state])
            Q[state][action] += ALPHA * (target - Q[state][action])

            state = next_state
            episode_return += reward

        recent_returns.append(episode_return)
        if episode % PRINT_EVERY == 0:
            print(
                f"episode {episode:5d} | epsilon {epsilon:.3f} | "
                f"return (avg over last {PRINT_EVERY}) {np.mean(recent_returns):6.1f}"
            )

    qtable_path = Path(os.environ["DATASETS_ROOT"]) / "models" / "trained" / "cartpole" / "qtable.npy"
    qtable_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(qtable_path, Q)
    print(f"Q 表已保存到 {qtable_path}")
    env.close()


if __name__ == "__main__":
    main()
