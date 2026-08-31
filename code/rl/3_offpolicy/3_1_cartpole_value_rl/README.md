# CartPole 值学习入门：Q-learning → DQN

本模块在**同一个任务**（CartPole 保持杆子不倒）上，用两级由简到繁的值学习算法各训一个策略，展示"值学习"如何从手工分桶的表格进化到能自我泛化的神经网络。两级脚本结构刻意保持一致，**版本之间的差异就是这一讲的知识点**。

## 两个版本

| 脚本 | 算法 | 关键机制 | 缺什么 |
|---|---|---|---|
| `train_v1_qlearning.py` | 表格 Q-learning | 状态手工分桶 + 查表 TD 更新 | 无神经网络、无梯度；维度一高桶数就组合爆炸 |
| `train_v2_dqn.py` | DQN | 网络 Q + **经验回放** + **目标网络** | 仍靠 argmax 选离散动作，连续动作无解 |

`train_v1_qlearning.py` **故意没有网络**，用最原始的查表方式让"表格方法为什么会失效"这件事看得见摸得着。它照样套着完整的 Lightning 四件套（`TabularQ` / `CartPoleTabularDataset` / `CartPoleTabularData` / `TabularQLearning` + `trainer.fit`），只是 `configure_optimizers` 返回 `None`、关掉了自动优化——**两级共用同一副骨架，换掉的只是骨架里的零件**，diff 才读得出知识点。`train_v2_dqn.py` 换成网络后立刻遇到样本相关、目标漂移两个新问题，经验回放和目标网络就是分别对症的两个稳定支柱。

DQN 依然只能对离散动作取 argmax；下一包 `rl/3_offpolicy/3_2_so101_offpolicy/` 里的 DDPG 引入确定性策略网络，专门解决连续动作空间的问题。

## 怎么跑

```bash
cd code
python rl/3_offpolicy/3_1_cartpole_value_rl/train_v1_qlearning.py
python rl/3_offpolicy/3_1_cartpole_value_rl/train_v2_dqn.py
```

两级都打印回合回报（每 N 回合一次滑动均值），直接看打印结果对照，不需要额外的 rollout/对照脚本。CartPole 纯 CPU 即可训练，无需 GPU。

- `train_v1_qlearning.py` 训完把 Q 表存到 `DATASETS_ROOT/models/trained/cartpole/qtable.npy`。
- `train_v2_dqn.py` 训完把在线网络权重存到 `DATASETS_ROOT/models/trained/cartpole/dqn.pt`。

## 课程口径

这是 off-policy 值学习的地基：先在 CartPole 这个离散动作、状态维度低的任务上把"表格 → 网络 → 回放 → 目标网络"这条主线讲清楚，再在 `3_2_so101_offpolicy/` 里换到 SO101 连续控制任务，把值学习升级成 DDPG → TD3 → SAC 的确定性/随机策略阶梯。
