# 第 14 讲：强化学习基础 —— 从 MDP 到机器人策略

> **代码**：[`code/lecture14/`](../../code/lecture14/)

## 14.1 教学目标

理解 MDP、state/action/reward/policy；PPO/SAC/TD3 区别；机器人 RL 为何主要在仿真训练。

## 14.2 核心知识点

RL 框架、MDP、V/Q/advantage、actor-critic；reward shaping 与 sparse/dense；样本效率、reward hacking、sim-to-real。

## 14.3 课堂案例：Reaching

- state：current_eef、target_eef、joint_state
- action：eef_delta 或 joint_delta
- reward：-distance_to_target

## 14.4 Demo

Stable-Baselines3 + Gymnasium/ManiSkill/MuJoCo/Isaac：定义环境 → PPO/SAC → 改 reward → rollout。

## 14.5 作业

环境定义、训练曲线、reward 设计对比、为何不能只看 reward 曲线。

## 参考

Stable-Baselines3、Isaac Lab、ManiSkill — 见 [`references/links.md`](../../references/links.md#lecture-14)。
