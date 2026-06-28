# 第 16 讲：机械臂抓取 RL 后训练与 Recovery Policy

> **代码**：[`code/lecture16/`](../../code/lecture16/)

## 16.1 教学目标

理解后训练与 Recovery Policy：在 BC/ACT/DP/VLA 或规则策略基础上，针对失败场景 RL 优化。

## 16.2 核心知识点

后训练思路：初始策略 → rollout 失败 → 分类 → recovery reward → 仿真优化 → 真机评测。

Recovery：抓偏重接近、未抓住重试、滑落重检测、放置失败重试。

风险：reward hacking、抖动、仿真过度优化。

## 16.3–16.5 Demo

- **有硬件**：SO101 规则/IL 策略失败样本 → recovery 逻辑验证
- **无硬件**：ManiSkill/Isaac 加载初始策略 → recovery RL → 成功率对比

## 16.6 作业

失败样本表、recovery reward 设计、前后成功率对比、sim-to-real 风险分析。

## 统一复盘（第四部分）

见 [`part4-reinforcement-learning/00-part-overview.md`](00-part-overview.md) 及 [`references/links.md`](../../references/links.md#lecture-16)。
