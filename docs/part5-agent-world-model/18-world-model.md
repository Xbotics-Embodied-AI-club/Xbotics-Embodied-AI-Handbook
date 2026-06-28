# 第 18 讲：世界模型、数据飞轮与具身智能未来方向

> **代码**：[`code/lecture18/`](../../code/lecture18/)

## 18.1 教学目标

理解世界模型预测什么；与世界模型 vs VLA 的边界；数据飞轮如何驱动系统迭代。

## 18.2 核心知识点

1. 世界模型可预测：下一帧、下一状态、动作后果、失败风险、可达性
2. model-based RL、Dreamer 系列、生成式世界模型
3. 价值：低成本试错、规划、失败预判、数据增强、虚拟训练
4. G1 世界模型需预测：姿态、接触、平衡、行走、操作后果、跌倒风险
5. 数据飞轮：采集 → 训练 → 部署 → 评测 → 失败回流 → 再训练
6. 未来方向：通用基础模型、长程任务、Loco-Dexterous、人形、开放世界泛化

## 18.3 教学设计（最终闭环）

```
感知 → 动作 → 数据 → 模型 → 部署 → 评测 → 失败 → 再采集 → 再训练
```

## 18.4–18.5 综合 Demo

**任务 A — SO101/xLeRobot**：语言 pick-place 全链路 + episode + 失败样本

**任务 B — G1**：walk_to → turn_to → detect → check_state

**无硬件**：LangGraph mock + ManiSkill + LeRobot 记录 + G1 仿真 + 失败库

## 18.6–18.7 实验步骤与最终交付

可运行代码、演示视频、episode、成功率、失败分析、数据改进计划、系统架构图、答辩材料。

## 18.8–18.9 复盘与参考

DreamerV3、OpenVLA、OpenPI、LeRobot、Unitree Sim IsaacLab — 见 [`references/links.md`](../../references/links.md#lecture-18)。
