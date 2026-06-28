# 第 8 讲：端到端策略学习导论 —— 从 Pipeline 到 Policy

> **所属部分**：第三部分 · 端到端机器人操作  
> **代码**：[`code/lecture08/`](../../code/lecture08/)

## 8.1 教学目标

理解 policy = observation → action；state/image/language-conditioned policy 区别；rollout 验证与安全边界。

## 8.2 核心知识点

1. Pipeline → Policy 演进
2. observation / action 类型与 multimodal
3. covariate shift、compounding error、normalization、clip
4. 模块化 vs 端到端

## 8.3 教学设计

```
手写 controller → 模块化 pipeline → state policy → image policy → language policy → VLA
```

## 8.4–8.5 Demo

- **有硬件**：SO101 预训练 state-based policy 闭环推理（clip/scale/smoothing）
- **无硬件**：ManiSkill/MuJoCo reaching policy rollout

## 8.6–8.8 实验、作业、复盘

对比 controller vs policy；说明为何 policy 必须闭环执行、loss 下降 ≠ rollout 成功。

## 8.9 参考开源项目

robomimic、LeRobot、ACT、Diffusion Policy、OpenVLA — 见 [`references/links.md`](../../references/links.md#lecture-08)。
