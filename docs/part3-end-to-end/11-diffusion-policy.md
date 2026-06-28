# 第 11 讲：生成式动作策略 —— Diffusion Policy 的意义与部署

> **代码**：[`code/lecture11/`](../../code/lecture11/)

## 11.1 教学目标

理解 DP 将动作序列作为生成对象；条件扩散、receding horizon、prediction/execution horizon 与部署延迟。

## 11.2 核心区别

- **ACT**：直接预测 action chunk（序列回归）
- **DP**：从噪声动作序列条件去噪生成轨迹（分布建模）

## 11.3 核心知识点

条件输入、视觉编码、receding horizon（预测 N 步、执行 K 步）、inference steps、部署风险（延迟、重采样抖动、视角敏感）。

## 11.4–11.6 Demo

SO101 或仿真：加载/训练小型 DP → 设置 horizon → rollout → 对比 ACT。

## 11.7–11.9 作业、复盘、配图

动作序列可视化、ACT/DP 对比表、延迟统计、生成式 vs chunk 预测说明。

## 11.10 参考

Diffusion Policy、LeRobot — 见 [`references/links.md`](../../references/links.md#lecture-11)。
