# 第 10 讲：模仿学习训练 —— BC、ACT 与 Action Chunk

> **代码**：[`code/lecture10/`](../../code/lecture10/)

## 10.1 教学目标

基于第 9 讲 episode 完成 BC/ACT 训练；理解 action chunk 与 rollout 选 checkpoint。

## 10.2 核心知识点

- BC：单步 observation-action 监督
- ACT：Transformer 预测 action chunk
- chunk_size 1/5/10 与抖动、延迟权衡
- 数据质量、train/val/test、best val loss vs best rollout

## 10.3–10.5 Demo

SO101 或仿真数据 → LeRobotDataset → BC + ACT（chunk 对比）→ rollout。

## 10.6–10.8 实验、作业、复盘

训练曲线、action 曲线、rollout 视频、checkpoint 选择说明、失败归因（数据/模型/动作空间/部署）。

## 10.9–10.10 配图与参考

BC vs ACT、chunk 曲线、loss vs rollout — LeRobot、ACT++、ALOHA 等见 [`references/links.md`](../../references/links.md#lecture-10)。
