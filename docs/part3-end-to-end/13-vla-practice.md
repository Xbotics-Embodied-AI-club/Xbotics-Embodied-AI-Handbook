# 第 13 讲：VLA 实操 —— 微调、动作尺度与真机部署调试

> **代码**：[`code/lecture13/`](../../code/lecture13/)

## 13.1 教学目标

掌握 VLA fine-tuning pipeline；**action normalization 是部署核心**；混训与相机视角、频率、安全边界。

## 13.2 核心知识点

Pipeline：采集 → 清洗 → instruction → action 统计 → normalize → 微调 → checkpoint → inference → rollout → 失败回流。

部署：推理/控制频率、插值、延迟、workspace limit、emergency stop。

混训风险：动作空间不一致、HIL/recovery 数据带偏主任务。

## 13.3–13.5 Demo

**动作尺度实验**：数据 A（大幅度）、B（小幅度）、C（混训）→ histogram → SO101 rollout 对比。

## 13.6–13.8 实验、作业、复盘

normalization 代码、三组实验表、部署问题分析报告、混训筛选策略。

## 13.9 配图

微调到部署流程、normalization 流程、尺度 histogram、loss vs rollout、视角变化失败、安全边界。

## 13.10 参考

OpenVLA、OpenVLA-OFT、Octo、OpenPI、LIBERO — 见 [`references/links.md`](../../references/links.md#lecture-13)。
