# 第 12 讲：视觉语言动作模型 VLA —— 理论基础

> **代码**：[`code/lecture12/`](../../code/lecture12/)

## 12.1 教学目标

区分 VLM / VLA / 世界模型；掌握 VLA 样本结构；理解跨任务、跨场景、跨本体泛化难点。

## 12.2 核心知识点

1. VLA 输入：image、instruction、proprioception、history
2. VLA 输出：连续 action、action token、chunk、trajectory
3. 代表模型：RT-1/2、OpenVLA、Octo、π0/OpenPI、GR00T
4. Generalist policy 与 robot foundation model
5. SO101 / xLeRobot / Imeta / G1 数据进入 VLA 的差异
6. 局限：动作尺度、实时性、安全、分布外失败

## 12.3–12.5 Demo

从 SO101 或 LeRobot episode 构造最小 VLA sample：

```python
sample = {
    "image": image,
    "state": robot_state,
    "instruction": "pick up the red block",
    "action": action,
    "timestamp": timestamp,
    "episode_id": episode_id,
}
```

## 12.6–12.8 实验、作业、复盘

样本字段说明、动作尺度表、跨本体混训分析。

## 12.9–12.10 配图与参考

VLM/VLA/世界模型对比、样本结构、混训难点 — OpenVLA、Octo、OpenPI 等见 [`references/links.md`](../../references/links.md#lecture-12)。
