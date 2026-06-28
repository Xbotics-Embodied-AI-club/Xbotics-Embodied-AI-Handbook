# 第 21 讲：具身智能前沿进展 —— 综合闭环与课程答辩

> **所属部分**：第六部分 · Agent、世界模型与进展  
> **Part 负责人**：雨浩  
> **代码**：[`code/lecture21/`](../../code/lecture21/)

## 21.1 本讲目标

梳理具身智能 **前沿方向**（通用基础模型、长程任务、Loco-Dexterous、人形、开放世界），并完成 **全课程综合任务** 与答辩交付。

本讲结束后，学生应能够回答：

1. 2024–2026 具身智能有哪些清晰可跟的工程主线（不堆论文名单）？
2. 如何把 Agent、VLN、VLA、世界模型、数据飞轮放进同一张系统图？
3. 一门课学完后，下一步应选哪条深扎路线？

本讲结束后，学生应能够完成：

- **阶段项目四**：语言指令驱动的移动 + 操作（或仿真等价）综合 Demo
- 提交答辩材料：架构图、成功率、失败样本、下一轮数据采集计划

## 21.2 核心知识点

1. **通用机器人基础模型**：multi-embodiment、generalist policy、Open X-Embodiment、π0 / GR00T 等 **工程含义**（非排行榜）
2. **长程任务**：Agent 规划 + VLN 移动 + VLA/规则操作 + check / recovery
3. **Loco-Dexterous Manipulation**：G1 / 人形 + 全身协调；与 Part 4 locomotion 衔接
4. **开放世界泛化**：场景、物体、指令、本体的 OOD 与 data scaling
5. **数据飞轮落地**：什么样的失败样本最有价值、如何定下一轮采集优先级
6. **评测与产品化**：success rate、SPL、latency、安全、可维护 skill 库
7. **个人路线选择**：操作 / 导航 VLN / 运控 RL / 世界模型 / 数据工程

## 21.3 课堂任务 / 引入案例

**综合任务（仿真或真机）**：

> 指令：「找到红色方块，把它放到蓝色区域。」

**系统分解**：

```
Agent 规划 → VLN/导航接近桌面 → 感知检测 → VLA/规则 grab → place → check_success → 记录 episode → 失败回流
```

## 21.4 方法框架

**具身系统全栈闭环（课程总框架）**：

```
感知 → 决策(Agent/VLA/VLN/RL) → 控制 → 数据 → 训练 → 部署 → 评测 → 失败 → 再采集
```

**前沿跟踪三问**（避免追热点）：

1. 解决什么 **任务**？  
2. 需要什么 **数据与硬件**？  
3. 能否 **复现最小 Demo**？

## 21.5 有硬件版 Demo

**Demo A — xLeRobot 综合**：语言指令 → 导航 → 桌面抓取 → 放置 → 日志

**Demo B — G1 概览**：walk_to → turn_to → detect → 准备操作（安全演示级）

## 21.6 无硬件仿真版 Demo

LangGraph Agent + Habitat/mock VLN + ManiSkill pick-place + LeRobot episode 记录 + 失败库

## 21.7 实验步骤

1. 输入语言指令，生成 Agent 计划
2. 依次调用 navigate / detect / grasp / place / check skills
3. 保存 episode 与 success
4. 统计成功率，分类失败
5. 写「下一轮数据采集计划」
6. 整理答辩 PPT / 报告（架构图 + 指标 + 2 个 failure case）

## 21.8 作业交付（课程最终答辩）

1. 可运行综合 Demo 代码
2. 任务演示视频
3. episode 数据样例
4. 成功率 / SPL（若含导航）统计
5. 失败案例分析表
6. 综合系统架构图
7. 个人后续学习路线（1 页）

## 21.9 常见失败与复盘

- 综合任务只拼模块，没有统一日志格式
- 失败未分类，无法指导采集
- 追新模型无复现
- Agent 计划不可执行仍强行跑通

## 21.10 参考开源项目

OpenPI、OpenVLA、Habitat、LeRobot、Xbotics-Embodied-Guide 前沿章节 — 见 [`references/links.md`](../../references/links.md#lecture-21)。
