# 第 18 讲：视觉语言导航 VLN —— 理论基础

> **所属部分**：第五部分 · 视觉语言导航 VLN  
> **Part 负责人**：新梦  
> **代码**：[`code/lecture18/`](../../code/lecture18/)

## 18.1 本讲目标

本讲帮助学生理解 **VLN（Vision-Language Navigation）** 在具身智能中的位置：它与 VLA、传统导航、Robot Agent 分别解决什么问题，以及一条语言指令如何变成移动机器人的可执行轨迹。

本讲结束后，学生应能够回答：

1. VLN 和 VLA、SLAM 导航有什么区别？
2. 为什么 VLN 需要「看 + 听懂 + 走过去」，而不是只输出一次动作？
3. 仿真 VLN 任务通常包含哪些 observation 和 success 判定？

本讲结束后，学生应能够完成：

- 画出 VLN 在具身系统中的模块图（语言 → 感知 → 规划/策略 → 底盘控制 → 反馈）
- 说清 Habitat / VLN-CE 等基准任务的基本设置

## 18.2 核心知识点

1. **VLN 定义**：根据自然语言指令，在未知或部分已知环境中移动到目标位置
2. **与相邻方向的区别**
   - SLAM / 导航：侧重地图、定位、路径规划，语言可选
   - VLA：侧重操作动作（机械臂/夹爪），常是桌面或固定基座
   - VLN：侧重 **移动 + 语言理解 + 视觉序列决策**
   - Robot Agent：VLN 可作为 Agent 的一个 **navigate_to** skill
3. **典型任务形式**：Room-to-Room（R2R）、VLN-CE（连续环境）、ObjectNav、Instruction following with stopping
4. **Observation**：RGB / RGB-D 序列、位姿、语言指令、可选 top-down map、历史动作
5. **Action 空间**：离散转向/前进、连续 velocity、waypoint、子目标点
6. **评测指标**：Success Rate、SPL（Success weighted by Path Length）、NDTW、碰撞率
7. **方法谱系**：seq2seq + CNN/RNN → Transformer → 预训练 VLM + 策略头 → 分层规划（高层子目标 + 低层控制）
8. **xLeRobot / G1 场景**：xLeRobot 语言导航到桌面再操作；G1 行走 + 转向 + 面向目标，为后续操作做准备
9. **常见失败**：指令歧义、过拟合训练场景、sim-to-real 视觉差异、停止条件错误、碰撞与卡住

## 18.3 课堂任务 / 引入案例

**任务**：在 Matterport3D 仿真环境中，根据指令 *「离开当前房间，进入厨房，在冰箱前停下」*，理解 agent 每一步需要什么信息。

**为什么选这个任务**：VLN 的长程性、语言歧义性和「何时停止」问题，在一句话指令里都能体现。

## 18.4 方法框架

**VLN 系统五步框架**：

```
语言指令 → 指令 grounding（对齐到视觉/地图）
→ 子目标或 waypoint 生成 → 低层运动控制 → 停止/成功检测 → 失败重规划
```

**VLN 失败分析四类**：指令理解 · 视觉 grounding · 路径/控制 · 停止判定

## 18.5 有硬件版 Demo

**Demo 名称**：xLeRobot 语言导航到目标区域（概览级，为第 19 讲实操铺垫）

**流程**：解析指令 → 检测/定位目标区域 → `navigate_to` → 到达后 `check_state` → 为 arm 操作预留接口

**硬件**：xLeRobot 移动底盘 + 相机；可选 G1 `walk_to` + `turn_to` 高层演示

## 18.6 无硬件仿真版 Demo

**Demo 名称**：Habitat VLN-CE / R2R 最小 episode 走读

**平台**：Habitat-Sim / Habitat-Lab、VLN-CE 基准配置

**流程**：加载场景与指令 → 查看 observation 字段 → 运行 random / heuristic agent → 记录 success 与 SPL

## 18.7 实验步骤

1. 安装 Habitat-Lab（或课程提供的 Docker / 云环境）
2. 加载一个 VLN 数据集样例（R2R 或 VLN-CE）
3. 打印单步 observation：RGB、instruction、agent state
4. 运行官方或课程 heuristic baseline
5. 记录 trajectory 与 success
6. 对比「只按指令关键词走」与「结合视觉」的差异
7. 分析一次失败：指令、视觉、控制、停止哪一类问题

## 18.8 作业交付

1. VLN vs VLA vs SLAM 对比表（1 页）
2. VLN 系统模块图
3. Habitat（或等价仿真）运行截图 + 指标记录
4. 一段说明：为什么 VLN 适合作为 xLeRobot / G1 的长程移动 skill？

## 18.9 常见失败与复盘

- 把 VLN 当成「大模型直接输出电机指令」
- 忽略 SPL，只看是否到达
- 训练集场景过拟合，换房间就失败
- 停止过早 / 过晚

## 18.10 参考开源项目

Habitat-Lab、VLN-CE、L3MVN、DUET 等 — 见 [`references/links.md`](../../references/links.md#lecture-18)。
