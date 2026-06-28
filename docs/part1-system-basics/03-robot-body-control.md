# 第 3 讲：机器人本体与控制基础 —— 从硬件结构到动作空间

> **所属部分**：第一部分 · 机器人系统基础  
> **代码**：[`code/lecture03/`](../../code/lecture03/)

## 3.1 教学目标

理解机器人硬件结构、控制方式与动作空间，明确模型输出的 action 如何变成真实控制指令。

## 3.2 核心知识点

1. 本体：连杆、关节、电机、减速器、编码器、夹爪、底盘、相机、IMU
2. 形态：移动底盘、单/双臂、灵巧手、四足、人形
3. Xbotics 硬件：SO101 / xLeRobot / Imeta-Y1 / Unitree G1 差异
4. 控制：开环/闭环、位置/速度/力矩、PID、频率与限位
5. 运动学：关节空间 vs 笛卡尔空间、FK/IK、奇异位形
6. 动作空间：joint position/velocity、eef pose/delta、gripper、base velocity、action chunk
7. 学习与部署：normalization、尺度、平滑、控制频率、跨机器人迁移问题

## 3.3 教学设计

```
硬件结构 → 自由度与关节空间 → 运动学 → 控制方式 → 动作表示 → action 到控制指令
```

## 3.4 有硬件版 Demo

**SO101**：单关节/多关节/夹爪/eef delta 对比，记录轨迹曲线。

**G1 演示**（教师）：站立、行走、转向，对比全身协调复杂度。

## 3.5 无硬件仿真版 Demo

MoveIt2、MuJoCo Menagerie、ManiSkill、Isaac Lab、Unitree RL Gym G1：对比 joint vs eef 控制、动作尺度与频率影响。

## 3.6 实验步骤

读关节状态 → 单/多关节控制 → eef delta → 夹爪 → 记录曲线 → 修改幅度与频率 → 对比稳定性。

## 3.7 作业交付

硬件组成图、关节列表、三种动作空间对比表、轨迹曲线、实验视频、跨机器人迁移说明。

## 3.8 常见失败与复盘

动作过大、关节超限、频率过低、坐标系错误、仿真与真机尺度不一致、未做 normalization 等。

## 3.9 参考开源项目

MoveIt2、MuJoCo Menagerie、ManiSkill、Isaac Lab、LeRobot、Unitree SDK2 — 见 [`references/links.md`](../../references/links.md#lecture-03)。
