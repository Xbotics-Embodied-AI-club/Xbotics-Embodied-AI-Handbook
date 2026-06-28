# 第 15 讲：强化学习运动控制 —— 以 Unitree G1 Locomotion 为例

> **代码**：[`code/lecture15/`](../../code/lecture15/)

## 15.1 教学目标

理解 locomotion RL：站立、行走、转向、速度跟踪、扰动恢复；G1 state/action/reward；真机安全边界。

## 15.2 核心知识点

G1 state：base velocity、IMU、joint pos/vel、foot contact、command velocity。

G1 action：joint target/delta、PD target、lower-body / whole-body。

Reward：速度跟踪、姿态稳定、能耗/平滑/接触/摔倒/限位惩罚。

Recovery：扰动后恢复平衡与站立。

## 15.3–15.5 Demo

- **有硬件**：G1 安全演示（站立/行走/转向/读状态），**不跑未验证 RL**
- **无硬件**：Unitree RL Gym/Lab/Mjlab 或 Isaac Lab G1 locomotion + 扰动测试

## 15.6 作业

仿真截图/视频、state/action/reward 表、recovery 分析、真机 RL 风险说明。

## 参考

Unitree RL Gym/Lab/Mjlab、Isaac Lab — 见 [`references/links.md`](../../references/links.md#lecture-15)。
