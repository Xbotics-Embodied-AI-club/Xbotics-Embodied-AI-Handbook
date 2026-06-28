# 第 17 讲：Robot Agent、任务规划与技能调用

> **所属部分**：第六部分 · Agent、世界模型与进展  
> **Part 负责人**：雨浩 · 协同：富平（Agent）  
> **代码**：[`code/lecture17/`](../../code/lecture17/)

## 17.1 教学目标

设计 Robot Agent：语言指令分解为可执行 skill；LLM/VLM 做规划而非直接控电机。

## 17.2 核心知识点

Skill library、工具调用、任务分解、状态检查、长程执行、失败重规划、memory、安全边界。

**SO101 skills**：detect_object、move_to、grasp、place、check_success

**xLeRobot**：navigate_to、approach_table、mobile_grasp

**G1**：stand_up、walk_to、turn_to、detect_object、reach_object、check_state

## 17.3–17.5 Demo

**Demo A — xLeRobot**：「把红色方块放到蓝色区域」→ 6 步 skill 计划

**Demo B — G1**：「走到桌子旁，面向红色方块，准备抓取」

**无硬件**：LangGraph + ROS2 mock + ManiSkill/RoboCasa + G1 仿真 mock

## 17.6–17.8 实验、作业、复盘

skill 接口、Agent 计划日志、执行视频、失败重规划案例、G1 高层流程图。

## 17.9 参考

LangGraph、LangChain、ROS2 examples、MoveIt2、Unitree SDK2 — 见 [`references/links.md`](../../references/links.md#lecture-17)。
