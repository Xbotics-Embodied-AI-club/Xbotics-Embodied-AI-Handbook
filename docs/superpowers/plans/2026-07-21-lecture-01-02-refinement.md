# 第一、二讲精修实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将第一讲收束为具身智能总览课，并把第二讲重构为重点讲解机器人软硬件系统架构的入门核心课。

**Architecture:** 第一讲只建立“为什么、是什么、系统闭环和路线地图”，避免提前展开第二讲的工程架构。第二讲采用“硬件组成—软件栈—软硬件接口—功能分层—ROS2 组织—最小闭环”的认知顺序，并用同一个六关节机械臂任务贯穿。

**Tech Stack:** Markdown、fenced text 流程图、Markdown 表格、仓库现有 Python/ROS2 教学示例。

## Global Constraints

- 面向高校学生、企业开发者和机器人初学者，语言简单直接。
- 理论够用、实验优先，不写成论文综述或 ROS2 API 手册。
- 第一讲保持地图层，第二讲重点解释机器人系统架构及软硬件协作。
- 不虚构已经完成的真机适配或可运行 ROS2 Package。
- 保留现有图片和有效本地链接，不修改无关讲次。

---

### Task 1: 收束第一讲

**Files:**
- Modify: `docs/part1-system-basics/01-introduction.md`
- Modify: `docs/appendix/teaching-principles.md`

- [x] 将教师视角的“课堂组织方式”移入教学原则附录。
- [x] 将人形机器人从技术路线移动到本体与硬件体系。
- [x] 明确二维 reaching Demo 的抽象边界并补充实际运行入口。
- [x] 将作业调整为三项必做、两项选做。
- [x] 用全书实际顺序替换不一致的后续课程路线。

### Task 2: 重构第二讲软硬件系统架构

**Files:**
- Modify: `docs/part1-system-basics/02-ros2-architecture.md`

- [x] 在学习目标中加入软硬件组成、接口与故障定位能力。
- [x] 重写 2.2，覆盖本体、传感器、执行器、计算平台、通信、电源与安全，以及固件、驱动、控制、感知、策略、应用、数据与监控。
- [x] 增加软硬件接口契约、上下行数据流和时序关系。
- [x] 保留四层功能视图，但说明它与物理硬件/软件部署视图的区别。
- [x] 压缩与第一讲重复的前沿模型内容，改为系统位置映射。
- [x] 增加系统架构阅读与设计检查清单。

### Task 3: 两讲衔接与一致性验证

**Files:**
- Modify: `docs/part1-system-basics/01-introduction.md`
- Modify: `docs/part1-system-basics/02-ros2-architecture.md`
- Modify: `code/lecture01/README.md`
- Modify: `code/lecture02/README.md`

- [x] 在第一讲结尾明确第二讲将把概念闭环落实为软硬件系统。
- [x] 在第二讲开头明确承接第一讲，避免重复定义具身智能。
- [x] 修正 README 中对现有 Demo 完成度的表述，不引用不存在的脚本。
- [x] 检查单一一级标题、标题编号、代码围栏、表格和本地链接。
- [x] 检查 Git diff，确认未修改用户的未跟踪下载文件。
