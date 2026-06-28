# 第五部分：视觉语言导航 VLN（第 18–19 讲）

## 部分定位

第五部分专门讲 **视觉语言导航（VLN）**：如何让移动机器人根据自然语言指令，在环境中完成「走过去、找对地方、适时停下」，并为后续操作任务（Part 2 移动操作、Part 6 Agent）提供 **`navigate_to` 能力**。

VLN 与 VLA（操作）、SLAM（建图导航）的区别，是本部分的理论主线。

## 覆盖讲次

| 讲次 | 主题 |
|------|------|
| 第 18 讲 | VLN 理论基础（任务定义、指标、系统框架） |
| 第 19 讲 | VLN 实操与评测（Habitat / 固定 seed 评测、skill 接口） |

**负责人**：**新梦**（主责）· 雨浩协同 · 昊旺 / 志凯（移动底盘与 xLeRobot 支持）

## 阶段项目：语言驱动导航评测

**目标**：在仿真（或 xLeRobot）中，对固定指令集完成 ≥20 次 rollout，输出 success / SPL / 失败归因。

| 路径 | 流程 |
|------|------|
| 无硬件 | Habitat VLN-CE / R2R → 批量评测 → `navigate_to` skill mock |
| 有硬件 | xLeRobot 语言导航到目标区域 → 到达判定 → 日志 |

**交付**：评测脚本、结果表、2 个 failure case 分析、`navigate_to` 接口说明

## 与 Part 6 的衔接

- Part 6 第 17 讲 Agent 将调用本部分产出的 **navigate_to** skill  
- Part 6 第 21 讲综合答辩将组合 **VLN + 操作 + Agent**

**建议学习顺序**：可先学 L17 Agent 概念，再学 L18–L19 VLN；或按讲次号 L18→L19 完成本 Part 后进入 Part 6 串联。
