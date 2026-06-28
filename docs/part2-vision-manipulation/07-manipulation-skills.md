# 第 7 讲：机器人操作技能 —— 基于位姿的抓取、放置与失败恢复

> **所属部分**：第二部分 · 机器人视觉操作  
> **代码**：[`code/lecture07/`](../../code/lecture07/)

## 7.1 教学目标

理解完整 pick-place 状态机：pre-grasp → approach → grasp → lift → place → check → retry。

## 7.2 核心知识点

1. 位姿驱动：object pose、grasp pose、pre-grasp、lift、place、retreat
2. 从 object pose 生成 grasp：中心、夹爪方向、offset、可达性过滤
3. 控制：joint/eef/trajectory、MoveIt2、visual servo、gripper
4. 成功检测：夹爪宽度、抬升、视觉、力/电流
5. 失败恢复：重检测、改抓取点、降速、失败样本回流
6. 模块化 vs 端到端对比

## 7.3 关键流程（从位姿到动作）

给定 `object_pose_base`，依次生成 pre_grasp、grasp、lift、place；桌面抓取常见从上往下或侧面接近。

**示例**：

```
object_pose_base = [0.20, 0.05, 0.03, 0,0,0,1]
pre_grasp_pose   = [0.20, 0.05, 0.12, ...]
grasp_pose       = [0.20, 0.05, 0.05, ...]
lift_pose        = [0.20, 0.05, 0.18, ...]
place_pose       = [0.10,-0.15, 0.12, ...]
```

## 7.4–7.6 Demo

- **有硬件**：SO101 规则版 pick-place 全流程
- **无硬件**：MuJoCo/Isaac/ManiSkill/robosuite/MoveIt2 仿真 pick-place

## 7.7 作业交付

pick-place 代码、pose 转换代码、四体位姿可视化、演示视频、成功/失败记录表、失败归因说明。

## 7.8–7.11 复盘、配图、参考

Pick-Place 状态机、轨迹图、失败来源分析、模块化 vs 端到端对比 — MoveIt2、ManiSkill、robosuite 等见 [`references/links.md`](../../references/links.md#lecture-07)。
