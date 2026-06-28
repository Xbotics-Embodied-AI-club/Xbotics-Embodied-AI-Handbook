# 第 9 讲：操作数据采集、任务设计与评测

> **所属部分**：第三部分 · 端到端机器人操作  
> **代码**：[`code/lecture09/`](../../code/lecture09/)

## 9.1 教学目标

掌握 episode 结构、多模态对齐、数据格式、采集方式、任务设计与评测协议。

## 9.2 核心知识点

1. episode 字段：observation、action、instruction、reward、done、success、timestamp、failure_reason
2. 格式：HDF5、Parquet+MP4、LeRobotDataset、RLDS、rosbag/mcap
3. 采集：脚本、遥操作、kinesthetic、仿真、UMI、ego data（Ego4D/EgoDex）
4. 任务设计：初始/目标状态、成功/失败条件、随机性、固定/随机 seed
5. 评测：success rate、completion time、collision rate、failure 分类

## 9.3 教学设计

```
任务定义 → 采集 → episode 组织 → 格式转换 → 回放 → 固定/随机 seed 评测 → 失败分析 → 回流
```

## 9.4–9.5 Demo

- **有硬件**：SO101 pick-place 20 次采集 → LeRobotDataset
- **无硬件**：ManiSkill/Isaac/RoboCasa rollout 与转换

## 9.6–9.8 实验、作业、复盘

至少 5 条 episode、replay 脚本、字段说明、评测表、failure 分类、LeRobot/HDF5 转换。

## 9.9 配图

episode 结构、多模态对齐、采集方式对比。

## 9.10 参考开源项目

LeRobot、RLDS、robomimic、UMI、Open X-Embodiment、Ego4D — 见 [`references/links.md`](../../references/links.md#lecture-09)。
