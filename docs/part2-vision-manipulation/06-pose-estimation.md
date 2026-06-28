# 第 6 讲：机器人感知与位姿估计

> **所属部分**：第二部分 · 机器人视觉操作  
> **代码**：[`code/lecture06/`](../../code/lecture06/)

## 6.1 教学目标

掌握从 RGB-D 到检测、分割、点云、6D pose、grasp pose 的完整流程。**检测不是终点，grasp pose 才是机器人需要的输出。**

## 6.2 核心知识点

1. 感知任务分类：检测、分割、3D 定位、6D pose、grasp pose、affordance
2. 流程：RGB-D → bbox → mask → 点云 → 3D 中心/6D pose → grasp pose → base 坐标 → 执行
3. bbox / mask / center / object pose / grasp pose 区别
4. 工具：YOLO、SAM/SAM2、Grounded-SAM、Open3D、FoundationPose、SAM-6D
5. 抓取位姿：pre-grasp、grasp、lift、offset、夹爪方向
6. 失败类型：遮挡、反光、透明、深度缺失、杂乱堆叠、mask 错误

## 6.3 教学设计

```
RGB-D → 检测 → 分割 → 点云 → 3D/6D → grasp pose → 坐标转换 → 抓取
```

## 6.4–6.5 Demo

- **有硬件**：SO101 + YOLO + SAM → 点云 → grasp/pre-grasp → 接近抓取
- **无硬件**：样例 RGB-D → 可视化点云与抓取位姿

## 6.6–6.8 实验、作业、复盘

完整 13 步实验流程；交付检测图、mask、点云、pose 代码、失败案例分析。

## 6.9 配图建议

完整流程图、bbox/mask/pose 对比、mask→点云、object vs grasp pose、失败案例、SO101 系统图。

## 6.10 参考开源项目

Ultralytics YOLO、SAM/SAM2、Grounded-SAM、FoundationPose、SAM-6D、AnyGrasp — 见 [`references/links.md`](../../references/links.md#lecture-06)。
