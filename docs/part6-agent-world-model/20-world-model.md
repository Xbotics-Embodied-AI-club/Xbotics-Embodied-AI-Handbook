# 第 20 讲：世界模型 —— 预测、规划与数据飞轮

> **所属部分**：第六部分 · Agent、世界模型与进展  
> **Part 负责人**：雨浩  
> **代码**：[`code/lecture20/`](../../code/lecture20/)

## 20.1 本讲目标

理解 **世界模型** 在具身智能中预测什么、与 VLA / VLN / RL 如何配合，以及 **数据飞轮** 在世界模型迭代中的作用。

本讲结束后，学生应能够回答：

1. 世界模型和 VLA 的输入输出有什么不同？
2. 世界模型可预测哪些量（下一帧、下一状态、动作后果）？
3. 为什么世界模型能降低真机试错成本？

本讲结束后，学生应能够完成：

- 画出「策略 + 世界模型 + 数据飞轮」关系图
- 运行或走读一个 Dreamer / 简化 world model 预测 Demo

## 20.2 核心知识点

1. **世界模型定义**：学习环境动态，预测 future state / observation / reward / termination
2. **与 VLA / VLN 区别**
   - VLA/VLN：policy，observation + language → action
   - 世界模型：state + action → next state / next image / risk
3. **预测对象**：下一帧图像、proprioception、接触、失败风险、可达性
4. **方法谱系**：model-based RL、Dreamer 系列、生成式 video prediction、World Action Model
5. **应用价值**：仿真内 rollout、规划、失败预判、数据增强、虚拟试错
6. **SO101 / xLeRobot / G1 差异**
   - 桌面操作：预测物体位移、抓取后果
   - 移动操作：预测 base + arm 联合后果
   - 人形：平衡、接触、行走、操作的多模态预测
7. **数据飞轮（本讲聚焦）**
   ```
   采集 → 训练 → 部署 → 评测 → 失败回流 → 再采集 / 再训练
   ```
8. **局限**：模型漂移、长程误差累积、sim-to-real、计算成本

## 20.3 课堂任务 / 引入案例

**任务**：给定 SO101 抓取 episode 的前 3 帧图像与动作，讨论「如果继续执行当前 action，下一帧夹爪与物体关系可能如何变化」，并用简化 world model 或手工规则做一步预测对比。

## 20.4 方法框架

**世界模型使用三步**：

1. **预测**：当前 state + candidate action → 预测 next state / risk  
2. **决策**：选低 risk / 高 success 的 action（规划或过滤 policy 输出）  
3. **回流**：预测与实际不符的 case → 加入训练集  

## 20.5 有硬件版 Demo

**Demo 名称**：SO101 / G1 状态序列记录 + 一步预测对比（教师演示或离线）

记录 short episode → 用预训练 small world model 预测下一 state → 与真实 next state 对比误差

## 20.6 无硬件仿真版 Demo

**Demo 名称**：DreamerV3 / 简化 RSSM 在 ManiSkill 或课程 toy env 中的一步预测

**流程**：加载 env → 收集 transition → 训练或加载 small world model → 可视化 predicted vs actual observation

## 20.7 实验步骤

1. 理解 transition 格式：`(s, a, r, s', done)`
2. 运行简化 world model 训练或 inference 脚本
3. 可视化一步预测误差
4. 列出 3 类「预测失败」case
5. 说明这些 case 如何进入数据飞轮

## 20.8 作业交付

1. 世界模型 vs VLA vs VLN 对比图
2. 数据飞轮闭环图（结合本课程前 19 讲模块）
3. 预测 Demo 截图或短视频
4. 预测失败 case 分析（≥2 条）

## 20.9 常见失败与复盘

- 把 world model 当成 another VLA
- 只做 video prediction 不接 control
- 忽略 error accumulation
- 失败样本未回流

## 20.10 参考开源项目

DreamerV3、LeRobot、ManiSkill、Unitree Sim IsaacLab — 见 [`references/links.md`](../../references/links.md#lecture-20)。
