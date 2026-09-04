# Xbotics 具身智能教程

从机器人基础到具身前沿的 **21 讲实践课程**。

这套教程围绕真实机器人任务展开：先讲清概念，再跑通实验，最后分析失败。
六大部分依次覆盖机器人系统基础、视觉操作、端到端操作、强化学习、视觉语言导航，以及 Agent 与世界模型。

## 从这里开始

- 第一次阅读：先看[总定位与学习路径](00-preface/01-positioning.md)，建立全书地图。
- 准备做实验：按[双路径设计](00-preface/02-dual-path.md)选择有硬件或无硬件方案，再查看[环境配置](appendix/hardware-setup.md)。
- 查找某一讲：使用左侧章节导航或搜索；也可以打开[全书目录](SUMMARY.md)。

## 课程目录

```{toctree}
:maxdepth: 1
:caption: 前言

总定位与学习路径 <00-preface/01-positioning>
双路径设计：有硬件版与无硬件仿真版 <00-preface/02-dual-path>
AI 时代的学习方法 <00-preface/03-ai-era-learning>
```

```{toctree}
:maxdepth: 1
:caption: 第一部分：机器人系统基础（第 1–4 讲）

部分导言 <part1-system-basics/00-part-overview>
第 1 讲：具身智能导论 <part1-system-basics/01-introduction>
第 2 讲：机器人系统架构：硬件、软件与 ROS2 <part1-system-basics/02-ros2-architecture>
第 3 讲：机器人本体与控制基础 <part1-system-basics/03-robot-body-control>
第 4 讲：传感器与感知基础 <part1-system-basics/04-sensors-coordinates>
```

```{toctree}
:maxdepth: 1
:caption: 第二部分：机器人视觉操作（第 5–7 讲）

部分导言 <part2-vision-manipulation/00-part-overview>
第 5 讲：仿真环境与操作任务搭建 <part2-vision-manipulation/05-simulation>
第 6 讲：机器人感知与位姿估计 <part2-vision-manipulation/06-pose-estimation>
第 7 讲：机器人操作技能 <part2-vision-manipulation/07-manipulation-skills>
```

```{toctree}
:maxdepth: 1
:caption: 第三部分：端到端机器人操作（第 8–13 讲）

部分导言 <part3-end-to-end/00-part-overview>
第 8 讲：端到端策略导论 —— 从 Pipeline 到 Policy <part3-end-to-end/08-端到端策略导论>
第 9 讲：操作数据闭环 —— 采一份能训练的数据集 <part3-end-to-end/09-操作数据闭环>
第 10 讲：模仿学习实战 —— ACT、Diffusion 与 Flow Matching <part3-end-to-end/10-模仿学习实战>
第 11 讲：VLA 模型导览 —— 从 OpenVLA 到 π0 家族 <part3-end-to-end/11-VLA模型导览>
第 12 讲：VLA 微调实战 —— 从全量 SFT 到 LoRA 上真机 <part3-end-to-end/12-VLA微调实战>
第 13 讲：VLA 前沿 —— 跑得更快、记得更久、用上全身 <part3-end-to-end/13-VLA前沿>
```

```{toctree}
:maxdepth: 1
:caption: 第四部分：强化学习（第 14–16 讲）

部分导言 <part4-reinforcement-learning/00-part-overview>
第 14 讲：强化学习入门 —— 从策略梯度到 PPO <part4-reinforcement-learning/14-强化学习入门>
第 15 讲：GRPO 后训练 —— 让 VLM 学会数数、让 VLA 自我提升 <part4-reinforcement-learning/15-GRPO后训练>
第 16 讲：Off-policy 强化学习 —— 从仿真提速到真机落地 <part4-reinforcement-learning/16-真机强化学习>
```

```{toctree}
:maxdepth: 1
:caption: 第五部分：视觉语言导航 VLN（第 17–18 讲）

部分导言 <part5-vln/00-part-overview>
第 17 讲：VLN 理论基础 <part5-vln/17-vln-theory>
第 18 讲：VLN 实操与评测 <part5-vln/18-vln-practice>
```

```{toctree}
:maxdepth: 1
:caption: 第六部分：Agent、世界模型与前沿进展（第 19–21 讲）

部分导言 <part6-agent-world-model/00-part-overview>
第 19 讲：Embodied Agent <part6-agent-world-model/19-embodied-agent>
第 20 讲：世界模型 —— 预测、规划与数据飞轮 <part6-agent-world-model/20-world-model>
第 21 讲：具身智能前沿进展 —— 综合闭环与答辩 <part6-agent-world-model/21-frontier-progress>
```

```{toctree}
:maxdepth: 1
:caption: 附录与阅读指南

全书目录 <SUMMARY>
全书风格指南 <style-guide>
硬件与仿真环境配置 <appendix/hardware-setup>
教程实施原则 <appendix/teaching-principles>
最终学习成果 <appendix/learning-outcomes>
核心亮点 <appendix/highlights>
```

## 代码与贡献

配套实验代码保留在 [GitHub 仓库的 code 目录](https://github.com/Xbotics-Embodied-AI-club/Xbotics-Embodied-AI-Handbook/tree/main/code)。
开源资料可查阅[资源链接汇总](https://github.com/Xbotics-Embodied-AI-club/Xbotics-Embodied-AI-Handbook/blob/main/references/links.md)。

如果你发现问题或希望参与写作，请先阅读[贡献指南](https://github.com/Xbotics-Embodied-AI-club/Xbotics-Embodied-AI-Handbook/blob/main/CONTRIBUTING.md)。
