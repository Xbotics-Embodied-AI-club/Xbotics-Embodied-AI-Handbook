<div align="center">

  <pre style="font-family: 'Courier New', monospace; font-size: 13px; color: #1a1a2e; margin: 0; padding: 0; line-height: 1.2;">
  ██╗  ██╗██████╗  ██████╗ ████████╗██╗ ██████╗███████╗
  ╚██╗██╔╝██╔══██╗██╔═══██╗╚══██╔══╝██║██╔════╝██╔════╝
   ╚███╔╝ ██████╔╝██║   ██║   ██║   ██║██║     ███████╗
   ██╔██╗ ██╔══██╗██║   ██║   ██║   ██║██║     ╚════██║
  ██╔╝ ██╗██████╔╝╚██████╔╝   ██║   ██║╚██████╗███████║
  ╚═╝  ╚═╝╚═════╝  ╚═════╝    ╚═╝   ╚═╝ ╚═════╝╚══════╝
  </pre>

  # Xbotics 具身智能教程（21 讲）

  <p align="center">
    <em>从机器人基础到具身前沿 · Agent / VLN / 世界模型 · 真机优先、仿真兜底</em>
  </p>

  <p align="center">
    📌 <a href="#quick-start">快速开始</a> · 📚 <a href="#course-map">课程地图</a> · 👥 <a href="#team">团队分工</a> · 📅 <a href="#schedule">时间安排</a> · ✍️ <a href="#writing-style">写作风格</a> · 🤝 <a href="CONTRIBUTING.md">贡献指南</a>
  </p>

  <p align="center">
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-0052cc?style=for-the-badge&labelColor=1a1a2e" alt="License"></a>
    <a href="https://github.com/Xbotics-Embodied-AI-club/Xbotics-Embodied-Guide"><img src="https://img.shields.io/badge/社区-Xbotics%20Embodied-4ecdc4?style=for-the-badge&labelColor=1a1a2e" alt="Community"></a>
    <img src="https://img.shields.io/badge/讲次-21_讲-ff6b6b?style=for-the-badge&labelColor=1a1a2e" alt="21 Lectures">
  </p>

</div>

---

《Xbotics 具身智能教程（21 讲）》是 [Xbotics 具身智能社区](https://github.com/Xbotics-Embodied-AI-club/Xbotics-Embodied-Guide) 出品的**系统实践教程**：不是论文综述，也不是纯科普，而是一套围绕真实机器人任务、可跑通 Demo、可复盘失败的工程实践课程。

本仓库包含 **21 讲**书稿正文 · 每讲代码 · 配图资料 · 协作规范（**六大部分**），与社区 [Xbotics-Embodied-Guide](https://github.com/Xbotics-Embodied-AI-club/Xbotics-Embodied-Guide) 互补——Guide 帮你看全局，**本教程**带你动手跑完整闭环。

> 仓库名 [Xbotics-Embodied-AI-Handbook](https://github.com/Xbotics-Embodied-AI-club/Xbotics-Embodied-AI-Handbook)；课程已从原「十八讲」扩展为 **二十一讲**（新增 VLN×2，世界模型与前沿进展分讲）。

![具身智能学习路线示意](https://github.com/user-attachments/assets/054c89b9-d114-4477-b751-a01f2e7a6376)

> 上图与 [Embodied-Guide](https://github.com/Xbotics-Embodied-AI-club/Xbotics-Embodied-Guide) 共用社区视觉素材；各讲专用配图见 [`assets/figures/`](assets/figures/)。

<span id="quick-start"></span>

## 快速开始

**三步上手本仓库：**

```bash
git clone git@github.com:Xbotics-Embodied-AI-club/Xbotics-Embodied-AI-Handbook.git
cd Xbotics-Embodied-AI-Handbook
```

1. **读目录**：打开 [`docs/SUMMARY.md`](docs/SUMMARY.md)，从 [总定位](docs/00-preface/01-positioning.md) 建立系统地图。
2. **认领讲次**：在 [`meta/status.md`](meta/status.md) 更新负责讲次；分工见 [团队](#team)。
3. **写稿 + 跑 Demo**：文稿按 [`templates/lecture-template.md`](templates/lecture-template.md)；代码放 `code/lectureNN/`；配图放 `assets/figures/lectureNN/`。

**试跑第 1 讲 Demo（无硬件）：**

```bash
cd code/lecture01
python -m venv .venv && .venv\Scripts\activate   # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python simulation/minimal_reach.py
```

| 目录 | 说明 |
|------|------|
| [`docs/`](docs/) | 书稿正文（**六大部分** · 21 讲） |
| [`code/`](code/) | 每讲 Demo（`lecture01` … `lecture21`，共 21 讲） |
| [`assets/`](assets/) | 配图与演示视频 |
| [`references/`](references/) | 开源链接集中维护 |
| [`meta/`](meta/) | 大纲、进度、贡献者 |
| [`templates/`](templates/) | 单讲标准模板 |

---

<span id="course-map"></span>

## 课程地图

编号已连续：**L1–L16 → L17–L18 VLN → L19 Agent → L20 世界模型 → L21 答辩**。
每讲可展开子目录；点击标题进入文稿。

### 前言

| 模块 | 说明 | 阅读 |
|------|------|------|
| 总定位与学习路径 | 课程要解决什么、21 讲主线 | [📖 阅读](docs/00-preface/01-positioning.md) |
| 双路径设计 | 有硬件版 / 无硬件仿真版 | [📖 阅读](docs/00-preface/02-dual-path.md) |
| AI 时代学习方法 | 问题驱动、云平台、协作写作 | [📖 阅读](docs/00-preface/03-ai-era-learning.md) |

### 第一部分 · 机器人系统基础（第 1–4 讲）

**阶段项目**：机器人基础控制闭环 · [部分导言](docs/part1-system-basics/00-part-overview.md)

#### 第 1 讲 · 具身智能导论

[📖 文稿](docs/part1-system-basics/01-introduction.md) · [💻 代码](code/lecture01/)

<details>
<summary>本讲子目录（9）</summary>

- 本讲提要
- 1.1 本讲导入：为什么 AI 进入物理世界后问题变了？
- 1.2 第一张地图：什么是具身智能？
- 1.3 具身智能系统闭环：机器人到底如何完成一个任务？
- 1.4 技术路线总览：从传统机器人到 VLA 与世界模型
- 1.5 应用场景与课程硬件体系
- 1.6 本讲体验 Demo：从目标到动作
- 1.7 作业、讨论与复盘
- 1.8 推荐学习资源与本讲小结

</details>

#### 第 2 讲 · 机器人系统架构（硬件 / 软件 / ROS2）

[📖 文稿](docs/part1-system-basics/02-ros2-architecture.md) · [💻 代码](code/lecture02/)

<details>
<summary>本讲子目录（9）</summary>

- 2.1 本讲主线：从“单个模型”到“机器人系统”
- 2.2 机器人系统架构：硬件、软件与接口
- 2.3 ROS2 基础：机器人模块如何通信？
- 2.4 学习模型如何接入机器人系统
- 2.5 机器人系统闭环：从状态读取到数据记录
- 2.6 Demo 设计：基于 ROS2 的 LeRobot / SO101 最小闭环
- 2.7 实验步骤
- 2.8 作业交付与失败复盘
- 2.9 参考开源项目

</details>

#### 第 3 讲 · 机器人本体与控制基础

[📖 文稿](docs/part1-system-basics/03-robot-body-control.md) · [💻 代码](code/lecture03/)

<details>
<summary>本讲子目录（10）</summary>

- 3.1 本讲主线：从机器人身体到 action 含义
- 3.2 机器人本体的基本组成
- 3.3 运动结构基础
- 3.4 常见机器人形态与课程平台
- 3.5 控制基础：从开环到关节控制
- 3.6 PID 与模型 action 的执行边界
- 3.7 运动学：从关节状态到末端动作
- 3.8 动作空间：模型输出如何进入机器人接口
- 3.9 动作尺度、归一化与控制频率
- 3.10 人形机器人为什么更复杂？

</details>

#### 第 4 讲 · 传感器、坐标系与相机模型

[📖 文稿](docs/part1-system-basics/04-sensors-coordinates.md) · [💻 代码](code/lecture04/)

<details>
<summary>本讲子目录（12）</summary>

- 本讲目标
- 4.1 本讲主线：从“看到”到“能执行”
- 4.2 机器人常用传感器：特点、参数与任务选型
- 4.3 传感器数据如何进入机器人系统？
- 4.4 感知结果如何服务机器人任务？
- 4.5 多传感器数据对齐与融合
- 4.6 相机原理与坐标转换
- 4.7 感知误差与任务失败
- 4.8 标定：让传感器和真实空间对齐
- 4.9 坐标系、位姿表示与 ROS 2 TF
- 4.10 Demo 设计、作业交付与工程实战建议
- 4.11 参考开源项目

</details>


### 第二部分 · 机器人视觉操作（第 5–7 讲）

**阶段项目**：视觉目标定位与模块化抓取 · [部分导言](docs/part2-vision-manipulation/00-part-overview.md)

#### 第 5 讲 · 仿真环境与操作任务搭建

[📖 文稿](docs/part2-vision-manipulation/05-simulation.md) · [💻 代码](code/lecture05/)

<details>
<summary>本讲子目录（8）</summary>

- 1 仿真里的机器人动了，为什么还不算任务搭好
- 2 先写任务契约：把真实任务翻译成七个约定
- 3 在 MuJoCo 中第一次兑现契约：SO-101
- 4 从一次运行到一套环境：Isaac Lab
- 5 换成 G1：任务契约经得住人形机器人吗
- 6 仿真成功为什么不等于真机成功
- 7 把失败变成下一次实验
- 8 本讲小结：从任务契约走向真实感知

</details>

#### 第 6 讲 · 机器人感知与位姿估计

[📖 文稿](docs/part2-vision-manipulation/06-pose-estimation.md) · [💻 代码](code/lecture06/)

<details>
<summary>本讲子目录（8）</summary>

- 1 从"看得到"到"抓得到"：位姿估计在机器人操作中的定位
- 2 机器人感知任务体系与标准抓取流水线
- 3 视觉感知前端：目标检测分割与点云提取
- 4 6D 物体位姿估计：原理与主流技术方案
- 5 抓取位姿生成：从物体位姿到机器人可执行目标
- 6 感知失败分析与鲁棒性优化
- 7 位姿估计抓取实战实验与作业解析
- 8 本讲小结

</details>

#### 第 7 讲 · 操作技能：抓取、放置与失败恢复

[📖 文稿](docs/part2-vision-manipulation/07-manipulation-skills.md) · [💻 代码](code/lecture07/)

<details>
<summary>本讲子目录（7）</summary>

- 1 坐标有了，动作还没有
- 2 从物体位姿到夹爪该去哪里
- 3 五个位姿怎么串成一次不会乱跑的任务
- 4 机械臂动完了，不等于任务成功
- 5 失败不是一句“没抓起来”
- 6 实验：把整套抓取—放置流程跑起来
- 7 收一下：从一个坐标，到一次完整任务

</details>


### 第三部分 · 端到端机器人操作（第 8–13 讲）

**阶段项目**：端到端操作策略学习 · [部分导言](docs/part3-end-to-end/00-part-overview.md)

#### 第 8 讲 · 端到端策略导论：从 Pipeline 到 Policy

[📖 文稿](docs/part3-end-to-end/08-端到端策略导论.md) · [📄 PDF](docs/part3-end-to-end/pdf/第8讲_端到端策略导论.pdf) · [💻 代码](code/vla/1_policy_rollout/)

<details>
<summary>本讲子目录（5）</summary>

- 1 两种机器人：手写 pipeline 还是端到端 policy
- 2 什么是 policy
- 3 从训练到部署：为什么 loss 低了还会失败
- 4 实验：用 $\pi_0$ 跑通第一个策略闭环
- 5 本讲小结

</details>

#### 第 9 讲 · 操作数据闭环：采一份能训练的数据集

[📖 文稿](docs/part3-end-to-end/09-操作数据闭环.md) · [📄 PDF](docs/part3-end-to-end/pdf/第9讲_操作数据闭环.pdf) · [💻 代码](code/vla/2_data_collection/)

<details>
<summary>本讲子目录（6）</summary>

- 1 为什么数据决定 VLA 的上限
- 2 遥操作采集方法概述
- 3 LeRobot 框架与数据集
- 4 实验：采一段抓取数据
- 5 任务设计与评测协议
- 6 小结与下一讲

</details>

#### 第 10 讲 · 模仿学习实战：ACT、Diffusion 与 Flow Matching

[📖 文稿](docs/part3-end-to-end/10-模仿学习实战.md) · [📄 PDF](docs/part3-end-to-end/pdf/第10讲_模仿学习实战.pdf) · [💻 代码](code/vla/3_imitation_learning/)

<details>
<summary>本讲子目录（6）</summary>

- 1 朴素行为克隆撞墙：为什么需要生成式策略
- 2 ACT：用 Action Chunking + CVAE 建模示教分布
- 3 实验：训练、部署并解析 ACT
- 4 Diffusion Policy：把去噪过程搬进动作生成
- 5 Flow Matching：从扩散到连续流
- 6 本讲小结

</details>

#### 第 11 讲 · VLA 模型导览：从 OpenVLA 到 π0 家族

[📖 文稿](docs/part3-end-to-end/11-VLA模型导览.md) · [📄 PDF](docs/part3-end-to-end/pdf/第11讲_VLA模型导览.pdf) · [💻 代码](code/vla/4_vla_inference/)

<details>
<summary>本讲子目录（8）</summary>

- 1 从 LLM 到 VLA：模型是怎么学会"动手"的
- 2 OpenVLA：开源视觉-语言-动作模型
- 3 $\pi_0$：基于 Flow Matching 的视觉-语言-动作流模型
- 4 $\pi_0$-FAST：高效动作 Token 化
- 5 $\pi_{0.5}$：开放世界泛化的 VLA
- 6 VLA-0：把动作直接当文本说出来
- 7 SmolVLA：小模型与异步推理
- 8 本讲小结

</details>

#### 第 12 讲 · VLA 微调实战：从全量 SFT 到 LoRA 上真机

[📖 文稿](docs/part3-end-to-end/12-VLA微调实战.md) · [📄 PDF](docs/part3-end-to-end/pdf/第12讲_VLA微调实战.pdf) · [💻 代码](code/vla/5_vla_finetune/)

<details>
<summary>本讲子目录（6）</summary>

- 1 微调概述：两个分类维度
- 2 全量 SFT
- 3 LoRA SFT
- 4 多卡训练指南
- 5 Sim2Real 与 Real2Sim
- 6 本讲小结

</details>

#### 第 13 讲 · VLA 前沿：跑得更快、记得更久、用上全身

[📖 文稿](docs/part3-end-to-end/13-VLA前沿.md) · [📄 PDF](docs/part3-end-to-end/pdf/第13讲_VLA前沿.pdf)

<details>
<summary>本讲子目录（4）</summary>

- 1 实时推理：让大模型跟上控制频率
- 2 长时记忆：让 VLA 记得住、想得到
- 3 人形全身 VLA：让语言驱动整个身体
- 4 本讲小结与下一讲

</details>


### 第四部分 · 强化学习（第 14–16 讲）

[部分导言](docs/part4-reinforcement-learning/00-part-overview.md)

#### 第 14 讲 · 强化学习入门：从策略梯度到 PPO

[📖 文稿](docs/part4-reinforcement-learning/14-强化学习入门.md) · [📄 PDF](docs/part4-reinforcement-learning/pdf/第14讲_强化学习入门.pdf) · [💻 代码](code/rl/1_rl_basics/)

<details>
<summary>本讲子目录（7）</summary>

- 1 从模仿到试错
- 2 把「学走路」写成强化学习问题
- 3 强化学习方法分类
- 4 REINFORCE：最朴素的策略梯度方法
- 5 A2C：请一位评分员来降方差
- 6 PPO：给更新步子加护栏
- 7 本讲小结

</details>

#### 第 15 讲 · GRPO 后训练：让 VLM 学会数数、让 VLA 自我提升

[📖 文稿](docs/part4-reinforcement-learning/15-GRPO后训练.md) · [📄 PDF](docs/part4-reinforcement-learning/pdf/第15讲_GRPO后训练.pdf) · [💻 代码](code/rl/2_grpo_posttraining/)

<details>
<summary>本讲子目录（10）</summary>

- 1 从 PPO 到 GRPO：为什么不再请评分员
- 2 可验证奖励，和一个题眼
- 3 VLM 后训练：从 DeepSeek-R1 到会数数的小模型
- 4 实验：让 VLM 学会数数
- 5 VLA 后训练：把「答对没」换成「干成没」
- 6 SimpleVLA-RL：把 GRPO 搬上 VLA 的代表作
- 7 RLinf：VLA 强化学习的基建化
- 8 实验：让 VLA-0 自我提升成功率
- 9 $\pi^{*}_{0.6}$：让 VLA 从真实经验中学习
- 10 本讲小结

</details>

#### 第 16 讲 · Off-policy 强化学习：从仿真提速到真机落地

[📖 文稿](docs/part4-reinforcement-learning/16-真机强化学习.md) · [📄 PDF](docs/part4-reinforcement-learning/pdf/第16讲_真机强化学习.pdf) · [💻 代码](code/rl/3_offpolicy/)

<details>
<summary>本讲子目录（10）</summary>

- 1 上一讲留下的账：真机上样本有多贵
- 2 价值方法与 off-policy：把稀疏的成功传播开
- 3 DDPG：把确定性策略带进连续动作
- 4 TD3：用两个 Critic 治高估
- 5 SAC：把探索变成一条原则
- 6 squint：从像素学会，顺手把数据也生成了
- 7 同一个 G1，换 off-policy 再走一遍：15 分钟学会走路
- 8 真机压轴：HIL-SERL 让 SO-101 学会接触型任务
- 9 结课全景：三种强化学习，与一条完整的链路
- 10 本讲小结

</details>


### 第五部分 · 视觉语言导航 VLN（第 17–18 讲）

**阶段项目**：语言驱动导航评测 · [部分导言](docs/part5-vln/00-part-overview.md)

#### 第 17 讲 · 视觉语言导航：任务与基准

[📖 文稿](docs/part5-vln/17-vln-theory.md) · [💻 代码](code/lecture17/)

<details>
<summary>本讲子目录（5）</summary>

- 1 引言：问题定义与系统边界
- 2 典型任务范式
- 3 仿真模拟器
- 4 基准数据集
- 5 评估指标

</details>

#### 第 18 讲 · 视觉语言导航：方法与实践

[📖 文稿](docs/part5-vln/18-vln-practice.md) · [💻 代码](code/lecture18/)

<details>
<summary>本讲子目录（5）</summary>

- 1 引言：VLN 方法演进图谱
- 2 序列建模与端到端学习
- 3 Transformer架构与预训练
- 4 LLM 驱动的通用导航
- 5 总结

</details>


### 第六部分 · Agent、世界模型与前沿进展（第 19–21 讲）

**阶段项目**：移动 + 操作综合答辩 · [部分导言](docs/part6-agent-world-model/00-part-overview.md)

#### 第 19 讲 · Embodied Agent

[📖 文稿](docs/part6-agent-world-model/19-embodied-agent.md) · [💻 代码](code/lecture19/)

<details>
<summary>本讲子目录（11）</summary>

- 19.1 本讲目标
- 19.2 核心知识点：Agent 介绍
- 19.3 Embodied Agent 的通用架构
- 19.4 跟着代码运行一个 Embodied Agent
- 19.5 现在的 Embodied Agent 能做到什么？
- 19.6 有硬件版 Demo
- 19.7 无硬件仿真版 Demo
- 19.8 实验步骤
- 19.9 作业交付
- 19.10 常见失败与复盘
- 19.11 参考资料

</details>

#### 第 20 讲 · 世界模型与数据飞轮

[📖 文稿](docs/part6-agent-world-model/20-world-model.md) · [💻 代码](code/lecture20/) · [Fast-WAM](https://github.com/yuantianyuan01/FastWAM)

<details>
<summary>本讲子目录（6）</summary>

- 20.1 为什么需要世界模型
- 20.2 世界模型的三种视角
- 20.3 核心理论
- 20.4 技术路线与代表工作
- 20.5 Fast-WAM 复现教学
- 20.6 误区、复盘与小结

</details>

#### 第 21 讲 · 前沿进展与综合答辩

[📖 文稿](docs/part6-agent-world-model/21-frontier-progress.md) · [💻 代码](code/lecture21/)

<details>
<summary>本讲子目录（6）</summary>

- 21.1 本讲目标
- 21.2 方向一：WAM Scaling——世界模型正在走向基础设施
- 21.3 方向二：Agentic Action——Agent 正在进入动作接口
- 21.4 方向三：Simulation × RL——仿真正在成为经验工厂
- 21.5 方向四：第一人称数据——把人类经验接入机器人
- 21.6 参考资料

</details>


### 附录

| 模块 | 链接 |
|------|------|
| 全书风格指南 | [📖](docs/style-guide.md) |
| 硬件与仿真环境配置 | [📖](docs/appendix/hardware-setup.md) |
| 教程实施原则 | [📖](docs/appendix/teaching-principles.md) |
| 最终学习成果 | [📖](docs/appendix/learning-outcomes.md) |
| 核心亮点 | [📖](docs/appendix/highlights.md) |

**教程主线（一览）：**

```
L1 导论 → L2 架构/ROS2 → L3 本体控制 → L4 传感器坐标
→ L5 仿真 → L6 位姿 → L7 操作技能
→ L8 端到端导论 → L9 数据 → L10 模仿学习 → L11–13 VLA
→ L14–16 强化学习 → L17–18 VLN → L19 Agent → L20 世界模型 → L21 答辩
```

## 五个阶段项目

| 阶段 | 讲次 | 项目名称 |
|------|------|----------|
| 一 | 第 1–4 讲 | 机器人基础控制闭环 |
| 二 | 第 5–7 讲 | 视觉目标定位与模块化抓取 |
| 三 | 第 8–13 讲 | 端到端操作策略学习 |
| 四 | 第 17–18 讲 | 语言驱动导航评测（VLN） |
| 五 | 第 19、20–21 讲 | 移动 + 操作综合答辩 |

---

<span id="team"></span>

## 团队分工

| 角色 | 成员 | 负责范围 |
|------|------|----------|
| **第一部分负责人** | **丛林** | 第 1–4 讲及阶段项目一；协同：翼茗、锦丰 |
| **第二部分负责人** | **育帆** | 第 5–7 讲及阶段项目二；协同：昊旺、志凯、彤彤、宝华 |
| **第三 & 第四部分负责人** | **harry** | 第 8–16 讲及阶段项目三；协同：陈老师、诸老师、罗辑 |
| **第五部分负责人** | **新梦** | 第 17–18 讲（VLN）及阶段项目四；雨浩协同 |
| **第六部分负责人** | **雨浩** | 第 19、20–21 讲及阶段项目五；富平（L19）、煜恒（L20） |
| **大纲总控** | **丛林、木木** | 6 月 30 日整体大纲检查与编写规划 |
| **项目代码管理** | **志凯** | `code/` 结构、复现入口、跨讲依赖与 CI |
| **文稿校验** | **乙然** | 结构/术语/模板合规检查；进度提醒 |
| **进度跟踪** | **乙然** | 每周三 21:00 进度同步提醒 |

详细分工与认领表：[`meta/contributors.md`](meta/contributors.md) · 写作进度：[`meta/status.md`](meta/status.md)

---

<span id="schedule"></span>

## 时间安排

| 节点 | 日期 | 负责人 | 交付物 |
|------|------|--------|--------|
| 整体大纲检查 & 编写规划 | **6 月 30 日** | 丛林、木木 | 大纲定稿、分工确认、里程碑 |
| 各部分大纲完成 | **7 月 5 日前** | 丛林、育帆（Part 2）、**新梦（Part 5）**、harry、雨浩（Part 6） | 各 Part 细纲 |
| 第一版初稿 | **7 月 26 日前** | 全体 | 文稿 + Demo 骨架 + 配图清单 |
| 第一版修改版 | **8 月 2 日前** | 全体 | 根据校验与互审意见修订 |
| 例行进度同步 | **每周三 21:00** | 乙然提醒，各 Part 负责人汇报 | 进度更新 `meta/status.md` |

---

<span id="writing-style"></span>

## 课程内容风格

本课程**不是论文综述，也不是纯科普**，而是面向**高校学生、企业开发者和机器人初学者**的具身智能**工程实践**课程。

> **完整约定**（标题编号、配图、定稿检查清单、旧稿迁移优先级）：见 [`docs/style-guide.md`](docs/style-guide.md)。下文为摘要。

### 整体风格（四点）

| 原则 | 说明 |
|------|------|
| **简单直接** | 少用大词、少堆论文名；用真实机器人任务解释概念 |
| **任务驱动** | 每讲围绕具体任务，如「让机械臂到达目标点」「把碗放到盘子上」 |
| **讲清楚，再实验** | 概念讲到能解释「为何」和排错；不堆无关推导，也不为省篇幅砍必要直觉（详见风格指南 §0） |
| **重视失败复盘** | 不只展示成功 Demo，还要教怎么看失败、怎么归因、怎么改进 |

**推荐写法：**

```
提出问题 → 讲清概念 → 跑通 Demo → 分析失败 → 完成作业
```

### 每讲必备章节（10 模块）

写作请严格遵循 [`templates/lecture-template.md`](templates/lecture-template.md)：

| # | 模块 | 要求 |
|---|------|------|
| 1 | **本讲目标** | 学完后能回答什么、能完成什么（用问句列出） |
| 2 | **核心知识点** | 5–10 条，每条必须服务后面的实验，不堆概念 |
| 3 | **课堂任务 / 引入案例** | 用具体任务带出概念（如 LIBERO「把碗放到盘子上」） |
| 4 | **方法框架** | 把案例经验抽象为可复用框架（如训练效果分析三步法） |
| 5 | **有硬件版 Demo** | SO101 / xLeRobot / Imeta / G1 真机路径 |
| 6 | **无硬件仿真版 Demo** | MuJoCo / ManiSkill / Isaac Lab 等兜底 |
| 7 | **实验步骤** | 可逐步执行的操作清单 |
| 8 | **作业交付** | 明确产出物（代码、图、视频、表格） |
| 9 | **常见失败与复盘** | 失败现象 + 复盘问题 |
| 10 | **参考开源项目** | 链接汇总到 [`references/links.md`](references/links.md) |

**方法框架示例 — 训练效果分析三步法：**

1. 看 **loss**：模型是否学进数据  
2. 看 **成功率**：任务是否做成  
3. 看 **rollout 视频**：失败卡在哪里  

**失败分析五类框架**：视觉 · 动作 · 时机 · 目标位姿 · 数据覆盖

### 文章与配图风格

| 维度 | 约定 |
|------|------|
| **语言** | 中文为主；术语与 [`meta/outline.yaml`](meta/outline.yaml) 及 [Embodied-Guide](https://github.com/Xbotics-Embodied-AI-club/Xbotics-Embodied-Guide) 保持一致 |
| **结构** | 层级清晰：`# 第 N 讲` → `## N.1 本讲目标`；大块之间用 `---` 分隔 |
| **代码** | 正文只放关键片段；完整代码在 `code/lectureNN/` |
| **配图命名** | `assets/figures/lectureNN/fig-NN-M-英文名.png` |
| **社区共用图** | 系统地图、学习路线等可与 [Xbotics-Embodied-Guide](https://github.com/Xbotics-Embodied-AI-club/Xbotics-Embodied-Guide) 共用；引用时注明来源，专用示意图放本仓库 `assets/` |
| **图标约定** | ⭐ 必看 · 🧪 实作 · 📦 代码/数据 · 📄 论文 · 🎥 视频（与 Guide 一致） |
| **双路径** | 每讲必须分别写清有硬件 / 无硬件方案 |

---

## 项目理念

- **系统地图清晰**：21 讲、六大部分，先全局再模块，最后拼成闭环  
- **真机优先、仿真兜底**：没有硬件也能完成每讲核心实验  
- **数据和失败同等重要**：成功 Demo 之外，必须有失败样本与回流路径  
- **与社区联动**：路线查 [Embodied-Guide](https://github.com/Xbotics-Embodied-AI-club/Xbotics-Embodied-Guide)，求职查 [Embodied-AI-Job](https://github.com/Xbotics-Embodied-AI-club/Xbotics-Embodied-AI-Job)

## 相关仓库

| 仓库 | 关系 |
|------|------|
| [Xbotics-Embodied-Guide](https://github.com/Xbotics-Embodied-AI-club/Xbotics-Embodied-Guide) | 学习路线、资源导航、社区共用配图 |
| [Xbotics-Embodied-AI-Job](https://github.com/Xbotics-Embodied-AI-club/Xbotics-Embodied-AI-Job) | 具身智能求职信息 |
| [LeRobot](https://github.com/huggingface/lerobot) | 本教程主要数据采集与训练框架 |

---

## 参与贡献

欢迎 Xbotics 社区成员按分工提交 PR。完整流程见 [`CONTRIBUTING.md`](CONTRIBUTING.md)。

**协作要点：**

1. 只改你负责的 Part / 讲次（见 [团队分工](#team)）  
2. 文稿必须符合 [写作风格](#writing-style)、[`docs/style-guide.md`](docs/style-guide.md) 与单讲模板  
3. 合并前由 **乙然** 做结构/术语校验；**志凯** 审核代码可复现性  
4. 进度每周三更新 [`meta/status.md`](meta/status.md)

---

## License

MIT — 书稿与代码以教程传播与教学为目的开放协作；与 [Embodied-Guide](https://github.com/Xbotics-Embodied-AI-club/Xbotics-Embodied-Guide) 共用配图时请保留社区出处说明。
