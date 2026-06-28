#!/usr/bin/env python3
"""Generate lecture markdown files from bundled manuscript sections."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

# Part overviews (short)
PART_OVERVIEWS = {
    "part1-system-basics/00-part-overview.md": """# 第一部分：机器人系统基础

## 部分定位

第一部分解决「机器人系统是什么、机器人怎么动起来」的问题。

很多具身智能教程一上来就讲 VLA、大模型、Diffusion Policy，但学生并不知道机器人动作是如何被执行的，也不理解相机、控制器、状态反馈、坐标系之间的关系。

**第一部分任务**：建立具身智能系统底座，让学生明白：

- 机器人不是一个模型
- 机器人是一个感知—决策—控制—反馈闭环系统

## 覆盖讲次

- 第 1 讲：具身智能导论
- 第 2 讲：机器人系统架构
- 第 3 讲：机器人本体与动作空间
- 第 4 讲：坐标系、位姿表示与相机模型

## 阶段项目：机器人基础控制闭环

**目标**：给定目标点，让机器人或仿真机器人完成状态读取、动作生成和目标接近。

| 路径 | 流程 |
|------|------|
| 有硬件 | SO101 读状态 → 控制关节/末端/夹爪 → 相机坐标转换 → 目标到达 |
| 无硬件 | 仿真机械臂 → 目标点 → 状态反馈 → 简单控制器 → Open3D 可视化 |

**交付**：系统模块图、控制代码、轨迹图、坐标转换代码、实验报告
""",
    "part2-vision-manipulation/00-part-overview.md": """# 第二部分：机器人视觉操作

## 部分定位

第二部分解决「机器人如何看见目标、如何在环境中完成操作任务」。

重点不是端到端学习，而是**模块化机器人操作流程**。只有先理解传统抓取、放置、成功检测和失败处理，才能理解端到端策略在替代什么、增强什么。

## 覆盖讲次

- 第 5 讲：仿真环境与操作任务搭建
- 第 6 讲：机器人感知与位姿估计
- 第 7 讲：机器人操作技能

## 阶段项目：视觉目标定位与模块化抓取

**目标**：从图像找到目标，生成 target pose，完成规则版 pick-place，记录数据并评测。

**交付**：检测结果、target pose 代码、演示视频、episode 数据、20 次评测报告、失败样本分析表
""",
    "part3-end-to-end/00-part-overview.md": """# 第三部分：端到端机器人操作

## 部分定位

第三部分解决「机器人如何从数据中学习动作」。

前两部分已理解模块化系统如何运行；第三部分讲端到端策略学习：让模型从 observation 直接预测 action。

## 覆盖讲次

- 第 8 讲：端到端策略学习导论
- 第 9 讲：操作数据采集、任务设计与评测
- 第 10 讲：模仿学习 BC / ACT
- 第 11 讲：Diffusion Policy
- 第 12 讲：VLA 理论
- 第 13 讲：VLA 实操

## 阶段项目：端到端操作策略学习

**交付**：训练数据、checkpoint、训练曲线、执行视频、动作曲线、策略失败分析
""",
    "part4-reinforcement-learning/00-part-overview.md": """# 第四部分：强化学习

## 部分定位

从模仿学习 / VLA 进一步走向策略优化、运动控制和失败恢复。

强化学习的三个核心价值：

1. 在仿真中训练高频运动控制策略
2. 在已有策略基础上做后训练和性能优化
3. 通过 Recovery Policy 提升失败恢复能力

## 覆盖讲次

- 第 14 讲：强化学习基础（MDP → 机器人策略）
- 第 15 讲：强化学习运动控制（Unitree G1 Locomotion）
- 第 16 讲：机械臂抓取 RL 后训练与 Recovery Policy
""",
    "part5-vln/00-part-overview.md": """# 第五部分：视觉语言导航 VLN

见 `docs/part5-vln/00-part-overview.md`（已独立维护）。
""",
    "part6-agent-world-model/00-part-overview.md": """# 第六部分：Agent、世界模型与前沿进展

见 `docs/part6-agent-world-model/00-part-overview.md`（已独立维护）。
""",
}

APPENDIX = {
    "appendix/hardware-setup.md": """# 五、教程硬件与仿真环境配置

## 1. 标准教学班配置（20–30 人）

| 设备 | 数量 | 用途 |
|------|------|------|
| SO101 教学机械臂 | 6–8 套 | 分组实操 |
| xLeRobot 移动操作 | 2–4 套 | 移动操作综合项目 |
| Imeta-Y1 | 1 套 | 高阶演示 |
| Unitree G1 | 1 台 | 人形演示、仿真 RL、Agent/世界模型 |
| GPU 工作站 | 1–2 台 | 训练 |

## 2. 演示型配置

SO101：2–4 套；xLeRobot：1–2 套；Imeta：1 套；G1：1 台。以教师演示为主。

## 3. 实训型配置（5–10 天营）

SO101：8–12 套；xLeRobot：4–6 套；Imeta：2–3 套；G1：1–2 台；GPU：2–4 台。

## 4. 无硬件最低配置

- Ubuntu 22.04，Python 3.10+，ROS2 Humble
- NVIDIA GPU 8GB+（推荐），内存 16GB+（推荐 32GB），磁盘 100GB+

**轻量仿真**：ROS2 examples、TurtleBot3、Open3D、ManiSkill、robosuite、LeRobot、robomimic、Stable-Baselines3

**高阶仿真**：Isaac Lab、MuJoCo、Unitree RL 系列、OpenVLA / OpenPI / Octo、DreamerV3
""",
    "appendix/teaching-principles.md": """# 六、教程实施原则

## 1. 每讲教学结构

| 比例 | 内容 |
|------|------|
| 30% | 核心概念 |
| 40% | Demo 实操 |
| 30% | 调参、复盘与失败分析 |

**每讲交付**：代码、数据、可视化、实验视频、评测表、失败分析

## 2. 每四讲一个阶段项目

- 每讲都有 Demo
- 每几讲形成阶段项目
- 最后拼成完整系统

## 3. 安全原则

1. 教师确认后上电
2. 速度限制、空间边界、急停
3. 学生不得直接改低层危险参数
4. G1 真机只运行验证过的安全动作
5. RL 优先仿真训练验证
6. 真机实验保留日志和视频
7. 端到端模型部署前：仿真 + 低速测试
""",
    "appendix/learning-outcomes.md": """# 七、教程最终学习成果

完成教程后，学生应能够：

1. 理解具身智能系统整体架构
2. 使用 ROS2 / Python 控制基础机器人任务
3. 理解动作空间、坐标系和相机模型
4. 完成目标检测、分割和 target pose 生成
5. 搭建仿真 reach / pick-place / G1 locomotion Demo
6. 实现规则版 pick-place 和失败恢复
7. 采集、保存、回放 episode 数据
8. 设计机器人任务评测指标
9. 训练 BC / ACT / Diffusion Policy
10. 理解 RL、后训练和 recovery policy
11. 构造 VLA 训练样本
12. 分析动作尺度和真机部署问题
13. 设计 Robot Agent 技能调用流程
14. 理解世界模型和数据飞轮
15. 完成语言指令驱动的综合具身智能任务
""",
    "appendix/highlights.md": """# 八、核心亮点

| 亮点 | 说明 |
|------|------|
| 系统实践 | 非碎片化论文导读 |
| 真机优先、仿真兜底 | 双路径可自学 |
| 全栈覆盖 | 基础 → 世界模型 |
| 完整闭环 | 数据、训练、部署、评测、回流 |
| 多本体 | 机械臂、移动操作、人形 |
| 失败驱动 | 系统记录失败并形成数据飞轮 |
""",
}


def write(rel: str, content: str) -> None:
    path = DOCS / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n", encoding="utf-8")
    print(f"Wrote {path.relative_to(ROOT)}")


def main() -> None:
    for rel, content in PART_OVERVIEWS.items():
        write(rel, content)
    for rel, content in APPENDIX.items():
        write(rel, content)
    print("Part overviews and appendix done. Run import_lectures.py for full lecture bodies.")


if __name__ == "__main__":
    main()
