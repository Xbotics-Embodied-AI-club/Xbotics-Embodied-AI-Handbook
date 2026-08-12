# -*- coding: utf-8 -*-
"""Update indexes/titles after L17-L19 renumber."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(".")


def replace_file(path: str, replacements: list[tuple[str, str]]) -> None:
    p = ROOT / path
    text = p.read_text(encoding="utf-8")
    orig = text
    for a, b in replacements:
        text = text.replace(a, b)
    if text != orig:
        p.write_text(text, encoding="utf-8")
        print(f"updated {path}")
    else:
        print(f"no change {path}")


# SUMMARY
Path("docs/SUMMARY.md").write_text(
    """# 全书目录（21 讲 · 六大部分）

> **课程规模**：共 **21 讲**。Part 5 为 VLN（L17–L18），Part 6 为 Agent / 世界模型 / 前沿（L19、L20–L21）。  
> 团队分工 · 时间安排：见 [README](../README.md)  
> 写作风格（完整约定）：[`docs/style-guide.md`](style-guide.md)  
> 进度跟踪：[`meta/status.md`](../meta/status.md)（每周三 21:00 更新）

## 前言

- [总定位与学习路径](00-preface/01-positioning.md)
- [双路径设计：有硬件版与无硬件仿真版](00-preface/02-dual-path.md)
- [AI 时代的学习方法](00-preface/03-ai-era-learning.md)

---

## 第一部分：机器人系统基础（第 1–4 讲）

- [部分导言](part1-system-basics/00-part-overview.md)
- [第 1 讲：具身智能导论](part1-system-basics/01-introduction.md)
- [第 2 讲：机器人系统架构：硬件、软件与 ROS2](part1-system-basics/02-ros2-architecture.md)
- [第 3 讲：机器人本体与控制基础](part1-system-basics/03-robot-body-control.md)
- [第 4 讲：传感器与感知基础](part1-system-basics/04-sensors-coordinates.md)

---

## 第二部分：机器人视觉操作（第 5–7 讲）

- [部分导言](part2-vision-manipulation/00-part-overview.md)
- [第 5 讲：仿真环境与操作任务搭建](part2-vision-manipulation/05-simulation.md)
- [第 6 讲：机器人感知与位姿估计](part2-vision-manipulation/06-pose-estimation.md)
- [第 7 讲：机器人操作技能](part2-vision-manipulation/07-manipulation-skills.md)

---

## 第三部分：端到端机器人操作（第 8–13 讲）

- [部分导言](part3-end-to-end/00-part-overview.md)
- [第 8 讲：端到端策略导论 —— 从 Pipeline 到 Policy](part3-end-to-end/08-端到端策略导论.md)（[PDF](part3-end-to-end/pdf/第8讲_端到端策略导论.pdf)）
- [第 9 讲：操作数据闭环 —— 采一份能训练的数据集](part3-end-to-end/09-操作数据闭环.md)（[PDF](part3-end-to-end/pdf/第9讲_操作数据闭环.pdf)）
- [第 10 讲：模仿学习实战 —— ACT、Diffusion 与 Flow Matching](part3-end-to-end/10-模仿学习实战.md)（[PDF](part3-end-to-end/pdf/第10讲_模仿学习实战.pdf)）
- [第 11 讲：VLA 模型导览 —— 从 OpenVLA 到 π0 家族](part3-end-to-end/11-VLA模型导览.md)（[PDF](part3-end-to-end/pdf/第11讲_VLA模型导览.pdf)）
- [第 12 讲：VLA 微调实战 —— 从全量 SFT 到 LoRA 上真机](part3-end-to-end/12-VLA微调实战.md)（[PDF](part3-end-to-end/pdf/第12讲_VLA微调实战.pdf)）
- [第 13 讲：VLA 前沿 —— 跑得更快、记得更久、用上全身](part3-end-to-end/13-VLA前沿.md)（[PDF](part3-end-to-end/pdf/第13讲_VLA前沿.pdf)）

---

## 第四部分：强化学习（第 14–16 讲）

- [部分导言](part4-reinforcement-learning/00-part-overview.md)
- [第 14 讲：强化学习入门 —— 从策略梯度到 PPO](part4-reinforcement-learning/14-强化学习入门.md)（[PDF](part4-reinforcement-learning/pdf/第14讲_强化学习入门.pdf)）
- [第 15 讲：GRPO 后训练 —— 让 VLM 学会数数、让 VLA 自我提升](part4-reinforcement-learning/15-GRPO后训练.md)（[PDF](part4-reinforcement-learning/pdf/第15讲_GRPO后训练.pdf)）
- [第 16 讲：Off-policy 强化学习 —— 从仿真提速到真机落地](part4-reinforcement-learning/16-真机强化学习.md)（[PDF](part4-reinforcement-learning/pdf/第16讲_真机强化学习.pdf)）

---

## 第五部分：视觉语言导航 VLN（第 17–18 讲）

- [部分导言](part5-vln/00-part-overview.md)
- [第 17 讲：VLN 理论基础](part5-vln/17-vln-theory.md)
- [第 18 讲：VLN 实操与评测](part5-vln/18-vln-practice.md)

---

## 第六部分：Agent、世界模型与前沿进展（第 19、20–21 讲）

> 讲次顺序：L17–L18（Part 5 VLN）→ L19 Agent → L20 → L21

- [部分导言](part6-agent-world-model/00-part-overview.md)
- [第 19 讲：Embodied Agent](part6-agent-world-model/19-embodied-agent.md)
- [第 20 讲：世界模型 —— 预测、规划与数据飞轮](part6-agent-world-model/20-world-model.md)
- [第 21 讲：具身智能前沿进展 —— 综合闭环与答辩](part6-agent-world-model/21-frontier-progress.md)

---

## 附录

- [全书风格指南](style-guide.md)
- [硬件与仿真环境配置](appendix/hardware-setup.md)
- [教程实施原则](appendix/teaching-principles.md)
- [最终学习成果](appendix/learning-outcomes.md)
- [核心亮点](appendix/highlights.md)

---

## 资源索引

- [开源项目链接汇总](../references/links.md)
- [写作与配图进度](../meta/status.md)
""",
    encoding="utf-8",
)
print("wrote SUMMARY")

# outline.yaml lectures section - rewrite key lines via full file read
outline = Path("meta/outline.yaml").read_text(encoding="utf-8")
outline = outline.replace("lectures: [18, 19]", "lectures: [17, 18]")
outline = outline.replace("lectures: [17, 20, 21]", "lectures: [19, 20, 21]")
outline = outline.replace(
    "  - { id: 17, slug: embodied-agent, title: Embodied Agent, path: docs/part6-agent-world-model/17-embodied-agent.md, code: code/lecture17, part: 6 }\n"
    "  - { id: 18, slug: vln-theory, title: VLN 理论基础, path: docs/part5-vln/18-vln-theory.md, code: code/lecture18, part: 5 }\n"
    "  - { id: 19, slug: vln-practice, title: VLN 实操与评测, path: docs/part5-vln/19-vln-practice.md, code: code/lecture19, part: 5 }\n",
    "  - { id: 17, slug: vln-theory, title: VLN 理论基础, path: docs/part5-vln/17-vln-theory.md, code: code/lecture17, part: 5 }\n"
    "  - { id: 18, slug: vln-practice, title: VLN 实操与评测, path: docs/part5-vln/18-vln-practice.md, code: code/lecture18, part: 5 }\n"
    "  - { id: 19, slug: embodied-agent, title: Embodied Agent, path: docs/part6-agent-world-model/19-embodied-agent.md, code: code/lecture19, part: 6 }\n",
)
Path("meta/outline.yaml").write_text(outline, encoding="utf-8")
print("updated outline.yaml")

# Part overviews
p5 = Path("docs/part5-vln/00-part-overview.md").read_text(encoding="utf-8")
p5 = p5.replace("第 18–19 讲", "第 17–18 讲")
p5 = p5.replace("| 第 18 讲 |", "| 第 17 讲 |")
p5 = p5.replace("| 第 19 讲 |", "| 第 18 讲 |")
p5 = p5.replace("Part 6 第 17 讲 Agent", "Part 6 第 19 讲 Agent")
p5 = p5.replace(
    "**建议学习顺序**：可先学习 L17 Agent 基础概念，再学习 L18–L19 VLN 理论与实践；也可按照课程结构依次完成 L18→L19，掌握导航能力后进入 Part 6 的智能体综合应用。",
    "**建议学习顺序**：按编号连续学习 L17→L18 VLN，再进入 Part 6 的 L19 Agent、L20 世界模型与 L21 答辩。",
)
Path("docs/part5-vln/00-part-overview.md").write_text(p5, encoding="utf-8")
print("updated part5 overview")

p6 = Path("docs/part6-agent-world-model/00-part-overview.md").read_text(encoding="utf-8")
p6 = p6.replace("第 17、20–21 讲", "第 19、20–21 讲")
p6 = p6.replace("第五部分**（第 18–19 讲）", "第五部分**（第 17–18 讲）")
p6 = p6.replace("| 第 17 讲 | Embodied Agent | 富平（Agent） |", "| 第 19 讲 | Embodied Agent | 富平（Agent） |")
p6 = p6.replace("## 第 17 讲大纲：Embodied Agent", "## 第 19 讲大纲：Embodied Agent")
p6 = p6.replace("| 1 | L17 Agent | 先建立 skill 与规划框架 |", "| 1 | L17–L18 VLN | 补齐语言导航 skill |")
p6 = p6.replace("| 2 | L18–L19 VLN | 第五部分，补移动 skill |", "| 2 | L19 Agent | 规划与技能调用框架 |")
Path("docs/part6-agent-world-model/00-part-overview.md").write_text(p6, encoding="utf-8")
print("updated part6 overview")

# Code READMEs
Path("code/lecture17/README.md").write_text(
    """# Lecture 17 · VLN 理论

> 对应文稿：[`docs/part5-vln/17-vln-theory.md`](../../docs/part5-vln/17-vln-theory.md)

## 本讲 Demo

Habitat VLN episode 走读与指标说明
""",
    encoding="utf-8",
)
Path("code/lecture18/README.md").write_text(
    """# Lecture 18 · VLN 实操

> 对应文稿：[`docs/part5-vln/18-vln-practice.md`](../../docs/part5-vln/18-vln-practice.md)

## 本讲 Demo

VLN 批量评测与 navigate_to skill
""",
    encoding="utf-8",
)
Path("code/lecture19/README.md").write_text(
    """# Lecture 19 · Embodied Agent

> 对应文稿：[`docs/part6-agent-world-model/19-embodied-agent.md`](../../docs/part6-agent-world-model/19-embodied-agent.md)

## 本讲 Demo

LangGraph skill 调用 mock
""",
    encoding="utf-8",
)
print("updated code READMEs")

# README team section leftovers
replace_file(
    "README.md",
    [
        ("| **第五部分负责人** | **新梦** | 第 18–19 讲（VLN）及阶段项目四；雨浩协同 |", "| **第五部分负责人** | **新梦** | 第 17–18 讲（VLN）及阶段项目四；雨浩协同 |"),
        ("| **第六部分负责人** | **雨浩** | 第 17、20–21 讲及阶段项目五；富平（L17）、煜恒（L20） |", "| **第六部分负责人** | **雨浩** | 第 19、20–21 讲及阶段项目五；富平（L19）、煜恒（L20） |"),
    ],
)

replace_file(
    "CONTRIBUTING.md",
    [
        ("| 第五部分 L18–L19 | **新梦**（雨浩协同） | `docs/part5-vln/`、`code/lecture18–19` |", "| 第五部分 L17–L18 | **新梦**（雨浩协同） | `docs/part5-vln/`、`code/lecture17–18` |"),
        ("| 第六部分 L17、L20–L21 | 雨浩（富平 L17、煜恒 L20） | `docs/part6-agent-world-model/`、`code/lecture17、20–21` |", "| 第六部分 L19、L20–L21 | 雨浩（富平 L19、煜恒 L20） | `docs/part6-agent-world-model/`、`code/lecture19、20–21` |"),
    ],
)

replace_file(
    "docs/appendix/teaching-principles.md",
    [
        ("| 四 | L18–L19 | 语言驱动导航评测（VLN） |", "| 四 | L17–L18 | 语言驱动导航评测（VLN） |"),
        ("| 五 | L17、L20–L21 | 移动 + 操作综合答辩 |", "| 五 | L19、L20–L21 | 移动 + 操作综合答辩 |"),
    ],
)

replace_file(
    "docs/style-guide.md",
    [
        ("[`17-embodied-agent.md`](part6-agent-world-model/17-embodied-agent.md)", "[`19-embodied-agent.md`](part6-agent-world-model/19-embodied-agent.md)"),
    ],
)

replace_file(
    "docs/part6-agent-world-model/21-frontier-progress.md",
    [
        ("[27] Xbotics 第 17 讲《Embodied Agent》。", "[27] Xbotics 第 19 讲《Embodied Agent》。"),
    ],
)

# meta contributors / status - best effort
for path in ("meta/contributors.md", "meta/status.md"):
    if Path(path).exists():
        replace_file(
            path,
            [
                ("L18–L19", "L17–L18"),
                ("L17、L20–L21", "L19、L20–L21"),
                ("| L17 | 6 | 雨浩/富平 |", "| L19 | 6 | 雨浩/富平 |"),
                ("| L18 | 5 | 新梦 |", "| L17 | 5 | 新梦 |"),
                ("| L19 | 5 | 新梦 |", "| L18 | 5 | 新梦 |"),
                ("| L17 | Embodied Agent | 雨浩 | 富平 |", "| L19 | Embodied Agent | 雨浩 | 富平 |"),
                ("| L18 | VLN 理论 | 新梦 |", "| L17 | VLN 理论 | 新梦 |"),
                ("| L19 | VLN 实操 | 新梦 |", "| L18 | VLN 实操 | 新梦 |"),
                ("`L01…L16 → L17 Agent → L18–L19 VLN → L20 世界模型 → L21 答辩`", "`L01…L16 → L17–L18 VLN → L19 Agent → L20 世界模型 → L21 答辩`"),
                ("## 学习顺序（L17 在 Part 6，建议在 L18 前）", "## 学习顺序（编号连续）"),
                ("part6-agent-world-model/17-embodied-agent.md", "part6-agent-world-model/19-embodied-agent.md"),
                ("**六** 前沿 | L17、L20–L21", "**六** 前沿 | L19、L20–L21"),
                ("**五** VLN | L18–L19", "**五** VLN | L17–L18"),
            ],
        )

# preface positioning learning path if present
pref = Path("docs/00-preface/01-positioning.md")
if pref.exists():
    t = pref.read_text(encoding="utf-8")
    t2 = t
    t2 = t2.replace("→ Robot Agent（Part 6）→ VLN 理论 → VLN 实操（Part 5）", "→ VLN 理论 → VLN 实操（Part 5）→ Embodied Agent（Part 6）")
    t2 = t2.replace("| **Part 5** | **18–19** |", "| **Part 5** | **17–18** |")
    t2 = t2.replace("| **Part 6** | **17、20–21** |", "| **Part 6** | **19、20–21** |")
    if t2 != t:
        pref.write_text(t2, encoding="utf-8")
        print("updated preface")

print("done")
