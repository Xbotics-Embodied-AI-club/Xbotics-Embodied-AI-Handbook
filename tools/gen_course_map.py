# -*- coding: utf-8 -*-
"""Generate README course-map fragment with per-lecture sub-TOCs."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sections_for(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    chapters: list[str] = []
    for line in text.splitlines():
        m = re.match(r"^# (\d+ .+)$", line)
        if m:
            t = m.group(1)
            if t.startswith("参考"):
                continue
            chapters.append(t)
    if len(chapters) >= 3:
        return chapters

    items: list[str] = []
    for line in text.splitlines():
        if not line.startswith("## "):
            continue
        t = line[3:].strip().replace("**", "")
        if t.startswith("参考") or t in {"术语表", "关联代码", "配图"}:
            continue
        if t in {"本讲提要", "本讲目标", "学习目标"} or re.match(r"^\d+\.\d+", t) or re.match(
            r"^\d+\s", t
        ):
            items.append(t)
    if len(items) > 12:
        slim = [
            t
            for t in items
            if t in {"本讲提要", "本讲目标", "学习目标"}
            or re.match(r"^\d+\.1\b", t)
            or re.match(r"^\d+\s", t)
        ]
        if len(slim) >= 4:
            return slim
    # dedupe
    out, seen = [], set()
    for s in items:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out[:12]


LECTURES = [
    (1, "具身智能导论", "docs/part1-system-basics/01-introduction.md", "code/lecture01/"),
    (2, "机器人系统架构（硬件 / 软件 / ROS2）", "docs/part1-system-basics/02-ros2-architecture.md", "code/lecture02/"),
    (3, "机器人本体与控制基础", "docs/part1-system-basics/03-robot-body-control.md", "code/lecture03/"),
    (4, "传感器、坐标系与相机模型", "docs/part1-system-basics/04-sensors-coordinates.md", "code/lecture04/"),
    (5, "仿真环境与操作任务搭建", "docs/part2-vision-manipulation/05-simulation.md", "code/lecture05/"),
    (6, "机器人感知与位姿估计", "docs/part2-vision-manipulation/06-pose-estimation.md", "code/lecture06/"),
    (7, "操作技能：抓取、放置与失败恢复", "docs/part2-vision-manipulation/07-manipulation-skills.md", "code/lecture07/"),
    (8, "端到端策略导论：从 Pipeline 到 Policy", "docs/part3-end-to-end/08-端到端策略导论.md", "code/vla/1_policy_rollout/"),
    (9, "操作数据闭环：采一份能训练的数据集", "docs/part3-end-to-end/09-操作数据闭环.md", "code/vla/2_data_collection/"),
    (10, "模仿学习实战：ACT、Diffusion 与 Flow Matching", "docs/part3-end-to-end/10-模仿学习实战.md", "code/vla/3_imitation_learning/"),
    (11, "VLA 模型导览：从 OpenVLA 到 π0 家族", "docs/part3-end-to-end/11-VLA模型导览.md", "code/vla/4_vla_inference/"),
    (12, "VLA 微调实战：从全量 SFT 到 LoRA 上真机", "docs/part3-end-to-end/12-VLA微调实战.md", "code/vla/5_vla_finetune/"),
    (13, "VLA 前沿：跑得更快、记得更久、用上全身", "docs/part3-end-to-end/13-VLA前沿.md", None),
    (14, "强化学习入门：从策略梯度到 PPO", "docs/part4-reinforcement-learning/14-强化学习入门.md", "code/rl/1_rl_basics/"),
    (15, "GRPO 后训练：让 VLM 学会数数、让 VLA 自我提升", "docs/part4-reinforcement-learning/15-GRPO后训练.md", "code/rl/2_grpo_posttraining/"),
    (16, "Off-policy 强化学习：从仿真提速到真机落地", "docs/part4-reinforcement-learning/16-真机强化学习.md", "code/rl/3_offpolicy/"),
    (17, "视觉语言导航：任务与基准", "docs/part5-vln/17-vln-theory.md", "code/lecture17/"),
    (18, "视觉语言导航：方法与实践", "docs/part5-vln/18-vln-practice.md", "code/lecture18/"),
    (19, "Embodied Agent", "docs/part6-agent-world-model/19-embodied-agent.md", "code/lecture19/"),
    (20, "世界模型与数据飞轮", "docs/part6-agent-world-model/20-world-model.md", "code/lecture20/"),
    (21, "前沿进展与综合答辩", "docs/part6-agent-world-model/21-frontier-progress.md", "code/lecture21/"),
]

PARTS = [
    (1, 4, "第一部分 · 机器人系统基础", "机器人基础控制闭环", "docs/part1-system-basics/00-part-overview.md"),
    (5, 7, "第二部分 · 机器人视觉操作", "视觉目标定位与模块化抓取", "docs/part2-vision-manipulation/00-part-overview.md"),
    (8, 13, "第三部分 · 端到端机器人操作", "端到端操作策略学习", "docs/part3-end-to-end/00-part-overview.md"),
    (14, 16, "第四部分 · 强化学习", None, "docs/part4-reinforcement-learning/00-part-overview.md"),
    (17, 18, "第五部分 · 视觉语言导航 VLN", "语言驱动导航评测", "docs/part5-vln/00-part-overview.md"),
    (19, 21, "第六部分 · Agent、世界模型与前沿进展", "移动 + 操作综合答辩", "docs/part6-agent-world-model/00-part-overview.md"),
]

PDF = {
    8: "docs/part3-end-to-end/pdf/第8讲_端到端策略导论.pdf",
    9: "docs/part3-end-to-end/pdf/第9讲_操作数据闭环.pdf",
    10: "docs/part3-end-to-end/pdf/第10讲_模仿学习实战.pdf",
    11: "docs/part3-end-to-end/pdf/第11讲_VLA模型导览.pdf",
    12: "docs/part3-end-to-end/pdf/第12讲_VLA微调实战.pdf",
    13: "docs/part3-end-to-end/pdf/第13讲_VLA前沿.pdf",
    14: "docs/part4-reinforcement-learning/pdf/第14讲_强化学习入门.pdf",
    15: "docs/part4-reinforcement-learning/pdf/第15讲_GRPO后训练.pdf",
    16: "docs/part4-reinforcement-learning/pdf/第16讲_真机强化学习.pdf",
}


def main() -> None:
    by_id = {n: (title, doc, code) for n, title, doc, code in LECTURES}
    lines: list[str] = []
    lines += [
        '<span id="course-map"></span>',
        "",
        "## 课程地图",
        "",
        "编号已连续：**L1–L16 → L17–L18 VLN → L19 Agent → L20 世界模型 → L21 答辩**。",
        "每讲可展开子目录；点击标题进入文稿。",
        "",
        "### 前言",
        "",
        "| 模块 | 说明 | 阅读 |",
        "|------|------|------|",
        "| 总定位与学习路径 | 课程要解决什么、21 讲主线 | [📖 阅读](docs/00-preface/01-positioning.md) |",
        "| 双路径设计 | 有硬件版 / 无硬件仿真版 | [📖 阅读](docs/00-preface/02-dual-path.md) |",
        "| AI 时代学习方法 | 问题驱动、云平台、协作写作 | [📖 阅读](docs/00-preface/03-ai-era-learning.md) |",
        "",
    ]

    for a, b, pname, proj, overview in PARTS:
        lines.append(f"### {pname}（第 {a}–{b} 讲）")
        lines.append("")
        if proj:
            lines.append(f"**阶段项目**：{proj} · [部分导言]({overview})")
        else:
            lines.append(f"[部分导言]({overview})")
        lines.append("")
        for n in range(a, b + 1):
            title, doc, code = by_id[n]
            links = [f"[📖 文稿]({doc})"]
            if n in PDF:
                links.append(f"[📄 PDF]({PDF[n]})")
            if code:
                links.append(f"[💻 代码]({code})")
            if n == 20:
                links.append("[Fast-WAM](https://github.com/yuantianyuan01/FastWAM)")
            lines.append(f"#### 第 {n} 讲 · {title}")
            lines.append("")
            lines.append(" · ".join(links))
            lines.append("")
            secs = sections_for(ROOT / doc)
            if secs:
                lines.append("<details>")
                lines.append(f"<summary>本讲子目录（{len(secs)}）</summary>")
                lines.append("")
                for s in secs:
                    lines.append(f"- {s}")
                lines.append("")
                lines.append("</details>")
                lines.append("")
        lines.append("")

    lines += [
        "### 附录",
        "",
        "| 模块 | 链接 |",
        "|------|------|",
        "| 全书风格指南 | [📖](docs/style-guide.md) |",
        "| 硬件与仿真环境配置 | [📖](docs/appendix/hardware-setup.md) |",
        "| 教程实施原则 | [📖](docs/appendix/teaching-principles.md) |",
        "| 最终学习成果 | [📖](docs/appendix/learning-outcomes.md) |",
        "| 核心亮点 | [📖](docs/appendix/highlights.md) |",
        "",
        "**教程主线（一览）：**",
        "",
        "```",
        "L1 导论 → L2 架构/ROS2 → L3 本体控制 → L4 传感器坐标",
        "→ L5 仿真 → L6 位姿 → L7 操作技能",
        "→ L8 端到端导论 → L9 数据 → L10 模仿学习 → L11–13 VLA",
        "→ L14–16 强化学习 → L17–18 VLN → L19 Agent → L20 世界模型 → L21 答辩",
        "```",
        "",
        "## 五个阶段项目",
        "",
        "| 阶段 | 讲次 | 项目名称 |",
        "|------|------|----------|",
        "| 一 | 第 1–4 讲 | 机器人基础控制闭环 |",
        "| 二 | 第 5–7 讲 | 视觉目标定位与模块化抓取 |",
        "| 三 | 第 8–13 讲 | 端到端操作策略学习 |",
        "| 四 | 第 17–18 讲 | 语言驱动导航评测（VLN） |",
        "| 五 | 第 19、20–21 讲 | 移动 + 操作综合答辩 |",
        "",
    ]

    out = ROOT / "docs/_course_map_fragment.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
