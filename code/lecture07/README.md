# Lecture 07 — 操作技能：基于位姿的抓取、放置与失败恢复

> 对应文稿：[`docs/part2-vision-manipulation/07-manipulation-skills.md`](../../docs/part2-vision-manipulation/07-manipulation-skills.md)

## 本讲 Demo

规则版 pick-place 状态机（方块 A→B + 瓶子入盒），统一后端接口：

| 后端 | 路径 | 说明 |
|------|------|------|
| `MockBackend` | `simulation/`（默认） | 无外部依赖，必跑 |
| `SO101Backend` | `hardware/` | 真机适配模板 |
| `ManiSkillBackend` | `simulation/` | 仿真适配模板 |

## 目录结构

```text
lecture07/
├── README.md
├── requirements.txt
├── pyproject.toml
├── analyze_runs.py
├── robot_pick_place/          # 核心包：位姿生成 / 状态机 / 检测 / 恢复 / 日志
├── examples/                  # inspect_targets / inject_failure
├── tests/
├── hardware/                  # 真机说明与 SO-101 入口
└── simulation/                # 无硬件入口（Mock）
```

## 快速开始（仿真 / Mock，无硬件必做）

```bash
cd code/lecture07
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate
pip install -e .
python simulation/pick_place_fsm.py --task cube
python simulation/pick_place_fsm.py --task bottle
python analyze_runs.py
python -m examples.inspect_targets
python -m examples.inject_failure
python -m unittest discover -s tests -v
```

运行结果写入 `runs/`：每次任务一个目录，`events.jsonl` 记录状态进入、检测、失败码与重试；`analyze_runs.py` 汇总为 `runs/summary.csv`。

等价入口：

```bash
python -m robot_pick_place.run_demo --task cube
```

## 真机路径（SO-101）

见 [`hardware/README.md`](hardware/README.md)。`SO101Backend` 是接入模板，需自行注入规划、末端位姿读取与物体检测函数；**示例坐标不可直接用于真机**。

## 状态

- [x] 仿真 Demo 可运行（Mock）
- [ ] 真机 Demo 可运行（适配模板已提供，需本机标定）
- [x] 与文稿实验步骤一致
- [x] 常见失败已写入文稿

## 贡献

修改本讲代码请开 PR，标题格式：`[Part 2 / L07] ...`
