# Lecture 07 — 操作技能：基于位姿的抓取、放置与失败恢复

> 对应文稿：[`docs/part2-vision-manipulation/07-manipulation-skills.md`](../../docs/part2-vision-manipulation/07-manipulation-skills.md)

## 本讲 Demo

规则版 pick-place 状态机（方块 A→B + 瓶子入盒），统一后端接口：

| 后端 | 路径 | 说明 |
|------|------|------|
| `MockBackend` | `simulation/`（默认） | 无外部依赖，必跑 |
| `SO101Backend` | `hardware/` | 真机适配模板 |
| `MuJoCoBackend` | `simulation/` | 完整 MuJoCo pick-place 仿真（cube + bottle） |

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
└── simulation/                # Mock 入口与 MuJoCo 仿真
    ├── mujoco_pick_place.py   # MuJoCo 完整 pick-place 状态机入口
    ├── pick_place_fsm.py      # Mock 后端入口（无硬件）
    ├── assets/models/         # 任务物体 mesh（box / bottle）
    └── mujoco_tasks/          # 场景 / IK / FSM 后端 / 位姿可视化
        ├── fsm_backend.py     # MuJoCo FSM 后端（规划 + 增量 IK）
        ├── pose_targets.py    # 场景级任务配置与 5 个动作位姿
        ├── try_ik.py          # 键盘 IK 遥操作
        └── viz/pose_viz.py    # 5 位姿 + GraspNet 可视化
```

## 快速开始（仿真 / Mock，无硬件必做）

> 要求 Python >= 3.11。单元测试与 MuJoCo 仿真依赖 `mujoco`、`numpy`（见 `requirements.txt`）。

```bash
cd code/lecture07
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate
pip install -e .
pip install -r requirements.txt
python simulation/pick_place_fsm.py --task cube
python simulation/pick_place_fsm.py --task bottle
python analyze_runs.py
python -m examples.inspect_targets
python -m examples.inject_failure
python -m unittest discover -s tests -v
```

## MuJoCo 资产场景（可选）

场景定义在 `simulation/mujoco_tasks/envs/scene.py`，加载
`code/platform/so101_sim/.../so101.urdf`，并在白色桌面上绘制 8 cm × 8 cm 的红色 A 区与蓝色 B 区方框。

```bash
# 完整 pick-place 状态机：cube（90° 翻转双指夹取）/ bottle（水平径向抓取入盒）
python simulation/mujoco_pick_place.py --task cube
python simulation/mujoco_pick_place.py --task bottle
# 场景中绘制 5 个动作位姿（pre_grasp/grasp/lift/place/retreat）
python simulation/mujoco_pick_place.py --task bottle --show-poses
# 无窗口离屏运行并录制视频（另需 imageio[pyav] / imageio[ffmpeg]，见 requirements.txt）
python simulation/mujoco_pick_place.py --task bottle --viewer null --video runs/bottle.mp4
```

FSM 运行闭环：移动到 pre_grasp → 接近 → 双指闭合抓取 → 抬升 → 转运 → 放置 → retreat → 回 home，
失败自动恢复（`RECOVER`），状态与事件写入 `runs/` 下的 `events.jsonl`。
使用 CPU 仿真与 MuJoCo 原生 viewer。Windows、Linux 和 macOS 均可运行；
无显示器环境可设置 `MUJOCO_GL=egl` 使用离屏渲染。

运行结果写入 `runs/`：每次任务一个目录，`events.jsonl` 记录状态进入、检测、失败码与重试；`analyze_runs.py` 汇总为 `runs/summary.csv`。

键盘 IK 遥操作入口（调试模型 / IK / 碰撞体用）：

```bash
python simulation/mujoco_tasks/try_ik.py --task cube --show-grasp
```

## 真机路径（SO-101）

见 [`hardware/README.md`](hardware/README.md)。`SO101Backend` 是接入模板，需自行注入规划、末端位姿读取与物体检测函数；**示例坐标不可直接用于真机**。

## 状态

- [x] 仿真 Demo 可运行（Mock）
- [x] MuJoCo 仿真 Demo 可运行（cube 翻转夹取 + bottle 水平抓取入盒）
- [ ] 真机 Demo 可运行（适配模板已提供，需本机标定）
- [x] 与文稿实验步骤一致
- [x] 常见失败已写入文稿

## 贡献

修改本讲代码请开 PR，标题格式：`[Part 2 / L07] ...`
