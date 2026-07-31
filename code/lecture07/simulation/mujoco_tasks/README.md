# MuJoCo Tasks — Lecture 07

SO-101 机械臂在 MuJoCo 中的任务场景、差分 IK 遥操作，以及 GraspNet 风格的 6D 抓取可视化。

## 快速开始

在 `code/lecture07` 目录下安装 MuJoCo 依赖并启动：

```bash
pip install -e ".[mujoco]"
python simulation/mujoco_tasks/try_ik.py --show-grasp --task bottle
```

切换任务只需改 `--task`：

```bash
python simulation/mujoco_tasks/try_ik.py --show-grasp --task cube
python simulation/mujoco_tasks/try_ik.py --show-grasp --task bottle
```

`--show-grasp` 会在腕部坐标系上显示一条红色演示抓取（平行夹爪四块 box + RGB 坐标轴），用于对照 GraspNet 的 6D grasp 表示，并尝试用 IK 将末端移向该抓取位姿。

## 键盘操作

启动后终端会打印帮助。常用按键：

| 按键 | 功能 |
|------|------|
| W/S/A/D/E/Q 或方向键 | 平移 IK 目标（+X 朝桌面方向） |
| T/G、Y/H、U/J | 绕末端坐标轴旋转（默认启用） |
| Z / X | 夹爪张开 / 闭合 |
| R | 复位到 home 位姿 |
| Esc | 退出 |

Windows 上按住平移键会每 0.10 秒持续累积目标；可用 `--position-only` 关闭旋转 IK。

## 目录结构

```
mujoco_tasks/
├── try_ik.py              # 入口：键盘 IK 遥操作 + 可选抓取可视化
├── envs/                  # 场景与任务定义
│   ├── scene.py           # 程序化构建 MuJoCo 场景（桌面、机器人、任务物体）
│   ├── gym_env.py         # Gymnasium 环境封装
│   ├── cube.py            # 方块抓取任务
│   ├── bottle.py          # 瓶子放入盒子任务
│   └── utils/             # 执行器、碰撞体、mesh/URDF 工具
├── motion/                # 运动学与抓取
│   ├── solver.py          # SO-101 差分 IK 求解器
│   ├── grasp_pose.py      # 6D 抓取位姿数据结构
│   ├── grasp_to_gripper.py# 抓取坐标系 ↔ 末端执行器映射
│   ├── sticky_grasp.py    # 接触检测 + 粘住辅助（见下文）
│   └── control.py         # 控制辅助
└── viz/                   # 可视化
    ├── grasp_viz.py       # GraspNet 风格 3D 抓取可视化
    └── scene_viz.py       # 场景相机配置
```

任务物体的 mesh 与贴图位于 `simulation/assets/models/`（`box.obj`、`bottle.obj` 等）。场景在 `envs/scene.py` 中通过 MjSpec API 程序化构建，不依赖单独的 XML 场景文件。

## GraspNet 风格抓取可视化

`viz/grasp_viz.py` 在腕部 `so101_gripper_link` 上挂载演示抓取，随机械臂一起运动。坐标系约定：

- 抓取旋转矩阵与腕部 roll 坐标系一致
- **Z（蓝）** = 腕部 roll 轴；**X、Z** 张成夹取平面，**Y** 为平面法向
- 接近物体方向为 **-Z**

启用方式：`--show-grasp`。home 位姿下显示一条 `score=1.0` 的演示抓取，便于与 IK 目标对照。

## 简化抓取：接触粘住（Sticky Grasp）

为降低仿真复杂度，本模块**未实现**基于摩擦与接触力的完整物理抓取，而是采用 `motion/sticky_grasp.py` 中的辅助逻辑：

1. 检测左右夹爪碰撞体是否**同时**与任务物体发生接触，并满足最小穿透深度
2. 条件满足后，将物体**运动学绑定**到末端，随夹爪一起移动
3. 夹爪张开超过阈值时**自动释放**

默认开启（`--sticky-grasp`，可用 `--no-sticky-grasp` 关闭）。粘住/释放时终端会打印 `STICKY_GRASP_ATTACH` / `STICKY_GRASP_RELEASE` 事件。

这使键盘遥操作能完成“抓起—移动—放下”的完整流程，而无需调参摩擦系数或依赖稳定的接触力仿真。完整 pick-place 状态机见上级目录的 `simulation/pick_place_fsm.py`。

## 其他常用参数

| 参数 | 说明 |
|------|------|
| `--task {cube,bottle}` | 任务场景（默认 `cube`） |
| `--show-grasp` | 显示演示抓取可视化 |
| `--show-collision` | 半透明显示碰撞几何体 |
| `--position-only` | 仅位置 IK，固定 home 姿态 |
| `--verbose` | 打印 IK 目标与误差遥测 |
| `--no-sticky-grasp` | 关闭接触粘住辅助 |
