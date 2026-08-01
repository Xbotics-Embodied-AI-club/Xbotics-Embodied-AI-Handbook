# MuJoCo Tasks — Lecture 07

SO-101 机械臂在 MuJoCo 中的任务场景、差分 IK 运动、双指 pick-place 状态机，以及 GraspNet 风格的 6D 抓取可视化。

## 快速开始

在 `code/lecture07` 目录下安装 MuJoCo 依赖并启动：

```bash
pip install -e ".[mujoco]"
python simulation/mujoco_pick_place.py --task cube
python simulation/mujoco_pick_place.py --task bottle
```

常用参数：

| 参数 | 说明 |
|------|------|
| `--task {cube,bottle}` | cube 为 90° 翻转双指夹取；bottle 为水平径向抓取放入盒子 |
| `--viewer null` | 无窗口离屏运行 |
| `--show-poses` | 场景中绘制 5 个动作位姿（pre_grasp/grasp/lift/place/retreat） |
| `--show-grasp` / `--no-show-grasp` | 夹爪末端 GraspNet 可视化（默认开） |
| `--grasp-roll` | cube 夹爪额外 roll（默认 90°） |
| `--video runs/x.mp4` | 录制运行视频 |

状态机逻辑在 `fsm_backend.py`：关节空间规划 + 增量 IK 兜底，双指接触后 sticky 附着，
放置时保持物体高度护栏，失败自动恢复，任务结束回到 home。5 个动作位姿配置在 `pose_targets.py`。

## 键盘 IK 遥操作

`try_ik.py` 用于调试模型、碰撞体与任务布局（差分 IK + 键盘控制，不跑状态机）：

```bash
python simulation/mujoco_tasks/try_ik.py --task cube --show-grasp
```

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
├── fsm_backend.py         # MuJoCo FSM 后端（pick-place 闭环）
├── pose_targets.py        # 场景级任务配置与 5 个动作位姿
├── try_ik.py              # 键盘 IK 遥操作入口
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
    ├── pose_viz.py        # 5 位姿 + GraspNet 夹爪 + RGB 坐标轴
    ├── grasp_viz.py       # GraspNet 风格 3D 抓取可视化（基础组件）
    └── scene_viz.py       # 场景相机配置
```

任务物体的 mesh 与贴图位于 `simulation/assets/models/`（`box.obj`、`bottle.obj` 等）。场景在 `envs/scene.py` 中通过 MjSpec API 程序化构建，不依赖单独的 XML 场景文件。

## 位姿可视化

`--show-poses` 时在场景中绘制 5 个动作位姿：每个位姿一个 GraspNet 风格平行夹爪
（四块 box）+ RGB 坐标轴（X=红、Y=绿、Z=蓝），运行到的位姿不透明高亮，其余透明。
`--show-grasp` 在夹爪末端实时显示 GraspNet 抓取表示。坐标系约定：

- 抓取旋转矩阵与腕部 roll 坐标系一致
- **Z（蓝）** = 腕部 roll 轴；**X、Z** 张成夹取平面，**Y** 为平面法向
- 接近物体方向为 **-Z**

## 简化抓取：接触粘住（Sticky Grasp）

双指**真实闭合接触**抓取：左右夹爪碰撞体同时接触任务物体后，物体运动学绑定到末端随夹爪移动，夹爪张开即释放。逻辑见 `motion/sticky_grasp.py`，默认开启（`--no-sticky-grasp` 可关闭）。

## try_ik.py 参数

| 参数 | 说明 |
|------|------|
| `--task {cube,bottle}` | 任务场景（默认 `cube`） |
| `--show-grasp` | 显示演示抓取可视化 |
| `--show-collision` | 半透明显示碰撞几何体 |
| `--position-only` | 仅位置 IK，固定 home 姿态 |
| `--verbose` | 打印 IK 目标与误差遥测 |
| `--no-sticky-grasp` | 关闭接触粘住辅助 |
