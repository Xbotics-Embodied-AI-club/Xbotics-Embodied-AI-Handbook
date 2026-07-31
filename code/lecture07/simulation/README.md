# Lecture 07 — 无硬件仿真路径

默认使用 `MockBackend`：不依赖机器人、仿真器或第三方库，用于验证位姿生成、状态机、检测、恢复与日志。

```bash
cd code/lecture07
pip install -e .
python simulation/pick_place_fsm.py --task cube
python simulation/pick_place_fsm.py --task bottle
```

MuJoCo 资产场景见 `simulation/mujoco_tasks/`，场景在 `envs/scene.py` 中程序化构建，任务物体 mesh 位于 `simulation/assets/models/`。
入口是 `simulation/mujoco_tasks/try_ik.py`。

```bash
pip install -e ".[mujoco]"
python simulation/mujoco_tasks/try_ik.py --task cube
python simulation/mujoco_tasks/try_ik.py --task bottle
python simulation/mujoco_tasks/try_ik.py --task cube --show-grasp
```

`robot_pick_place/backends/mujoco_adapter.py` 负责把外部 MuJoCo
控制器接到本讲状态机；`try_ik.py` 用于验证模型、碰撞体、任务布局与 IK 遥操作。

GraspNet 风格 6D 抓取位姿的 MuJoCo 3D 可视化（平行夹爪四块 box + 坐标轴）见 `viz/grasp_viz.py`，
在 `try_ik.py` 中通过 `--show-grasp` 启用。
