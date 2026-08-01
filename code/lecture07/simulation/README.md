# Lecture 07 — 无硬件仿真路径

两个入口：

- `pick_place_fsm.py` — `MockBackend`，不依赖机器人、仿真器或第三方库，验证位姿生成、状态机、检测、恢复与日志。
- `mujoco_pick_place.py` — 完整 MuJoCo pick-place：SO-101 机械臂真实运动学，双指闭合抓取、
  5 个动作位姿（pre_grasp / grasp / lift / place / retreat）、失败恢复、任务完成回 home。

```bash
cd code/lecture07
pip install -e .
python simulation/pick_place_fsm.py --task cube

pip install -e ".[mujoco]"
python simulation/mujoco_pick_place.py --task cube
python simulation/mujoco_pick_place.py --task bottle --show-poses
python simulation/mujoco_pick_place.py --task bottle --viewer null --video runs/bottle.mp4
```

MuJoCo 场景在 `mujoco_tasks/envs/scene.py` 中程序化构建，任务物体 mesh 位于 `assets/models/`。
任务逻辑位于 `mujoco_tasks/`：

- `fsm_backend.py` — MuJoCo FSM 后端：关节空间规划 + 增量 IK 兜底、双指接触后附着、放置高度护栏、失败恢复。
- `pose_targets.py` — 场景级任务配置与 5 个动作位姿生成。
- `try_ik.py` — 键盘 IK 遥操作，用于调试模型、碰撞体与任务布局。
- `viz/pose_viz.py` — 5 位姿可视化：GraspNet 风格夹爪 + RGB 坐标轴，运行到的位姿高亮、其余透明。
