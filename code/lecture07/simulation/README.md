# Lecture 07 — 无硬件仿真路径

默认使用 `MockBackend`：不依赖机器人、仿真器或第三方库，用于验证位姿生成、状态机、检测、恢复与日志。

```bash
cd code/lecture07
pip install -e .
python simulation/pick_place_fsm.py --task cube
python simulation/pick_place_fsm.py --task bottle
```

ManiSkill 适配器见 `robot_pick_place/backends/maniskill_adapter.py`（需自行安装 ManiSkill 并注入环境相关函数）。
