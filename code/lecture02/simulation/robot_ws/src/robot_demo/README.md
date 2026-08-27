# robot_demo — Lecture 02：六关节机械臂 ROS2 闭环 Demo

> 对应讲义：第 2 讲「机器人系统架构：硬件、软件与 ROS2」2.5–2.8 节。

用最少模块展示一个 ROS2 闭环：**让六关节机械臂从当前关节位置移动到目标关节位置，并记录完整过程**。默认走 mock 无硬件路径。

**完整文档（架构图、Topic 契约、构建运行步骤、关键约束、运行证据、失败排查）见本讲 README：[`code/lecture02/README.md`](../../../../README.md)。**

## 构建与运行速查

```bash
source /opt/ros/humble/setup.bash
cd code/lecture02/simulation/robot_ws
colcon build --symlink-install
source install/setup.bash
ros2 launch robot_demo minimal_loop.launch.py
```

## 包内结构

| 文件 | 作用 |
| --- | --- |
| `robot_demo/common.py` | 共享常量（关节名/限位/阈值/频率），改关节数只改这一个文件 |
| `robot_demo/target_publisher.py` | 发布 `/target_joint` 六关节绝对目标 |
| `robot_demo/robot_state_node.py` | 发布 `/joint_states`，**唯一状态来源** |
| `robot_demo/policy_node.py` | 由误差生成 `/action_command` 位置增量（含 `build_action()`） |
| `robot_demo/controller_node.py` | 校验维度 + 防御性限幅，更新 mock |
| `robot_demo/task_status_node.py` | 发布 `/task_status`（running/reached/timeout/stale_state） |
| `robot_demo/episode_recorder.py` | 记录 episode 到 JSONL |
| `launch/minimal_loop.launch.py` | 一键启动全部 6 个节点 |
