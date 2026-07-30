# Lecture 02 — 机器人系统架构：硬件、软件与 ROS2

> 对应文稿：见 `docs/` 中第 2 讲

## 本讲 Demo

理解机器人软硬件组成与接口，并将状态读取、目标发布、动作生成、控制执行和 episode 记录组织成 ROS2 闭环。

## 目录结构

```
lecture02/
├── README.md           # 本文件
├── requirements.txt    # Python 依赖（按需）
├── hardware/           # 真机脚本（SO101 / xLeRobot / G1）
└── simulation/         # 无硬件可运行路径
```

## 当前代码状态

正文描述了包含六个节点的 `robot_demo` 教学 Package，但当前目录尚未提交对应 Package 和 `simulation/mock_ros2_loop.py`。因此暂时没有可直接复制运行的快速开始命令；课堂使用前需要先补齐并在指定 ROS2 环境中复验。

计划中的最小节点包括：

- `robot_state_node`
- `target_publisher`
- `policy_node`
- `controller_node`
- `task_status_node`
- `episode_recorder`

## 真机路径

见 `hardware/README.md`（待补充）。

## 状态

- [ ] 仿真 Demo 可运行
- [ ] 真机 Demo 可运行
- [ ] 与文稿实验步骤一致
- [ ] 常见失败已写入文稿

## 贡献

修改本讲代码请开 PR，标题格式：`[Lecture 02] ...`
