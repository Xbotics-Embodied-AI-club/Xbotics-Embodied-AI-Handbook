# 第 2 讲：机器人系统架构 —— ROS2、LeRobot 与数据闭环

> **所属部分**：第一部分 · 机器人系统基础  
> **代码**：[`code/lecture02/`](../../code/lecture02/)

## 2.1 教学目标

理解机器人系统是持续通信、反馈、记录数据的模块集合；理解 ROS2、LeRobot 与数据闭环在具身智能中的位置。

## 2.2 核心知识点

1. **四层结构**：本体层 / 控制层 / 感知层 / 智能层
2. **前沿研究位置**：VLM→感知层；VLA/ACT/DP→策略层；RL→运控；世界模型→预测与规划；数据飞轮贯穿全程
3. **ROS2**：Node、Topic、Service、Action、Launch
4. **LeRobot 数据流**：camera、joint state、eef pose、language、action、gripper、task feedback、episode
5. **数据闭环**：成功/失败记录 → 标注 → 回流训练 → 再部署

## 2.3 教学设计

```
本体 → ROS2 驱动 → 状态读取 → 感知 → 策略 → 控制器 → 执行 → 反馈 → episode 记录 → 训练与再部署
```

## 2.4 有硬件版 Demo

**Demo**：基于 ROS2 的 LeRobot / SO101 状态读取与动作下发闭环

```
连接机械臂 → 启动驱动 → 读 joint_states → 发布 target_eef/target_joint
→ 生成 action_command → 执行 → 读反馈 → 保存 episode_0001
```

**Topic 示例**：`/joint_states` `/target_pose` `/action_command` `/gripper_command` `/task_status`

## 2.5 无硬件仿真版 Demo

**节点**：camera_node、robot_state_node、target_publisher、policy_node、controller_node、task_status_node、episode_recorder

## 2.6 实验步骤

1. 创建 ROS2 workspace 与 demo package
2. 编写各节点并 `rqt_graph` 查看连接
3. 修改目标点与控制参数，观察闭环稳定性

## 2.7 作业交付

1. ROS2 package 源码
2. rqt_graph 截图
3. Topic/Service/Action 说明
4. LeRobot 数据流图
5. 闭环运行视频
6. episode 数据样例
7. 说明：VLA / 世界模型 / 控制器在系统中的位置

## 2.8 常见失败与复盘

节点未启动、控制频率低、坐标系错误、只记录动作未记录观测、策略无安全限制等。

## 2.9 参考开源项目

LeRobot、lerobot-ros、so101_ros2、so101-ros-physical-ai — 见 [`references/links.md`](../../references/links.md#lecture-02)。
