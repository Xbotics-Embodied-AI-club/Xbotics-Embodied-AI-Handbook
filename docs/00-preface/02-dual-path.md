# 二、教程双路径设计

教程第一版采用 **「真机优先、仿真兜底」** 的双路径设计。

## 2.1 有硬件版

| 平台 | 用途 |
|------|------|
| **SO101** 教学机械臂 | 基础控制、数据采集、模仿学习、小模型部署 |
| **xLeRobot** 移动操作套件 | 移动操作、目标接近、移动抓取、Agent 综合任务 |
| **Imeta-Y1** 科研机械臂 | 高阶操作、科研演示、复杂任务验证（优先级较低） |
| **Unitree G1** 人形机器人 | 本体认知、仿真 RL、站立/行走/恢复、世界模型案例 |

## 2.2 无硬件仿真版

| 真机任务 | 仿真替代 |
|----------|----------|
| SO101 机械臂 | MuJoCo / ManiSkill / robosuite / MoveIt2 |
| xLeRobot 移动操作 | TurtleBot3 + 机械臂 mock / ROS2 mock / Isaac Lab |
| Imeta-Y1 高阶任务 | 通用 6DoF 机械臂 / MoveIt2 / MuJoCo / Isaac Lab |
| Unitree G1 | unitree_rl_gym / unitree_rl_lab / unitree_sim_isaaclab |

## 2.3 每讲结构模板

1. 本讲目标  
2. 核心知识点  
3. 教学设计  
4. 有硬件版 Demo  
5. 无硬件仿真版 Demo  
6. 实验步骤  
7. 作业交付  
8. 常见失败与复盘  
9. 参考开源项目  

详见 [`templates/lecture-template.md`](../../templates/lecture-template.md)。
