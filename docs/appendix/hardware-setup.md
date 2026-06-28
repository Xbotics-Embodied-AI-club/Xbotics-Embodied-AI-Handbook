# 五、教程硬件与仿真环境配置

## 1. 标准教学班配置（20–30 人）

| 设备 | 数量 | 用途 |
|------|------|------|
| SO101 教学机械臂 | 6–8 套 | 分组实操 |
| xLeRobot 移动操作 | 2–4 套 | 移动操作综合项目 |
| Imeta-Y1 | 1 套 | 高阶演示 |
| Unitree G1 | 1 台 | 人形演示、仿真 RL、Agent/世界模型 |
| GPU 工作站 | 1–2 台 | 训练 |

## 2. 演示型配置

SO101：2–4 套；xLeRobot：1–2 套；Imeta：1 套；G1：1 台。以教师演示为主。

## 3. 实训型配置（5–10 天营）

SO101：8–12 套；xLeRobot：4–6 套；Imeta：2–3 套；G1：1–2 台；GPU：2–4 台。

## 4. 无硬件最低配置

- Ubuntu 22.04，Python 3.10+，ROS2 Humble
- NVIDIA GPU 8GB+（推荐），内存 16GB+（推荐 32GB），磁盘 100GB+

**轻量仿真**：ROS2 examples、TurtleBot3、Open3D、ManiSkill、robosuite、LeRobot、robomimic、Stable-Baselines3

**高阶仿真**：Isaac Lab、MuJoCo、Unitree RL 系列、OpenVLA / OpenPI / Octo、DreamerV3
