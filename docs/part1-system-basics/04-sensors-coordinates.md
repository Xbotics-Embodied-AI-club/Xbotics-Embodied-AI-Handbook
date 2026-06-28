# 第 4 讲：传感器与感知基础 —— 坐标系、位姿表示与多传感器数据

> **所属部分**：第一部分 · 机器人系统基础  
> **代码**：[`code/lecture04/`](../../code/lecture04/)

## 4.1 教学目标

理解传感器类型、坐标系、位姿表示、相机模型与 RGB-D 反投影，将感知结果转换为可执行目标点。

## 4.2 核心知识点

1. 传感器：RGB/RGB-D、双目、LiDAR、IMU、编码器、力矩、触觉、麦克风
2. 参数：分辨率、帧率、FOV、精度、延迟、时间戳、标定
3. 坐标系：world、map、odom、base、camera、eef、object
4. 位姿：平移、旋转矩阵、欧拉角、四元数、齐次变换
5. ROS2 TF：TF 树、static/dynamic transform
6. 相机模型：内参、畸变、外参、反投影
7. 标定：内参、外参、手眼、力传感器零点
8. 误差：深度噪声、时间不同步、外参误差、光照与材质

**反投影公式**：

```
Z = depth[u, v]
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
```

## 4.3 教学设计

```
传感器 → 坐标系 → 位姿 → TF → 内外参 → RGB-D 反投影 → base 坐标 target pose
```

## 4.4 有硬件版 Demo

SO101 / xLeRobot：点击像素 → 反投影 → camera_to_base → 机械臂接近目标。

## 4.5 无硬件仿真版 Demo

Open3D / ManiSkill / Isaac Lab 样例 RGB-D：反投影、外参修改、噪声模拟、点云可视化。

## 4.6 实验步骤

读 RGB/depth/内参 → 选像素 → 反投影 → 齐次变换到 base → Open3D 可视化 → 分析误差对抓取的影响。

## 4.7 作业交付

传感器对比表、TF 示意图、反投影与坐标转换代码、可视化截图、标定误差说明、录屏。

## 4.8 常见失败与复盘

depth 单位错误、u/v 顺序、RGB-depth 未对齐、外参方向反、矩阵乘法顺序、不可达区域等。

## 4.9 配图建议

| 编号 | 内容 |
|------|------|
| 图 4-1 | 常用传感器总览 |
| 图 4-2 | 坐标系关系 |
| 图 4-3 | 像素反投影 |
| 图 4-4 | camera → base 转换 |
| 图 4-5 | 标定误差影响抓取 |

## 4.10 参考开源项目

Open3D、pytransform3d、ROS2 geometry2、RealSense ROS、Kalibr — 见 [`references/links.md`](../../references/links.md#lecture-04)。
