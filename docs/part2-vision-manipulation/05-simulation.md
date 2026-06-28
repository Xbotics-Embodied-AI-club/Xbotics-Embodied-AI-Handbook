# 第 5 讲：仿真环境与操作任务搭建 —— MuJoCo 与 Isaac Lab 入门

> **所属部分**：第二部分 · 机器人视觉操作  
> **代码**：[`code/lecture05/`](../../code/lecture05/)

## 5.1 教学目标

理解仿真在训练、测试、评测与真机前验证中的作用；掌握 MuJoCo 与 Isaac Lab 基础；理解 Sim2Real gap。

## 5.2 核心知识点

1. 仿真价值：降成本、批量数据、复现失败、RL 训练、安全验证
2. 环境组成：robot、scene、object、camera、physics、controller、task、reward
3. MuJoCo：MJCF、轻量、适合控制验证
4. Isaac Lab：GPU 并行、人形/大规模 RL、传感器仿真
5. 典型任务：reach、pick-place、push、locomotion、G1 RL
6. Sim2Real：接触、摩擦、电机、噪声、频率、延迟、domain randomization

## 5.3–5.8 Demo 与实验

- **MuJoCo**：Menagerie 模型、reach/pick-place、改摩擦/质量
- **Isaac Lab**：机械臂任务 + G1 locomotion（observation/action/reward）
- **有硬件对照**：SO101 真机 reach vs 仿真 reach 轨迹对比

## 5.9 作业交付

MuJoCo/Isaac 截图、reach 或 pick-place 录屏、G1 locomotion 录屏、平台对比表、参数修改记录、Sim2Real 说明。

## 5.10 常见失败

安装/版本/CUDA 冲突、穿模、控制频率不合理、仿真成功真机失败等。

## 5.11 配图建议

仿真组成图、MuJoCo vs Isaac 对比、reach 任务示意、G1 locomotion、Sim2Real 来源图。

## 5.12 参考开源项目

MuJoCo、Isaac Lab、Unitree RL 系列、ManiSkill、robosuite、RoboCasa — 见 [`references/links.md`](../../references/links.md#lecture-05)。
