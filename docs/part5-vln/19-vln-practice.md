# 第 19 讲：视觉语言导航 VLN —— 实操与评测

> **所属部分**：第五部分 · 视觉语言导航 VLN  
> **Part 负责人**：新梦（协同雨浩；xLeRobot 支持：昊旺 / 志凯）  
> **代码**：[`code/lecture19/`](../../code/lecture19/)

## 19.1 本讲目标

在第 18 讲理论基础上，完成一次 **可复现的 VLN 仿真评测**，并理解如何将 VLN 模块接入 xLeRobot / Agent skill 接口。

本讲结束后，学生应能够回答：

1. 如何固定 seed 评测 VLN 成功率？
2. rollout 轨迹中如何判断 agent「走错了」还是「停错了」？
3. VLN 模块如何封装成 `navigate_to(instruction)` skill？

本讲结束后，学生应能够完成：

- 在 Habitat（或课程 mock 环境）完成 ≥10 次 episode 评测表
- 编写最小 `navigate_to` skill 接口（仿真 mock 即可）

## 19.2 核心知识点

1. **环境配置**：Habitat-Sim 版本、场景 mesh、episode 划分、action space 配置
2. **Baseline 类型**：Random、Forward-Only、Shortest-Path 上界、学习策略 checkpoint
3. **评测协议**：固定 seed / 多 seed、train-val-test 划分、SPL / success / collision
4. **数据记录**：instruction、RGB 序列、action 序列、pose、success、path length
5. **与 Agent 衔接**：VLN 输出子目标或停止信号 → 触发 arm / manipulation skill
6. **xLeRobot 真机路径**：地图/视觉导航 + 语言指令 + 到达判定（低速、安全边界）
7. **Sim2Real**：光照、纹理、动态障碍、相机高度与训练差异

## 19.3 课堂任务 / 引入案例

**任务**：在 VLN-CE 或课程简化环境中，对指令 *「走到沙发旁边的桌子前」* 运行 20 次 rollout，填写成功率与 SPL，并挑选 2 个失败 case 做归因。

## 19.4 方法框架

**VLN 实操评测四步**：

1. **固定任务集**：同一批 instruction + 初始 pose  
2. **跑 baseline / 策略**：记录完整 trajectory  
3. **算指标**：success、SPL、collision、平均步数  
4. **看失败视频**：分类为理解 / grounding / 控制 / 停止  

## 19.5 有硬件版 Demo

**Demo 名称**：xLeRobot 语言导航 + 到达判定

**流程**：

```
parse_instruction → navigate_to(target_region) → check_arrival → log episode
```

**安全**：限速、障碍检测、急停；本讲不要求长距离复杂导航，以 **可重复的到达判定** 为主。

## 19.6 无硬件仿真版 Demo

**Demo 名称**：Habitat VLN baseline 批量评测

**流程**：加载配置 → 固定 seed 运行 N episodes → 导出 CSV → 选失败 case 截图/短视频

**可选**：LangGraph mock Agent 调用 `navigate_to` skill，VLN 后端用 Habitat rollout 模拟

## 19.7 实验步骤

1. 复现第 18 讲 Habitat 最小环境
2. 编写 `run_vln_eval.py`：固定 seed、N 次 episode
3. 输出 success / SPL / collision 表
4. 实现 `skills/navigate_to.py` 接口（输入 instruction，输出 done + pose）
5. 与第 17 讲 Agent 计划联调（mock 即可）
6. 整理 2 个失败 case 与改进建议

## 19.8 作业交付

1. 评测脚本 + 20 次结果表
2. SPL / success 简要分析
3. `navigate_to` skill 接口定义与调用示例
4. 失败 case 归因（各 1 段）
5. 可选：xLeRobot 或仿真录屏

## 19.9 常见失败与复盘

- episode 初始 pose 未固定，结果不可比
- 只报 success 不算 SPL
- 碰撞仍判 success
- skill 接口缺少 `check_arrival` 状态

## 19.10 参考开源项目

Habitat-Lab、VLN-CE、AllenAct — 见 [`references/links.md`](../../references/links.md#lecture-19)。
