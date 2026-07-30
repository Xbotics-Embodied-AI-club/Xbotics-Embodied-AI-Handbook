# Lecture 07 — 有硬件路径（SO-101）

真机适配模板：`robot_pick_place/backends/so101_adapter.py`。

## 接入前必做

1. 核对关节方向、零位与机械臂标定  
2. 限制关节速度、末端速度与工作空间  
3. 空载验证预抓取、抬升、退出位姿  
4. 保持急停可用，操作者全程监护  
5. **不得直接使用示例坐标控制真实机械臂**

## 需要注入的函数

| 函数 | 作用 |
|------|------|
| `plan_pose(target, speed)` | 末端目标位姿 → 关节指令序列 |
| `read_eef_pose(raw_observation)` | 机器人观测 → 统一末端 `Pose` |
| `detect_object(raw_observation)` | 相机观测 → 目标物体位姿（可接第 6 讲结果） |

接入后，状态机入口与 Mock 相同：`PickPlaceStateMachine(backend, task, logger).run()`。
