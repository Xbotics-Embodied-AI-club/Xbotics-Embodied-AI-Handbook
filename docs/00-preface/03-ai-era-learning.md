# AI 时代的学习方法

本教程鼓励结合 AI 工具与云平台，降低环境配置成本，提高迭代效率。

## 1. 问题驱动 + AI 辅助

- 每讲开始前列出 3–5 个「应能回答的问题」
- 遇到概念阻塞时，用 AI 做**对比解释**（如 Pipeline vs VLA vs 世界模型）
- 失败复盘时，用 AI 辅助分类失败类型，但**必须以日志、视频和传感器数据为准**

## 2. 云平台减少环境配置

推荐场景：

| 场景 | 建议 |
|------|------|
| GPU 训练（BC / ACT / DP / 小型 VLA） | 云 GPU 实例 + LeRobot / robomimic |
| Isaac Lab / 大规模 RL | 高配 GPU 云工作站 |
| 无 GPU 入门 | MuJoCo + ManiSkill + ROS2 mock |

本地最低配置见 [`appendix/hardware-setup.md`](../appendix/hardware-setup.md)。

## 3. Vibe Coding 技巧（具身场景）

- **先跑通最小闭环**，再扩展模块（reach → pick-place → 数据 → 训练）
- 让 AI 生成**骨架代码**，人工核对：坐标系、action 单位、控制频率
- 每讲维护 `code/lectureNN/README.md` 作为「单一事实来源」
- 真机前必须在仿真完成同任务对照

## 4. 协作写作

- 书稿按讲次拆分，见 [`docs/SUMMARY.md`](../SUMMARY.md)
- 认领与进度：[`meta/status.md`](../../meta/status.md)
- 贡献流程：[`CONTRIBUTING.md`](../../CONTRIBUTING.md)
