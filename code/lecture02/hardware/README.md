# 真机 Demo

> **状态：未在真机验证。** 以下是从 mock 闭环迁移到真机的接入指引，SO101 / xLeRobot / G1 的具体运行步骤待补充。

## 只替换两个边界

本讲 Demo 的设计目标之一，就是让「换成真机」不需要重写整个闭环。四个核心 Topic 的契约（`/target_joint`、`/joint_states`、`/action_command`、`/task_status`）保持不变，只动两个节点：

| 节点 | mock 行为 | 真机需要改成 |
| --- | --- | --- |
| `robot_state_node` | 维护内存中的 mock 关节位置，20Hz 发布 `/joint_states` | 从 SDK 读取真实关节反馈后发布，**Topic 与消息格式不变** |
| `controller_node` | 把校验后的增量发布到 `/mock_joint_increment` | 向厂商控制器发送经过验证的驱动命令，**维度校验与限幅逻辑保持不变** |

其余四个节点（`target_publisher`、`policy_node`、`task_status_node`、`episode_recorder`）无需改动。内部 Topic `/mock_joint_increment` 在真机路径下随之移除。

## 接入前必须确认

1. **关节名称与顺序**：改 `robot_demo/common.py` 的 `JOINT_NAMES`，不要在各节点里散着改。
2. **关节限位**：`JOINT_LIMITS` 必须换成真机的实际限位，mock 用的 ±π 对真机通常是不安全的。
3. **单步限幅**：真机首次联调建议把 `MAX_STEP` 调得比 mock 更小，确认方向正确后再放大。
4. **绝不绕过厂商控制器**：`/action_command` 是位置增量，必须经过 `controller_node` 校验后交给验证过的驱动接口，不能直连电机。
5. **急停可达**：真机运行时保持物理急停在手边，先在关节空间小幅度试探再跑完整闭环。

## 待补充

- [ ] SO101 关节读写接口对接与实测
- [ ] xLeRobot 运行步骤
- [ ] G1 运行步骤
- [ ] 真机 episode 记录样例
