# 第 2 讲全文写作实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将现有第二讲大纲扩写为面向 ROS2 零基础到入门读者、以原理讲解为主并包含关键代码片段和可复现实验步骤的完整课程讲义。

**Architecture:** 只修改第二讲 Markdown 文件，保留 `2.1—2.9` 的教学顺序，并用“目标发布—状态读取—动作生成—安全检查—执行反馈—episode 记录”的机械臂任务贯穿全文。先完成系统与通信概念，再完成闭环 Demo 和实验，最后统一处理排错、作业、参考资料与格式验证。

**Tech Stack:** Markdown、ROS2 概念与 CLI、Python `rclpy` 教学片段、标准 ROS2 消息类型、Git。

## Global Constraints

- 只修改 `docs/part1-system-basics/02-ros2-architecture.md`；计划文档除外，不新增 Package、源码文件或图片。
- 保留现有 `2.1—2.9` 章节结构和知识范围，不改变课程大纲的教学顺序。
- 面向 ROS2 零基础到入门读者，以原理讲解为主，关键代码片段和实验步骤为辅。
- 与第一讲保持一致，采用场景导入、直观解释、流程图、对比表、关键示例和阶段小结相结合的讲义风格。
- 代码使用 ROS2 Python `rclpy`，只展示理解数据流所需的关键逻辑，不声称构成可直接部署到真实机械臂的完整工程。
- 统一使用 `/target_joint`、`/joint_states`、`/action_command` 和 `/task_status` 作为贯穿 Demo 的核心 Topic 名称。
- 无硬件 mock 闭环是默认实验路径，SO101 或兼容机械臂是替换驱动与控制节点后的扩展路径。
- 所有“待补充”占位必须替换为 Markdown 表格或 fenced text 流程图，不虚构图片素材。
- 真实硬件说明必须包含关节顺序、单位、限位、低速测试、急停和有人监护等安全边界。
- 保留用户已有的未跟踪 `.crdownload` 文件，不暂存、不修改、不删除。

---

## File Structure

**Modify:** `docs/part1-system-basics/02-ros2-architecture.md`

该文件承担本讲全部教学内容，继续沿用仓库中“一讲一个 Markdown 文件”的结构。章节内部职责如下：

- `2.1—2.2`：建立机器人系统整体认知和四层架构；
- `2.3—2.4`：解释 ROS2 通信机制和具身智能模型的位置；
- `2.5—2.7`：构建闭环、展示关键代码并指导实验；
- `2.8—2.9`：完成排错、作业、复盘和参考资料。

不拆分子文件，因为相邻章节均采用单文件讲义形式，而且贯穿案例需要在同一篇正文中保持叙事连续。

## Shared Demo Contract

所有任务使用以下接口，不在后续任务中改名：

| 节点 | 消费 | 产生 | 职责 |
| --- | --- | --- | --- |
| `target_publisher` | 用户输入或定时目标 | `/target_joint` | 发布目标关节位置 |
| `robot_state_node` | mock 对象或硬件反馈 | `/joint_states` | 作为唯一状态发布源，发布当前关节状态 |
| `policy_node` | `/target_joint`、`/joint_states` | `/action_command` | 根据目标误差生成限幅动作 |
| `controller_node` | `/action_command` | mock 对象更新或硬件控制指令 | 执行动作，但不重复发布 `/joint_states` |
| `task_status_node` | `/target_joint`、`/joint_states` | `/task_status` | 判断到达、运行、超时或异常 |
| `episode_recorder` | 上述四个核心 Topic | episode 记录 | 保存观测、动作、时间戳和结果 |

教学片段使用以下标准消息类型：

- `/joint_states`：`sensor_msgs/msg/JointState`；
- `/target_joint`：`std_msgs/msg/Float64MultiArray`；
- `/action_command`：`std_msgs/msg/Float64MultiArray`；
- `/task_status`：`std_msgs/msg/String`。

---

### Task 1: 扩写系统架构、四层结构和 ROS2 通信基础

**Files:**
- Modify: `docs/part1-system-basics/02-ros2-architecture.md:1-约第 280 行`
- Reference: `docs/part1-system-basics/01-introduction.md`
- Reference: `docs/superpowers/specs/2026-07-16-lecture-02-full-chapter-design.md`

**Interfaces:**
- Consumes: 现有 `2.1—2.4` 大纲、Shared Demo Contract。
- Produces: 完整的 `2.1—2.4` 正文，以及后续闭环与实验沿用的四层架构、节点名和 Topic 语义。

- [ ] **Step 1: 记录结构验证的失败基线**

Run:

```bash
rg -n "待补充|^### 2\.[1-4]\.|^## 2\.[1-4] " docs/part1-system-basics/02-ros2-architecture.md
```

Expected: 输出 `2.1—2.4` 的现有标题，并至少出现“四层结构总览”“Topic / Service / Action 对比表”“模型位置总览表”对应的待补充占位，证明这些部分仍是大纲而非完整讲义。

- [ ] **Step 2: 将 2.1 扩写为任务驱动的章节导入**

在 `2.1` 中保留现有学习目标和整体逻辑，新增以下完整内容：

- 用“策略模型已经给出动作，机器人为什么仍可能不动或撞到障碍”作为导入问题；
- 对比单次模型推理与持续机器人闭环；
- 解释驱动、状态、感知、策略、控制、安全和记录缺一不可；
- 用贯穿案例引出 `/target_joint → /action_command → /joint_states → /task_status`；
- 添加 `2.1` 小结，明确本讲关注模块边界和数据流，而不是训练某一种模型。

- [ ] **Step 3: 将 2.2 扩写为有输入输出边界的四层架构**

为本体层、控制层、感知层和策略层分别写出：职责、典型组件、接收的信息、产生的信息，以及在贯穿案例中的位置。用以下表格替换四层总览占位：

| 层级 | 核心问题 | 典型输入 | 典型输出 | 本章案例 |
| --- | --- | --- | --- | --- |
| 策略层 | 下一步做什么 | 任务、观测、机器人状态 | 目标或动作 | 由误差生成 `/action_command` |
| 感知层 | 环境和机器人现在怎样 | 图像、深度、传感器原始数据 | 目标、位姿、语义信息 | 为扩展抓取任务提供目标信息 |
| 控制层 | 动作怎样安全稳定地执行 | 目标、动作、当前状态 | 驱动可执行的控制指令 | 限幅并更新机械臂状态 |
| 本体层 | 系统能感知和作用什么 | 电机指令、物理环境 | 运动、关节和传感器反馈 | 机械臂与夹爪 |

补充“层级不是进程数量”的说明，避免读者误以为每层只能对应一个节点。

- [ ] **Step 4: 将 2.3 扩写为面向初学者的 ROS2 通信说明**

依次解释工作空间、Package、Node、Topic、Service、Action 和 Launch。每个概念都包含直观类比、适用场景、贯穿案例和一个最小命令或代码片段。使用统一对比表替换 Topic / Service / Action 占位，维度固定为：通信方式、是否持续、是否有过程反馈、能否取消和典型场景。

关键 Publisher 片段使用以下接口和结构：

```python
from std_msgs.msg import Float64MultiArray

self.target_pub = self.create_publisher(
    Float64MultiArray, "/target_joint", 10
)

msg = Float64MultiArray()
msg.data = [0.0, -0.4, 0.8, 0.0, 0.4, 0.0]
self.target_pub.publish(msg)
```

命令示例至少包含：

```bash
ros2 node list
ros2 topic list
ros2 topic echo /joint_states
ros2 topic hz /joint_states
```

- [ ] **Step 5: 将 2.4 扩写为模型位置与系统边界说明**

分别说明 VLM/检测/分割、VLA、ACT、Diffusion Policy、强化学习和世界模型的输入、输出和所在层级。明确 VLA 或模仿学习策略的输出不能绕过控制与安全检查直接发送给电机。用下表替换模型位置总览占位：

| 模型或方法 | 主要位置 | 典型输入 | 典型输出 | 接入系统时还需要什么 |
| --- | --- | --- | --- | --- |
| VLM / 检测 / 分割 | 感知层 | 图像、文本 | 类别、框、mask、语义 | 位姿估计与坐标变换 |
| VLA | 策略层 | 语言、图像、状态 | 动作或动作序列 | 动作适配、控制器、安全检查 |
| ACT / Diffusion Policy | 策略层 | 观测、状态 | action chunk | 时序执行、限幅、反馈 |
| 强化学习 | 策略层或控制层 | 状态、观测 | 高层动作或低层控制量 | 仿真验证与真实机安全约束 |
| 世界模型 | 预测与规划模块 | 历史状态、观测、动作 | 未来预测或候选轨迹 | 规划器、策略与真实反馈校正 |

- [ ] **Step 6: 验证 Task 1 内容和格式**

Run:

```bash
rg -n "待补充|/target_pose|/gripper_command" docs/part1-system-basics/02-ros2-architecture.md
rg -n "^## 2\.[1-4] |^### 2\.[1-4]\." docs/part1-system-basics/02-ros2-architecture.md
git diff --check -- docs/part1-system-basics/02-ros2-architecture.md
```

Expected: 第一条不再命中 `2.1—2.4` 范围内的占位或已弃用的贯穿接口；第二条显示 `2.1—2.4` 标题顺序完整；`git diff --check` 无输出并返回 0。

- [ ] **Step 7: 提交前半章扩写**

```bash
git add docs/part1-system-basics/02-ros2-architecture.md
git commit -m "扩写第二讲机器人系统与ROS2基础"
```

---

### Task 2: 扩写机器人闭环、关键代码和实验步骤

**Files:**
- Modify: `docs/part1-system-basics/02-ros2-architecture.md:约第 280-470 行`
- Reference: `docs/superpowers/specs/2026-07-16-lecture-02-full-chapter-design.md`

**Interfaces:**
- Consumes: Task 1 中确定的四层架构、Shared Demo Contract、四个核心 Topic 及消息类型。
- Produces: 完整 `2.5—2.7` 正文、关键 `rclpy` 片段、mock 和硬件两条实验路径，以及 Task 3 使用的验收命令和排错现象。

- [ ] **Step 1: 记录闭环与实验部分的失败基线**

Run:

```bash
rg -n "待补充|^### 2\.[5-7]\.|^## 2\.[5-7] " docs/part1-system-basics/02-ros2-architecture.md
```

Expected: 输出 `2.5—2.7` 标题，并命中 Demo 知识点占位，且实验部分主要表现为条目式步骤。

- [ ] **Step 2: 将 2.5 扩写为可追踪的数据闭环**

按“读取状态—接收目标—生成动作—安全检查—执行运动—读取反馈—判断结果—记录 episode”展开。为每一步说明输入、输出、失败条件和下一步消费者。加入状态新鲜度、控制频率、动作维度、关节限位和任务超时的解释。

episode 字段表固定包含：`timestamp`、`joint_state`、`target_joint`、`action_command`、`task_status`、`success` 和 `error`。给出一条 JSON 教学样例，明确它用于说明字段关系而非规定 LeRobot 的唯一存储格式。

- [ ] **Step 3: 在 2.6 编写 policy_node 的关键逻辑片段**

片段使用以下一致逻辑：

```python
import numpy as np
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

def build_action(current, target, gain=0.5, max_step=0.05):
    current = np.asarray(current, dtype=float)
    target = np.asarray(target, dtype=float)
    if current.shape != target.shape:
        raise ValueError("current and target must have the same shape")
    error = target - current
    return np.clip(gain * error, -max_step, max_step)

def on_joint_state(self, msg: JointState):
    if not msg.position or self.target is None:
        return
    action = build_action(msg.position, self.target)
    command = Float64MultiArray()
    command.data = action.tolist()
    self.action_pub.publish(command)
```

正文解释比例增益、单步限幅、维度检查和“策略输出不是最终电机命令”。不引入完整类、安装依赖或可直接部署承诺。

- [ ] **Step 4: 在 2.6 补充状态检查、任务判断和记录片段**

使用伪代码与短 Python 片段解释：

```python
state_age = self.get_clock().now() - self.last_state_time
if state_age.nanoseconds > 200_000_000:
    self.publish_status("stale_state")
    return

reached = max(abs(t - q) for t, q in zip(self.target, self.current)) < 0.02
self.publish_status("reached" if reached else "running")
```

记录样例应把同一时刻的观测、目标、动作、状态和结果组合为一条记录。说明生产系统需要更严格的时间同步和数据格式管理。

- [ ] **Step 5: 用节点与接口表替换 Demo 知识点占位**

表格列固定为：节点、输入、输出、对应知识点、替换硬件时是否保留。明确 `controller_node` 在 mock 模式下更新模拟对象，在硬件模式下必须替换为经过验证的驱动适配器；`robot_state_node` 在两种路径中都是 `/joint_states` 的唯一发布源。

Launch 仅展示组织关系：

```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(package="robot_demo", executable="robot_state_node"),
        Node(package="robot_demo", executable="target_publisher"),
        Node(package="robot_demo", executable="policy_node"),
        Node(package="robot_demo", executable="controller_node"),
        Node(package="robot_demo", executable="task_status_node"),
        Node(package="robot_demo", executable="episode_recorder"),
    ])
```

- [ ] **Step 6: 将 2.7 扩写为默认 mock 实验与硬件替换实验**

默认实验步骤依次包含：创建工作空间、创建教学 Package、准备消息依赖、启动节点、确认节点、确认 Topic、发布目标、观察状态与频率、查看 `rqt_graph`、检查任务状态和 episode。命令使用 Shared Demo Contract 的接口。

硬件替换路径明确检查：关节名和顺序、弧度或角度单位、控制模式、关节限位、速度限制、状态频率、急停、低速空载测试和现场监护。不得建议读者直接把示例 `/action_command` 接到电机接口。

- [ ] **Step 7: 验证 Task 2 的接口与代码围栏**

Run:

```bash
rg -n "/target_pose|/target_joint|/joint_states|/action_command|/task_status" docs/part1-system-basics/02-ros2-architecture.md
awk '/^```/{n++} END{print n; exit n % 2}' docs/part1-system-basics/02-ros2-architecture.md
git diff --check -- docs/part1-system-basics/02-ros2-architecture.md
```

Expected: 不再使用 `/target_pose` 作为贯穿 Demo 接口；四个核心 Topic 在概念、代码和实验中都有命中；代码围栏计数为偶数且命令返回 0；`git diff --check` 无输出并返回 0。

- [ ] **Step 8: 提交闭环与实验扩写**

```bash
git add docs/part1-system-basics/02-ros2-architecture.md
git commit -m "补全第二讲ROS2闭环实验"
```

---

### Task 3: 完成排错、作业、参考资料和整章校验

**Files:**
- Modify: `docs/part1-system-basics/02-ros2-architecture.md:约第 470 行至文件末尾`
- Reference: `docs/part1-system-basics/01-introduction.md`
- Reference: `docs/superpowers/specs/2026-07-16-lecture-02-full-chapter-design.md`

**Interfaces:**
- Consumes: Task 1 和 Task 2 完成的术语、Topic 名称、实验步骤和失败条件。
- Produces: 完整 `2.8—2.9`、统一润色后的第二讲全文和最终验证证据。

- [ ] **Step 1: 将常见失败 checklist 改为可操作排查表**

表格使用“现象、可能原因、检查方法、修复方向”四列，至少覆盖：节点缺失、Topic 无数据、消息类型不匹配、状态频率低、状态过期、动作维度错误、动作越界、目标超时、坐标或单位错误、episode 字段缺失。检查方法必须引用具体命令或字段，例如 `ros2 topic info -v /joint_states`、`ros2 topic hz /joint_states` 和 `task_status`。

- [ ] **Step 2: 完善作业交付与复盘问题**

保留现有交付项，并将验收标准写清：节点图能显示完整闭环、运行记录能展示目标变化、episode 包含规定字段、说明文字能正确放置 VLA、世界模型和控制器。复盘问题保留开放性，但每题都能从正文找到推理依据。

- [ ] **Step 3: 将参考资料改为标准 Markdown 链接**

至少包含以下官方入口并说明用途：

```markdown
- [ROS 2 Documentation](https://docs.ros.org/)：查询 ROS2 概念、安装说明和发行版文档。
- [ROS 2 Tutorials](https://docs.ros.org/en/rolling/Tutorials.html)：学习节点、Topic、Service、Action 和 Launch。
- [LeRobot GitHub](https://github.com/huggingface/lerobot)：查看机器人数据、策略训练与设备支持代码。
- [LeRobot on Hugging Face](https://huggingface.co/lerobot)：浏览数据集、模型和项目资源。
```

- [ ] **Step 4: 进行整章语言与一致性编辑**

从头到尾阅读全文并执行以下编辑：

- 删除大纲式重复句和没有解释作用的堆叠列表；
- 保证第一次出现的缩写有中文解释；
- 保证“状态、目标、动作、控制指令、反馈、episode”用词稳定；
- 保证四层架构与 ROS2 节点不是一一绑定的错误暗示；
- 保证 mock 与硬件路径的边界明确；
- 保证安全说明出现在首次执行动作前，而不是只放在结尾；
- 每个主要章节用一段过渡或小结连接前后内容。

- [ ] **Step 5: 运行最终结构验证**

Run:

```bash
test "$(rg -c '^# ' docs/part1-system-basics/02-ros2-architecture.md)" -eq 1
test "$(rg -c '^## 2\.[1-9] ' docs/part1-system-basics/02-ros2-architecture.md)" -eq 9
test -z "$(rg -n '待补充|TBD|TODO|暂时无法在飞书文档外展示' docs/part1-system-basics/02-ros2-architecture.md)"
awk '/^```/{n++} END{print n; exit n % 2}' docs/part1-system-basics/02-ros2-architecture.md
git diff --check -- docs/part1-system-basics/02-ros2-architecture.md
```

Expected: 前三个 `test` 均返回 0；代码围栏数为偶数且 `awk` 返回 0；`git diff --check` 无输出并返回 0。

- [ ] **Step 6: 运行最终内容验证**

Run:

```bash
rg -n "本体层|控制层|感知层|策略层|Node|Topic|Service|Action|Launch|VLA|Diffusion Policy|强化学习|世界模型|episode" docs/part1-system-basics/02-ros2-architecture.md
rg -n "/target_joint|/joint_states|/action_command|/task_status|ros2 node list|ros2 topic echo|ros2 topic hz|rqt_graph" docs/part1-system-basics/02-ros2-architecture.md
git status --short
```

Expected: 第一条覆盖全部必讲概念；第二条覆盖贯穿接口和实验工具；状态只显示计划内的第二讲修改以及用户原有的未跟踪 `.crdownload` 文件，不包含其他意外文件。

- [ ] **Step 7: 审阅最终差异并提交完整章节**

Run:

```bash
git diff --stat
git diff -- docs/part1-system-basics/02-ros2-architecture.md
```

确认差异只扩写第二讲，没有删除 `2.1—2.9` 的原有知识范围。然后提交：

```bash
git add docs/part1-system-basics/02-ros2-architecture.md
git commit -m "完成第二讲机器人系统架构全文"
```
