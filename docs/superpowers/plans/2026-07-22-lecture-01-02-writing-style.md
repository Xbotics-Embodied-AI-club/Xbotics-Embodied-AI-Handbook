# 第一、二讲写作风格改写实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不改变知识范围、技术事实、图片引用和实验要求的前提下，将第一、二讲改写为问题牵引、叙事连贯、兼具专业性与可读性的技术讲义。

**Architecture:** 两章保持现有文件与二级章节结构，分别围绕“会回答不等于做得成”和“模型不等于系统”建立主线。每个任务只改一个连续内容区间，改后立即检查标题、图片引用、代码围栏和差异范围；最终再做跨章术语、节奏与隔离验收。

**Tech Stack:** Markdown、Git、ripgrep、POSIX shell、现有课程讲义与图片引用。

## Global Constraints

- 仅修改 `docs/part1-system-basics/01-introduction.md` 与 `docs/part1-system-basics/02-ros2-architecture.md` 的正文，以及本实施计划文件。
- 不修改、生成、删除、移动或重命名任何图片文件。
- 不修改现有图片路径、图号和图注。
- 保留两讲现有一级、二级章节的知识范围和教学顺序。
- 保留代码逻辑、命令、接口名、路径、Demo、实验、作业、排错、参考资料和安全边界。
- 第一讲围绕“会回答不等于做得成”，第二讲围绕“模型不等于系统”。
- 语言采用短段落、问题牵引、直观解释、机制展开、案例验证和边界提醒，不使用营销化表达。
- 不将三篇参考文章中的事实、案例或具体句子移植到教材。
- 所有修改只发生在 `codex/lecture-text-style` 分支的独立 worktree 中。

---

### Task 1: 建立不可变基线与改写检查清单

**Files:**
- Inspect: `docs/part1-system-basics/01-introduction.md`
- Inspect: `docs/part1-system-basics/02-ros2-architecture.md`
- Inspect: `docs/superpowers/specs/2026-07-22-lecture-01-02-writing-style-design.md`

**Interfaces:**
- Consumes: 已确认的写作风格设计与当前两章 Markdown。
- Produces: 后续任务共同使用的标题、图片引用、图注和代码围栏基线。

- [ ] **Step 1: 确认工作树和分支隔离**

Run:

```bash
git branch --show-current
git status --short
```

Expected: 分支为 `codex/lecture-text-style`；计划提交后工作区无未提交改动。

- [ ] **Step 2: 记录两章结构与图片引用基线**

Run:

```bash
rg -n '^#{1,3} ' docs/part1-system-basics/01-introduction.md docs/part1-system-basics/02-ros2-architecture.md
rg -n '^!\[|^\*图 [12]-' docs/part1-system-basics/01-introduction.md docs/part1-system-basics/02-ros2-architecture.md
```

Expected: 第一讲包含 `1.1—1.8`，第二讲包含 `2.1—2.9`；所有现有图片引用和图注均可列出。

- [ ] **Step 3: 确认基线代码围栏为偶数**

Run:

```bash
for file in docs/part1-system-basics/01-introduction.md docs/part1-system-basics/02-ros2-architecture.md; do count=$(rg -c '^```' "$file"); printf '%s %s\n' "$file" "$count"; test $((count % 2)) -eq 0; done
```

Expected: 两个文件的围栏数量均为偶数，命令退出状态为 `0`。

### Task 2: 改写第一讲的概念主线

**Files:**
- Modify: `docs/part1-system-basics/01-introduction.md:1-300`

**Interfaces:**
- Consumes: 第一讲现有标题、图片、具身智能定义、七个组成部分、任务闭环和技术路线。
- Produces: 从“会回答”到“做得成”的连续叙事，为 Demo 与课程行动部分建立问题背景。

- [ ] **Step 1: 改写开篇与 1.1**

将开头组织为：读者熟悉的大模型能力 → “如何抓杯子”的语言答案 → 真实机器人仍可能失败 → 问题不在回答，而在物理闭环。保留学习目标、图 1-1、四类物理约束和对比表，使每段只承载一个中心判断。

- [ ] **Step 2: 改写 1.2**

先给出“身体不是模型的外壳，而是能力边界”的判断，再解释具身智能定义和七个组成部分。保留图 1-2、组成部分表格、感知信息列表和数据闭环，增加从身体、感知到数据的因果衔接。

- [ ] **Step 3: 改写 1.3**

以“把红色杯子放到托盘”为唯一贯穿案例，按任务输入、感知、状态估计、策略规划、控制、反馈、数据记录推进。每个模块先回答“为什么需要它”，再说明“它做什么”，保留图 1-3 和现有技术边界。

- [ ] **Step 4: 改写 1.4**

用同一任务比较传统 Pipeline、模仿学习、强化学习、VLA、世界模型与 World Action Model。每条路线明确“擅长解决什么、依赖什么、不能单独解决什么”，保留图 1-4、图 1-5、图 1-6 及现有表格和术语。

- [ ] **Step 5: 检查第一讲前半部分**

Run:

```bash
rg -n '^## 1\.[1-4]|^### 1\.[1-4]\.' docs/part1-system-basics/01-introduction.md
git diff --check -- docs/part1-system-basics/01-introduction.md
```

Expected: `1.1—1.4` 标题层级连续；`git diff --check` 无输出并返回 `0`。

### Task 3: 改写第一讲的场景、Demo 与收束

**Files:**
- Modify: `docs/part1-system-basics/01-introduction.md:301-513`

**Interfaces:**
- Consumes: Task 2 建立的“会回答不等于做得成”主线。
- Produces: 将概念地图落到硬件、Demo、作业与课程后续路线的完整第一讲。

- [ ] **Step 1: 改写 1.5**

从“具身智能最终必须落到什么身体和场景”切入，保留应用场景、机器人本体和课程硬件体系。解释场景、身体、传感器与动作空间之间的约束关系，避免设备名录式罗列。

- [ ] **Step 2: 改写 1.6**

把 Demo 写成对前文闭环的第一次验证：读者需要观察目标、状态、动作、反馈怎样连接。保留图 1-7、图 1-8、有硬件和无硬件路径、操作步骤与复盘问题，不改变代码或安全要求。

- [ ] **Step 3: 改写 1.7 与 1.8**

将作业、讨论、误区和评价标准组织为“如何证明自己真的理解闭环”。总结部分回收第一讲核心问题，保留图 1-9、推荐资源、五件事、一句话总结和术语表。

- [ ] **Step 4: 验证第一讲结构、图片与围栏**

Run:

```bash
test "$(rg -c '^# ' docs/part1-system-basics/01-introduction.md)" -eq 1
test $(( $(rg -c '^```' docs/part1-system-basics/01-introduction.md) % 2 )) -eq 0
diff <(git show HEAD:docs/part1-system-basics/01-introduction.md | rg '^!\[|^\*图 1-') <(rg '^!\[|^\*图 1-' docs/part1-system-basics/01-introduction.md)
git diff --check -- docs/part1-system-basics/01-introduction.md
```

Expected: 所有命令返回 `0`；图片引用和图注与基线完全一致。

- [ ] **Step 5: 提交第一讲**

```bash
git add docs/part1-system-basics/01-introduction.md
git commit -m "重写第一讲叙事表达"
```

Expected: 提交只包含 `01-introduction.md`。

### Task 4: 改写第二讲的系统与 ROS2 主线

**Files:**
- Modify: `docs/part1-system-basics/02-ros2-architecture.md:1-342`

**Interfaces:**
- Consumes: 第二讲学习目标、软硬件架构、ROS2 概念与统一机械臂任务。
- Produces: 从“模型不等于系统”到“模块如何可靠协作”的前半章叙事。

- [ ] **Step 1: 改写 2.1**

从“策略已经生成动作，机械臂为什么仍可能不动或乱动”切入，提出模型之外的系统问题。保留学习目标、整体逻辑、机械臂任务和图 2-1，强化全章统一问题。

- [ ] **Step 2: 改写 2.2**

围绕状态上行、指令下行和多级闭环解释硬件、计算平台、软件栈与接口契约。保留图 2-2、图 2-3、所有架构表格和安全边界，减少模块清单式表达。

- [ ] **Step 3: 改写 2.3**

按“模块为什么需要通信 → ROS2 解决什么 → Node、Topic、Service、Action、Launch 各自适合什么”推进。保留命令、代码、接口对比表和本章 Topic 名称，突出通信成功不等于任务正确。

- [ ] **Step 4: 改写 2.4**

解释 VLM、VLA、ACT、Diffusion Policy、强化学习和世界模型在系统中的位置，强调模型输出必须经过接口适配、控制和安全约束。保持现有技术范围，不补充未经核查的新趋势事实。

- [ ] **Step 5: 检查第二讲前半部分**

Run:

```bash
rg -n '^## 2\.[1-4]|^### 2\.[1-4]\.' docs/part1-system-basics/02-ros2-architecture.md
git diff --check -- docs/part1-system-basics/02-ros2-architecture.md
```

Expected: `2.1—2.4` 标题层级连续；`git diff --check` 无输出并返回 `0`。

### Task 5: 改写第二讲的闭环、Demo、实验与收束

**Files:**
- Modify: `docs/part1-system-basics/02-ros2-architecture.md:343-856`

**Interfaces:**
- Consumes: Task 4 建立的“模型不等于系统”主线和统一机械臂任务。
- Produces: 可执行、可观察、可验证、可恢复的完整第二讲。

- [ ] **Step 1: 改写 2.5**

以一次目标关节任务为线索，解释状态读取、动作生成、动作执行、任务反馈和 episode 记录。保留图 2-4、字段表和失败边界，强调时间戳、状态新鲜度和成功判定属于系统正确性。

- [ ] **Step 2: 改写 2.6**

把 Demo 组织为最小闭环的系统实现：节点职责 → 接口 → 关键代码 → mock 与硬件边界。保留图 2-5、所有代码片段、节点名、Topic 名和安全提醒，不声称代码可直接控制真实硬件。

- [ ] **Step 3: 改写 2.7**

在不改变命令和步骤顺序的前提下，给每一步补充“为什么做”和“看到什么才算正确”。保留 mock 默认路径、真实机械臂替换条件和实验完成标准。

- [ ] **Step 4: 改写 2.8 与 2.9**

将交付物、排错表和复盘问题收束到系统思维：可观察、可验证、可恢复。保留全部参考资料，不新增未经核查的链接。

- [ ] **Step 5: 验证第二讲结构、图片与围栏**

Run:

```bash
test "$(rg -c '^# ' docs/part1-system-basics/02-ros2-architecture.md)" -eq 1
test $(( $(rg -c '^```' docs/part1-system-basics/02-ros2-architecture.md) % 2 )) -eq 0
diff <(git show HEAD:docs/part1-system-basics/02-ros2-architecture.md | rg '^!\[|^\*图 2-') <(rg '^!\[|^\*图 2-' docs/part1-system-basics/02-ros2-architecture.md)
git diff --check -- docs/part1-system-basics/02-ros2-architecture.md
```

Expected: 所有命令返回 `0`；图片引用和图注与基线完全一致。

- [ ] **Step 6: 提交第二讲**

```bash
git add docs/part1-system-basics/02-ros2-architecture.md
git commit -m "重写第二讲叙事表达"
```

Expected: 提交只包含 `02-ros2-architecture.md`。

### Task 6: 跨章编辑与最终隔离验收

**Files:**
- Modify: `docs/part1-system-basics/01-introduction.md`
- Modify: `docs/part1-system-basics/02-ros2-architecture.md`

**Interfaces:**
- Consumes: 已完成改写的第一讲和第二讲。
- Produces: 术语一致、衔接自然、改动范围严格受控的最终文本。

- [ ] **Step 1: 检查重复表达与术语一致性**

Run:

```bash
rg -n '说得对|做得成|模型不等于系统|闭环|Pipeline|pipeline|ROS ?2|episode|Episode|Topic|Action|Service' docs/part1-system-basics/01-introduction.md docs/part1-system-basics/02-ros2-architecture.md
```

Expected: 核心判断出现于关键位置但不过度重复；`ROS2`、`Topic`、`Service`、`Action`、`episode` 等称呼在上下文中一致。

- [ ] **Step 2: 通读并修正跨章衔接**

第一讲结尾应自然引向第二讲的系统架构；第二讲开头不重复整段第一讲定义。删除机械式小结和重复定义，保留承担教学复盘作用的总结。

- [ ] **Step 3: 执行最终 Markdown 验证**

Run:

```bash
for file in docs/part1-system-basics/01-introduction.md docs/part1-system-basics/02-ros2-architecture.md; do test "$(rg -c '^# ' "$file")" -eq 1; count=$(rg -c '^```' "$file"); test $((count % 2)) -eq 0; done
git diff --check
```

Expected: 所有命令返回 `0`，无空白错误。

- [ ] **Step 4: 验证零图片改动与文件范围**

Run:

```bash
git diff --name-only feat/part1
git status --short
git diff --numstat feat/part1 | rg '\.(png|jpg|jpeg|gif|webp|svg)$' && exit 1 || true
```

Expected: 分支相对 `feat/part1` 的差异只包含设计说明、实施计划和两个章节 Markdown；没有图片扩展名。工作区无未提交文件。

- [ ] **Step 5: 提交跨章编辑与计划记录**

```bash
git add docs/part1-system-basics/01-introduction.md docs/part1-system-basics/02-ros2-architecture.md
git commit -m "统一第一二讲写作风格"
```

Expected: 最终提交只包含两章文字微调；提交后 `git status --short` 无输出。
