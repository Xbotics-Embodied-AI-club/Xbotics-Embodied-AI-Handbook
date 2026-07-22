# 第六部分：具身前沿 —— Agent、世界模型与进展（第 17、20–21 讲）

## 部分定位

第六部分是原「具身前沿」大章中除 VLN 以外的内容：**Robot Agent**（任务规划与 skill 调度）、**世界模型**（预测与数据飞轮）、**前沿进展**（方向梳理 + 全课综合答辩）。

VLN 已独立为 **第五部分**（第 18–19 讲）。

## 覆盖讲次

| 讲次 | 主题 | 协同 |
|------|------|------|
| 第 17 讲 | Robot Agent、任务规划与技能调用 | 富平（Agent） |
| 第 20 讲 | 世界模型 —— 预测、规划与数据飞轮 | 煜恒（世界模型） |
| 第 21 讲 | 具身智能前沿进展 —— 综合闭环与答辩 | 雨浩组 |

**负责人**：雨浩（主责）

## 第 17 讲大纲 - Robot Agent

理解 Robot Agent 尝试解决的问题，理解 Robot Agent 的通用框架：上层 VLM/LLM 规划器、中间安全边界和状态维护、底层电机执行器。

### 17.1 Agent 介绍

#### Agent 的定义

从 chatbot 到 autonomous task executor

#### 模型层演进

LLM、VLM、VLA、WAM

#### Agent 发展

##### 阶段 A：Prompt-only / Chatbot

最早是单轮或多轮对话，模型只输出自然语言。问题是输出不可控、不可稳定解析、不能直接执行动作。

##### 阶段 B：JSON / Structured Output

JSON 解决“模型输出可被程序读取”的问题，但普通 JSON mode 只保证输出是合法 JSON，不保证符合业务 schema；Structured Outputs 才强调按指定 schema 生成。这个阶段的关键词是 **可解析、可验证、可接入业务系统**。

##### 阶段 C：Function Call / Tool Call

Function calling 解决“模型如何调用外部能力”的问题：开发者把工具以 JSON Schema 描述给模型，模型选择工具并生成参数，由应用层执行真实动作。它是从 chatbot 到 agent 的关键转折点。

##### 阶段 D：ReAct / Plan-Act-Observe Loop

这一步让 Agent 不只是一次性调用工具，而是形成循环：**思考 → 行动 → 观察 → 修正计划 → 再行动**。ReAct 的核心就是让语言模型交替生成推理轨迹和任务相关动作，动作再从外部环境获得观察结果。

##### 阶段 E：RAG / Context Engineering

RAG 解决模型知识过期、私有知识缺失的问题。它让 LLM 在生成前检索外部知识库或企业文档，把相关信息注入上下文。这里可以把 **embedding、reranker、vector DB、knowledge graph、document parser** 都纳入“上下文工程”。

##### 阶段 F：Memory

Memory 解决跨会话持续性。建议不要只写 “memery”，而是写 **memory layer**，再补充 Memary / MemGPT / Letta 等代表。Memory 可以拆成五类：

| 记忆类型          | 作用                                 |
| ----------------- | ------------------------------------ |
| Working memory    | 当前上下文窗口里的临时状态           |
| Episodic memory   | 历史交互、执行轨迹、任务经验         |
| Semantic memory   | 用户偏好、事实知识、长期知识图谱     |
| Procedural memory | 可复用流程、技能、代码片段、操作套路 |
| Reflective memory | 失败总结、自我反思、策略修正         |

MemGPT 把 LLM 的上下文管理类比为操作系统的层级内存，用虚拟上下文管理突破固定上下文窗口；Memary 则定位为 autonomous agents 的开源 memory layer，强调自动生成记忆、用户偏好追踪和回放执行历史。
Generative Agents 也值得补充，它把 memory stream、reflection、planning 结合起来，是早期“会记忆、会反思、会计划”的 Agent 架构代表。

##### 阶段 G：MCP

MCP 不是 function call 的替代品，而是更高一层的 **工具/资源/提示词连接协议**。Function calling 更像“模型如何调用某个函数”；MCP 更像“Agent runtime 如何发现、连接、调用外部工具和数据源”。官方定义中，MCP 是连接 AI 应用与外部系统的开源标准，服务器可提供 **Resources、Prompts、Tools**。

##### 阶段 H：Skill

Skill 是把“经验流程”封装成可复用资产。可以理解为 **procedural memory 的工程化形态**：一个 skill 通常包含说明文件、脚本、参考资料、模板等。Agent Skills 规范把 skill 定义为包含 `SKILL.md` 的目录；OpenAI Codex Skills 也采用 `SKILL.md + scripts/references/assets` 的结构。
OpenClaw 文档也把 tool、skill、plugin 分得很清楚：tool 是可调用的 typed function；skill 是加载到 prompt 中的 `SKILL.md` 指令包；plugin 则扩展工具、技能、通道、模型提供商等运行时能力。

##### 阶段 I：Agent Runtime / Framework

OpenClaw、Hermes、LangGraph、AutoGen、CrewAI、LlamaIndex Agents 这类不要和 JSON、MCP 放在同一层。它们更适合归为 **Agent runtime / orchestration framework**。

OpenClaw 可以作为“工具、技能、插件、权限、沙箱、多 Agent 协调”的 runtime 示例；其文档明确区分 tools、skills、plugins。
Hermes 需要分两层讲：**Hermes 模型** 和 **Hermes Agent**。Nous 的 Hermes 3 强调长上下文、多轮对话和 agentic function-calling；Hermes Agent 文档则列出 memory system、skills system、MCP integration、tools、messaging gateway、security 等运行时能力。

### 17.2 Robot Agent 核心体现

- VLA 在长程任务语言理解上的局限
- Robot Agent 尝试解决的问题，相关论文介绍
- Robot Agent 自身的局限性

### 17.3 Robot Agent 框架及核心组件介绍

- MCP
- Skill
- sim2real
- 长程任务
- 运动空间限制
- 急停防护

### 17.4 Robot Agent 仿真与实机效果

- Robot Agent 仿真环境代码讲解与部署流程
- Robot Agent 实机展示与部署流程

## 第 20 讲大纲 - 世界模型

### 20.1 世界模型介绍

- 世界模型的基本概念
- 世界模型和 VLA/VLM 的区别
- 世界模型和传统仿真的区别
- 世界模型的关键难点

### 20.2 世界模型论文领读

- 世界模型的主要技术路线
- 亮点论文讲解
- https://arxiv.org/abs/1803.10122 World Models
- https://arxiv.org/abs/1912.01603 Dream to Control: Learning Behaviors by Latent Imagination
- https://arxiv.org/abs/2402.15391 Genie: Generative Interactive Environments
- https://arxiv.org/abs/2506.09985 V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning
- https://arxiv.org/abs/2309.17080 GAIA-1: A Generative World Model for Autonomous Driving
- https://arxiv.org/abs/2501.03575 Cosmos World Foundation Model Platform for Physical AI

### 20.3 世界模型微调和应用效果

- 世界模型仿真测评运行
- 世界模型实机效果部署

### 20.4 世界模型在机器人中的系统架构

- 和 VLA 配合
- 替换 VLA
- 替换 VLM

## 第 21 讲大纲 - 具身前沿进展

## 阶段项目：语言指令驱动的移动 + 操作综合任务

**目标**：串联 Agent 规划、Part 5 VLN 导航、感知/操作 skill，完成综合 Demo 与答辩。

```
Agent 规划 → VLN navigate_to → detect / grasp / place → check → episode → 失败回流
```

**交付**：综合代码、演示视频、架构图、成功率、失败样本、答辩材料

## 讲次顺序说明

| 推荐顺序 | 讲次 | 说明 |
|----------|------|------|
| 1 | L17 Agent | 先建立 skill 与规划框架 |
| 2 | L18–L19 VLN | 第五部分，补移动 skill |
| 3 | L20 世界模型 | 预测与数据飞轮 |
| 4 | L21 前沿与答辩 | 综合闭环 |
