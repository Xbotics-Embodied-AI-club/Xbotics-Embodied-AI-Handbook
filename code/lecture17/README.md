# Lecture 17 — Embodied Agent

> 对应文稿：[`docs/part6-agent-world-model/17-embodied-agent.md`](../../docs/part6-agent-world-model/17-embodied-agent.md)

## 本讲 Demo

LangGraph skill 调用 mock

## 目录结构

```
lecture17/
├── README.md           # 本文件
├── requirements.txt    # Python 依赖（按需）
├── hardware/           # 真机脚本（SO101 / xLeRobot / G1）
└── simulation/         # 无硬件可运行路径
```

## 快速开始（仿真）

```bash
cd code/lecture17
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python simulation/agent_skills.py
```

## 真机路径

见 `hardware/README.md`（待补充，需根据 SO101 / xLeRobot / G1 现场硬件选择其一）。

## 状态

- [x] 仿真 Demo 可运行
- [ ] 真机 Demo 可运行
- [x] 与文稿实验步骤一致
- [x] 常见失败已写入文稿

## 贡献

修改本讲代码请开 PR，标题格式：`[Lecture 17] ...`
