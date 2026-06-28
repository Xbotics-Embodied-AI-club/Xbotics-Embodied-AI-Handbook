# Lecture 14 — RL 基础

> 对应文稿：见 `docs/` 中第 14 讲

## 本讲 Demo

reaching PPO/SAC 环境

## 目录结构

```
lecture14/
├── README.md           # 本文件
├── requirements.txt    # Python 依赖（按需）
├── hardware/           # 真机脚本（SO101 / xLeRobot / G1）
└── simulation/         # 无硬件可运行路径
```

## 快速开始（仿真）

```bash
cd code/lecture14
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python simulation/reaching_rl_env.py
```

## 真机路径

见 `hardware/README.md`（待补充）。

## 状态

- [ ] 仿真 Demo 可运行
- [ ] 真机 Demo 可运行
- [ ] 与文稿实验步骤一致
- [ ] 常见失败已写入文稿

## 贡献

修改本讲代码请开 PR，标题格式：`[Lecture 14] ...`
