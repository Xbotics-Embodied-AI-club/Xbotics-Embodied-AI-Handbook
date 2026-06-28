# Lecture 11 — Diffusion Policy

> 对应文稿：见 `docs/` 中第 11 讲

## 本讲 Demo

DP 推理与 receding horizon

## 目录结构

```
lecture11/
├── README.md           # 本文件
├── requirements.txt    # Python 依赖（按需）
├── hardware/           # 真机脚本（SO101 / xLeRobot / G1）
└── simulation/         # 无硬件可运行路径
```

## 快速开始（仿真）

```bash
cd code/lecture11
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python simulation/dp_inference.py
```

## 真机路径

见 `hardware/README.md`（待补充）。

## 状态

- [ ] 仿真 Demo 可运行
- [ ] 真机 Demo 可运行
- [ ] 与文稿实验步骤一致
- [ ] 常见失败已写入文稿

## 贡献

修改本讲代码请开 PR，标题格式：`[Lecture 11] ...`
