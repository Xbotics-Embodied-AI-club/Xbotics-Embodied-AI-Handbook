# Lecture 20 — 世界模型

> 对应文稿：见 `docs/` 中第 20 讲

## 本讲 Demo

一步 state 预测对比 Demo

## 目录结构

```
lecture20/
├── README.md           # 本文件
├── requirements.txt    # Python 依赖（按需）
├── hardware/           # 真机脚本（SO101 / xLeRobot / G1）
└── simulation/         # 无硬件可运行路径
```

## 快速开始（仿真）

```bash
cd code/lecture20
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python simulation/world_model_predict.py
```

## 真机路径

见 `hardware/README.md`（待补充，需根据 SO101 / xLeRobot / G1 现场硬件选择其一）。

## 状态

- [x] 仿真 Demo 可运行
- [ ] 真机 Demo 可运行
- [x] 与文稿实验步骤一致
- [x] 常见失败已写入文稿

## 贡献

修改本讲代码请开 PR，标题格式：`[Lecture 20] ...`
