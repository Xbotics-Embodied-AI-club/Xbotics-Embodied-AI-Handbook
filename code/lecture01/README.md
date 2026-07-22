# Lecture 01 — 具身智能导论

> 对应文稿：见 `docs/` 中第 1 讲

## 本讲 Demo

纯 Python 二维 reaching 闭环，用二维点抽象“末端位置接近目标点”。本示例不包含机械臂运动学、动力学或碰撞仿真。

## 目录结构

```
lecture01/
├── README.md           # 本文件
├── requirements.txt    # Python 依赖（按需）
├── hardware/           # 真机脚本（SO101 / xLeRobot / G1）
└── simulation/         # 无硬件可运行路径
```

## 快速开始（仿真）

```bash
cd code/lecture01
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python simulation/minimal_reach.py
```

## 真机路径

见 `hardware/README.md`（待补充）。

## 状态

- [ ] 二维闭环 Demo 已在文稿指定环境复验
- [ ] 真机 Demo 可运行
- [ ] 与文稿实验步骤一致
- [ ] 常见失败已写入文稿

## 贡献

修改本讲代码请开 PR，标题格式：`[Lecture 01] ...`
