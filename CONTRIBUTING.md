# 贡献指南

完整团队分工与里程碑见 [README § 团队分工](README.md#team) 与 [`meta/contributors.md`](meta/contributors.md)。

## 1. 谁改什么

| 部分 | 主负责 | 可改目录 |
|------|--------|----------|
| 第一部分 L01–L04 | 丛林（翼茗、锦丰） | `docs/part1-*`、`code/lecture01–04` |
| 第二部分 L05–L07 | 育帆（昊旺、志凯、彤彤、宝华） | `docs/part2-*`、`code/lecture05–07` |
| 第三、四部分 L08–L16 | harry（陈老师、诸老师、罗辑） | `docs/part3-*`、`part4-*`、`code/lecture08–16` |
| 第五部分 L18–L19 | **新梦**（雨浩协同） | `docs/part5-vln/`、`code/lecture18–19` |
| 第六部分 L17、L20–L21 | 雨浩（富平 L17、煜恒 L20） | `docs/part6-agent-world-model/`、`code/lecture17、20–21` |
| 代码管理 | 志凯 | `code/` 结构、CI、跨讲依赖 |
| 校验 | 乙然 | 全文结构/术语/链接审查 |

跨 Part 修改（如术语统一、SUMMARY 结构）需在大群或 Issue 中先对齐。

## 2. 时间安排

| 节点 | 日期 |
|------|------|
| 整体大纲检查 & 编写规划 | **6 月 30 日**（丛林、木木） |
| 各 Part 细纲 | **7 月 5 日前**（丛林、育帆、**新梦**、harry、雨浩） |
| 第一版初稿 | **7 月 26 日前**（全体） |
| 第一版修改版 | **8 月 2 日前**（全体） |
| 进度同步 | **每周三 21:00**（乙然提醒） |

每次同步后，负责人更新 [`meta/status.md`](meta/status.md)。

## 3. 写作风格（必遵）

本课程是**工程实践课**，不是论文综述。详见 [README § 课程内容风格](README.md#writing-style)。

**四点原则**：简单直接 · 任务驱动 · 理论够用实验优先 · 重视失败复盘

**推荐节奏**：

```
提出问题 → 讲清概念 → 跑通 Demo → 分析失败 → 完成作业
```

**单讲 10 模块**：见 [`templates/lecture-template.md`](templates/lecture-template.md)

1. 本讲目标  
2. 核心知识点（5–10 条，服务实验）  
3. 课堂任务 / 引入案例  
4. 方法框架（可复用三步法 / 五类失败框架等）  
5. 有硬件版 Demo  
6. 无硬件仿真版 Demo  
7. 实验步骤  
8. 作业交付  
9. 常见失败与复盘  
10. 参考开源项目  

## 4. 分支与 PR

```
feat/part1-l02-ros2          # 按 Part / 讲次
fix/lecture06-typo
docs/writing-style-update
```

**PR 标题**：`[Part 2 / L06] 位姿估计文稿初稿`

**合并前检查**（提交者自查 + 乙然校验）：

- [ ] 符合 10 模块模板
- [ ] 有硬件 / 无硬件双路径均完整
- [ ] 含失败复盘，不只写成功 Demo
- [ ] 外链已写入 `references/links.md`
- [ ] `meta/status.md` 已更新
- [ ] 代码路径 `code/lectureNN/README.md` 可独立复现（志凯审核）

## 5. 代码规范

```
code/lectureNN/
├── README.md
├── requirements.txt
├── hardware/       # 真机
└── simulation/     # 无硬件必可运行
```

志凯维护 [`scripts/scaffold_code.py`](scripts/scaffold_code.py) 与跨讲公共依赖约定。

## 6. 配图规范

- 本讲专用：`assets/figures/lectureNN/fig-NN-M-英文名.png`
- 与 [Xbotics-Embodied-Guide](https://github.com/Xbotics-Embodied-AI-club/Xbotics-Embodied-Guide) 共用的系统图、路线图中，README 已注明出处；勿重复上传同一资源，优先链接 Guide 或社区 CDN
- 图标：⭐ 必看 · 🧪 实作 · 📦 代码 · 🎥 视频

## 7. 校验清单（乙然）

- [ ] 章节编号与 `docs/SUMMARY.md` 一致
- [ ] 术语与 `meta/outline.yaml`、Embodied-Guide 一致
- [ ] 每讲有具体「课堂任务」，非空泛概念
- [ ] 「方法框架」可独立复用到其他机器人/数据集
- [ ] 作业交付物可检查、可评分

---

**原则**：先地图后模块；每讲必有 Demo；失败样本与成功同等重要。
