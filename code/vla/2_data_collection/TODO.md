# TODO · 组2 补仿真采集线（讲9 兜底）

- [ ] `2_5_sim_teleop_record/`：SO-101 仿真器 + 键盘/手柄驱动从臂，采出与实物**同格式**的 LeRobot 数据集（讲9「无真机」兜底线，目前完全缺失）

## 代码风格（后续实现必须完全匹配既有风格）

1. **课堂演示取向**：常量就近内联（不集中 config 块），阅读顺序=讲解顺序，不堆 try/except/抽象层。
2. **禁止**：argparse/args、mock、monkeypatch、改第三方库、`os.environ.get`/默认值/存在性检查、机器名/绝对路径/内部任务号。
3. **自写训练四件套**：普通 `Dataset`+`LightningDataModule`、普通 `nn.Module`+`LightningModule`，入口 `trainer.fit(model, data)`（参照 rl/1_1、rl/2_2）。
4. **路径**：只直读 `DATASETS_ROOT`/`HF_HOME`；组内数据用 `parents[1]/"data"/...` 相对路径；产物落 `DATASETS_ROOT/models/trained/`。
5. **notebook**：先 `.py` 跑通再转同名 `.ipynb`；中文编号分节，开篇讲「做什么/为什么/与前后模块关系」，每节讲动机+关键行+与上一版 diff；代码 cell 与 .py 逐行一致、无输出（参照 rl/1_1/train_v1_reinforce.ipynb）。
6. **模块必备 README**（定位/文件表/运行/结果），环境走统一 `code/pyproject.toml` extra。
