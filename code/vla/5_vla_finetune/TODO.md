# TODO · 组5 VLA 微调（讲12）

- [ ] `5_1_vla0_full_sft/`：全量微调 VLA-0（0.5B）→ SO-101 语言抓取，训练曲线+成功率（命令链已在弱基座实验验证：`lerobot-train --policy.type=vla0_smol`，单卡）
- [ ] `5_2_smolvla_full_sft/`：全量微调 SmolVLA → SO-101，同口径
- [ ] `5_3_pi0_lora/`：LoRA 微调 π0 → SO-101（无真机则 LIBERO 同任务）
- [ ] `5_4_pi0_multi_gpu_full/`：π0 多卡全量微调（单卡放不下，讲义 §4.6 多卡实战）

数据依赖：SO-101 采集用 `../2_data_collection/` 工具链；数据集按 repo_id 落 `$HF_LEROBOT_HOME`。

## 代码风格（后续实现必须完全匹配既有风格）

1. **课堂演示取向**：常量就近内联（不集中 config 块），阅读顺序=讲解顺序，不堆 try/except/抽象层。
2. **禁止**：argparse/args、mock、monkeypatch、改第三方库、`os.environ.get`/默认值/存在性检查、机器名/绝对路径/内部任务号。
3. **自写训练四件套**：普通 `Dataset`+`LightningDataModule`、普通 `nn.Module`+`LightningModule`，入口 `trainer.fit(model, data)`（参照 rl/1_1、rl/2_2）。
4. **路径**：只直读 `DATASETS_ROOT`/`HF_HOME`；组内数据用 `parents[1]/"data"/...` 相对路径；产物落 `DATASETS_ROOT/models/trained/`。
5. **notebook**：先 `.py` 跑通再转同名 `.ipynb`；中文编号分节，开篇讲「做什么/为什么/与前后模块关系」，每节讲动机+关键行+与上一版 diff；代码 cell 与 .py 逐行一致、无输出（参照 rl/1_1/train_v1_reinforce.ipynb）。
6. **模块必备 README**（定位/文件表/运行/结果），环境走统一 `code/pyproject.toml` extra。
