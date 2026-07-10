# GRPO 微调小 VLM：学会数数

用 GRPO（组相对策略优化）微调 Qwen2.5-VL-3B，让它学会数清图里有几个物体。
这是 GRPO 外壳的第一次登场：组采样 → 组内相对优势（不需要 critic）→ 策略梯度，
奖励是可验证的「答案对不对 + 格式合不合规」。下一个模块（`../2_2_grpo_vla0_libero/`）
把同一套外壳原样搬到机器人动作 token 上。

## 文件

| 文件 | 职责 |
|---|---|
| `train_grpo_qwen2_vl.py` | 全流程：数据 → 奖励函数 → GRPO 训练 → 评测/预测，Lightning `trainer.fit(model, data)` |
| `train_grpo_qwen2_vl.ipynb` | 同内容的课堂走读版（中文分节） |

数据在 `../data/clevr_counting/`（CLEVR 数数训练集 + SuperCLEVR-200 评测子集，
含来源清单）；训后 LoRA adapter 与评测摘要在 `../result/2_1_grpo_vlm_counting/`。

## 结果

300 步 GRPO 后，SuperCLEVR-200 上 base → adapter：

| 指标 | before | after |
|---|---|---|
| 数数准确率 | 0.095 | **0.44** |
| 格式合规率 | 0.22 | **0.79** |

## 运行

```bash
cd experiments
uv sync --extra gpu_x86
uv run python rl/2_grpo_posttraining/2_1_grpo_vlm_counting/train_grpo_qwen2_vl.py
```

单张消费级 GPU 即可（4bit 量化 + LoRA）。
