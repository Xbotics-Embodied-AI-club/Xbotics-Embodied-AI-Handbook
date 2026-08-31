# GRPO 后训练：让模型自我提升

GRPO 的题眼是「**强化学习与策略解耦**」：同一套外壳——组采样、组内相对优势
（不需要 critic）、策略梯度——既能微调 VLM 生成的**文本 token**，也能微调 VLA
生成的**动作 token**。本模块提供文本 token 那一侧的动手实验：

| 模块 | 策略 | token | 奖励 | 结果 |
|---|---|---|---|---|
| `2_1_grpo_vlm_counting` | Qwen2.5-VL-3B | 文本（数数答案） | 答案对不对 | 准确率 0.095 → 0.44 |

动作 token 那一侧，讲义第 6 节按 SimpleVLA-RL 导读，不配动手代码。

## 运行

```bash
cd code
uv sync --extra gpu_x86
```

训练入口是 Lightning 的 `trainer.fit(model, data)`，配同名 `.ipynb` 供逐行走读；
数据在 `data/`，结果样例在 `result/`。
