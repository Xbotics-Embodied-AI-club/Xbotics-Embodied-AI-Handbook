# GRPO 后训练：让模型自我提升

GRPO 的题眼是「**强化学习与策略解耦**」：同一套外壳——组采样、组内相对优势
（不需要 critic）、策略梯度——既能微调 VLM 生成的**文本 token**，也能微调 VLA
生成的**动作 token**。下面两个递进的实验演示这一点：

| 模块 | 策略 | token | 奖励 | 结果 |
|---|---|---|---|---|
| `2_1_grpo_vlm_counting` | Qwen2.5-VL-3B | 文本（数数答案） | 答案对不对 | 准确率 0.095 → 0.44 |
| `2_2_grpo_vla0_libero` | VLA-0（0.5B） | 动作（数字串） | LIBERO 任务成没成 | 成功率 40–50% → 60–67% |

从 2_1 到 2_2 的关键增量是 **KL-to-ref 信赖域**：机器人任务的奖励是稀疏 0/1，
纯 REINFORCE 会把策略从 SFT 基座的胜任区拖走（消融：同一位置无 KL 16.7% vs
有 KL 66.7%），KL 把策略锚在基座附近才能涨点。细节见 `2_2` 的 README 与代码注释。

## 运行

```bash
cd code
bash platform/lerobot/fetch_lerobot.sh   # 2_2 需要 lerobot 源（打补丁后 editable 安装）
uv sync --extra gpu_x86         # 2_1 / 2_2 同一套环境
```

两个模块的训练入口都是 Lightning 的 `trainer.fit(model, data)`，各配同名
`.ipynb` 供逐行走读；数据在 `data/`，结果样例在 `result/`。
