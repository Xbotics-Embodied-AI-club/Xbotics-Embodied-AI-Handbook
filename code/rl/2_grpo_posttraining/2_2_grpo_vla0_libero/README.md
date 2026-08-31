# GRPO 后训练 VLA-0：让机器人策略自我提升成功率

`2_1_grpo_vlm_counting` 里那套 GRPO 教会了 VLM 数数。本模块把**同一套外壳**搬到
机器人上：VLA-0 的动作本来就是一串数字 token，于是「组采样 → 组内相对优势 →
策略梯度」几乎一行不改，只是奖励从「答案对不对」换成「LIBERO 任务成没成」。

## 文件结构

| 文件 | 职责 |
|---|---|
| `env.py` | LIBERO 同状态组环境：一组子环境锁同一初始状态，组 rollout 拿 0/1 成败 |
| `model.py` | VLA-0 采样与解码：生成动作块、数字串→连续动作、复算逐 token logprob |
| `train_grpo.py` | 训练入口：在线采样 Dataset + LightningModule（组相对优势 + KL-to-ref），`trainer.fit(model, data)` |
| `rollout.py` | before/after 贪婪确定性评测对照 |
| `vendor/vla0_smol/` | VLA-0 上游说明（policy 本体经 lerobot 补丁注入，见下） |

`.py` 与同名 `.ipynb` 内容一致，notebook 按讲解顺序分节，适合逐行走读。

## 环境

统一环境见 `code/pyproject.toml` 的 `gpu_x86` extra：

```bash
cd code
bash platform/lerobot/fetch_lerobot.sh          # 拉 lerobot 源并打补丁（含 vla0_smol policy）
uv sync --extra gpu_x86
```

## 弱基座从哪来

官方 VLA-0 在标准 LIBERO 上已经饱和（Object 套件 ~97%），没有提升空间，直接做
RL 看不出效果。所以先**故意训一个不饱和的弱基座**：

```bash
lerobot-train --policy.type=vla0_smol ...   # libero_object 少步 SFT
```

取早期 checkpoint（约 2000 步、task0 成功率 40–50%）作为 GRPO 起点，放到
`DATASETS_ROOT/models/trained/xbotics_rl_grpo_vla0/weak_base_2000/`。

## 训练与结果

```bash
uv run python rl/2_grpo_posttraining/2_2_grpo_vla0_libero/train_grpo.py
uv run python rl/2_grpo_posttraining/2_2_grpo_vla0_libero/rollout.py   # before/after 对照
```

在 libero_object task0 上的确定性成功率：

| 配置 | task0 成功率 |
|---|---|
| 弱基座（before） | 40–50% |
| GRPO 无 KL，2 次更新 | **16.7%**（不升反降） |
| **GRPO + KL-to-ref，2 次更新（after）** | **60–67%** |
| GRPO + KL-to-ref，4 次更新 | 16.7%（见顶后退化） |

三个教学要点，都写在代码注释里：

1. **组内要有混合成败才有信号**：温度 1.0（策略自然分布）+ 前沿初始状态池
   `{0,1,3,6}`（基座成功率非 0 非 1）。温度调高反而把有效成功率压没。
2. **KL-to-ref 是关键变量**：同一位置的 checkpoint，无 KL 16.7% vs 有 KL 66.7%。
   稀疏 0/1 奖励的纯 REINFORCE 没有信赖域，会把策略从 SFT 胜任区拖走。
3. **必须早停**：成功率峰值来得早（2 次更新），继续训会退化。所以密集存档，
   训完用 `rollout.py` 逐个评测挑峰值 checkpoint。

rollout 视频可用 `lerobot-eval` 对基座与训后 checkpoint 各录一遍（自动存
`videos/`），抽帧即得 before/after 对照图。

## 参考

- DeepSeekMath（GRPO 与 k3 KL 估计器）：arXiv 2402.03300
- VLA-0：动作即文本的极简 VLA 路线
