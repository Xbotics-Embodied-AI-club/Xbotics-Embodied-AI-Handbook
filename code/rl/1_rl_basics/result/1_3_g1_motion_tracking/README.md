# BeyondMimic Results

This directory contains the student-facing simulator output for the RL course BeyondMimic demo.

The current run trains a PyTorch Lightning PPO policy on:

```text
code/rl/1_rl_basics/data/g1_reference_motions/marshal-arts.npz
```

Full checkpoints and W&B run files go under:

```text
DATASETS_ROOT/models/trained/xbotics_rl_beyondmimic/beyondmimic-marshal-arts-lightning-10000/
```

The final simulator outputs are:

```text
code/rl/1_rl_basics/result/1_3_g1_motion_tracking/
├── marshal-arts-model_10000.json
└── marshal-arts-model_10000.mp4
```

Rollout videos hide mjlab's reference-motion ghost by default, so the result shows the executed robot only.

## 三个算法对照（动作跟随任务）

同一个动作跟随任务（marshal-arts）。评测口径一致：确定性动作，16 环境 × 400 步。

| 版本 | 算法 | 评测 mean_reward | 摔倒率 | W&B run |
|---|---|---|---|---|
| v1 | REINFORCE | 0.052 | 2.6% | `rl_class/n7w7w1ra` |
| v2 | A2C | 0.046 | 1.2% | `rl_class/f8nsehm5` |
| **v3** | **PPO** | **0.070** | **0.0%** | 见上 lightning-10000 |

视频里：**v3 做出干净的武术动作（宽马步、抬臂出击）且不摔**；v1/v2 明显跟不住、东倒西歪。**任务越难，朴素方法和完整 PPO 的差距越大** —— 和 `1_1_g1_walk_rl` 的行走对照（0.097 对 0.070 / 0.043）互为印证。

产物（`*.mp4` gitignore，仅本地；json 入库）：

```text
code/rl/1_rl_basics/result/1_3_g1_motion_tracking/
├── track-v1-reinforce.{json,mp4}
├── track-v2-a2c.{json,mp4}
└── track-v3-ppo-original.{json,mp4}
```

权重：v1/v2 在 `DATASETS_ROOT/models/trained/xbotics_rl_beyondmimic/{beyondmimic-reinforce,beyondmimic-a2c}/`。
