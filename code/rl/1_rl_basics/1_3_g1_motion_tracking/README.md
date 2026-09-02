# G1 动作跟随（BeyondMimic）：用 PPO 让 G1 跟住一段真实动作

本模块让宇树 G1 人形**逐帧贴住一段参考动作**（由真实视频重定向到 G1，存为 `rl/1_rl_basics/data/g1_reference_motions/marshal-arts.npz`）。这是比“按速度指令行走”（见 `rl/1_rl_basics/1_1_g1_walk_rl/`）更难的任务：策略要时刻匹配参考姿态，而不只是别摔。

## 同一个任务，三个算法（和 1_1_g1_walk_rl 一一对应）

| 脚本 | 算法 | 说明 |
|---|---|---|
| `train_v1_reinforce.py` | REINFORCE | 最朴素策略梯度：无 critic、无 GAE、无裁剪 |
| `train_v2_a2c.py` | A2C | 加 critic 基线 + GAE，但不裁剪、数据只用一遍、固定学习率 |
| `train_v3_ppo.py` | **PPO（完整版）** | clip + 多轮 minibatch 复用 + KL 自适应 + value clip |

三版**预算与超参完全一致**（4096 环境 × 24 步 × 3000 迭代、熵系数 0.01），与行走线同值，
所以这一组的差距只反映算法机制。

在动作跟随这种较难的任务上，简单算法（v1/v2）会明显跟不动，凸显出 `train_v3_ppo.py` 里那套机制的必要性——这正好和 1_1 的行走对照呼应：**任务越难，朴素方法和完整 PPO 的差距越大**。

## 组件

- `env.py` — `BeyondMimicEnv`：mjlab 动作跟踪环境（actor / critic 双观测，critic 含特权信息），
  并把三个观测维度暴露给训练脚本，免得维度被写死在别处。
- `motion.py` — 参考动作 `MotionClip` 的读取与校验，以及开训前的概况预览。
- `model.py` — 三个版本共用的 `ActorCritic` 与 `compute_gae`（与行走线逐字相同）。
- `train_v1_reinforce.py` / `train_v2_a2c.py` / `train_v3_ppo.py` — 三级算法阶梯。
- `rollout.py` — 按统一口径评测三版并录像。

## 怎么跑

```bash
python rl/1_rl_basics/1_3_g1_motion_tracking/train_v1_reinforce.py  # REINFORCE
python rl/1_rl_basics/1_3_g1_motion_tracking/train_v2_a2c.py        # A2C
python rl/1_rl_basics/1_3_g1_motion_tracking/train_v3_ppo.py        # PPO
python rl/1_rl_basics/1_3_g1_motion_tracking/rollout.py             # 三版逐个录像
```

权重存到 `DATASETS_ROOT/models/trained/xbotics_rl_beyondmimic/<run>/`，曲线在 W&B（project `rl_class`）。参考动作来自 `rl/1_rl_basics/1_0_video_to_g1_reference/` 的“视频→G1 参考动作”预处理管线。
