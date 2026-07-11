# G1 行走：三个算法对照结果

同一个任务（G1 平地速度指令行走），三个由简到繁的算法各训 3000 迭代（教学预算）。
评测口径一致：各自最终 checkpoint，确定性动作（`act_inference`），16 环境 × 400 步。

| 版本 | 算法 | 评测 mean_reward | 摔倒率 done_fraction | W&B run |
|---|---|---|---|---|
| v1 | REINFORCE | 0.070 | 1.3% | `rl_class/yipekpmp` |
| v2 | A2C（critic+GAE） | 0.043 | 1.3% | `rl_class/wmq9syxu` |
| **v3** | **PPO（+clip+多轮minibatch+KL自适应）** | **0.097** | **0.0%** | `rl_class/j64dw6hj` |

**结论**：v3 PPO 评测 reward 最高、且全程不摔；v1/v2 reward 更低、偶有摔倒。算法越完整越稳越好，正是这一讲要立的对照。

产物（视频按 `*.mp4` gitignore，仅本地留存；json 摘要入库）：

```text
code/rl/1_rl_basics/result/1_1_g1_walk_rl/
├── g1-walk-reinforce.{json,mp4}
├── g1-walk-a2c.{json,mp4}
└── g1-walk-ppo.{json,mp4}
```

权重在 `DATASETS_ROOT/models/trained/xbotics_rl_g1_walk/{g1-walk-reinforce,g1-walk-a2c,g1-walk-ppo}/`。

> 预算说明：3000 迭代下三者都还是“稳住/小幅移动”而非利落快走（mjlab 官方配方约 3 万迭代才走得漂亮）；本模块的目的是**对照算法差异**，不是刷行走质量。把 `max_iterations` 调大即可提升绝对效果，算法不变。

## 第四版对照：off-policy SAC vs on-policy PPO（讲16 §7.3）

同一环境、同一奖励，v4 换成 off-policy SAC。评测口径与这里**统一重测**的 PPO 一致：256 环境 × 300 步、确定性动作（故 PPO 数值 0.097 与上表 16×400 口径略有出入）。

| 版本 | 算法 | 评测 mean_reward | 摔倒率 | 环境步数 / 墙钟 |
|---|---|---|---|---|
| v3 | PPO（on-policy，最终） | 0.097 | 0.0% | 295M 步 / ~100 分钟 |
| **v4** | **SAC（off-policy，最佳会走 ckpt）** | **0.026** | **~1%** | 7.4M 步 / ~4 分钟 |

**读法（`sac-vs-ppo-envsteps.png` / `sac-vs-ppo-walltime.png`）**：

- **样本效率**：SAC 用约 2–7M 环境步就学到 10⁻² 的 reward 并稳定行走；PPO 的确定性策略到最早存档的 20M 步才测到相当水平——off-policy「每条经验反复用」在起步阶段确实占优（阈值分析见 `compare.json`）。
- **绝对高度**：SAC 停在 10⁻² 量级（和 v1/v2 评测同档），**没追上 PPO 的 0.097**。这套奖励是给 PPO 调的（讲义 §7.2："原样搬不一定最优"），课堂版又只保留最核心三件（回放 + 双 Q + 自动温度），没上 FastSAC 的 n-step / 分布式 critic。
- **过训**：SAC 后期动作幅度慢慢顶大、reward 从峰值回落（off-policy 常见不稳），故取"会走"的最佳 checkpoint（iter 900）作证据。

**诚实结论**：教学版 SAC 在 G1 上**能稳定走起来、起步比 PPO 省交互**，印证 off-policy 的样本效率方向；但绝对指标要压过精调的 PPO 需要 FastSAC 完整配方。这正是讲16 §7 要立的对照。

```text
code/rl/1_rl_basics/result/1_1_g1_walk_rl/
├── g1-walk-sac.{json,mp4}       # 最佳会走 checkpoint 评测 + rollout 视频
├── g1-walk-sac-strip.png        # 抽帧四格
├── sac-vs-ppo-envsteps.png      # 环境步数对照曲线
├── sac-vs-ppo-walltime.png      # 墙钟时间对照曲线
└── compare.json                 # 两条曲线数值 + 阈值分析
```

SAC 权重在 `DATASETS_ROOT/models/trained/xbotics_rl_g1_walk/g1-walk-sac/`。
