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
