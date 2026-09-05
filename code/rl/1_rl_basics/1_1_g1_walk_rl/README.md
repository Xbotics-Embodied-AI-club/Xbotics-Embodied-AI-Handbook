# G1 行走强化学习：从最朴素的策略梯度到 PPO

本模块在**同一个任务**（让宇树 G1 人形按速度指令稳定行走）上，用**三个由简到繁的算法**各训一个策略，让你亲眼看到：算法越完善，学得越稳、走得越好。三份训练脚本结构刻意保持一致，**版本之间的差异就是这一讲的知识点**。

## 任务：速度指令行走（不需要任何参考动作）

环境基于 mjlab 的 `Mjlab-Velocity-Flat-Unitree-G1`：每个回合给机器人一个随机的**目标速度**（前进 / 侧移 / 转向），奖励 = 跟上这个速度 + 保持直立，再加一些惩罚项（关节限位、动作抖动、脚滑等）。它不依赖任何示教数据，是入门强化学习最干净的载体。

- `env.py` — `G1WalkEnv`：把 mjlab 环境包成简单接口（`reset / get_observations / step`），并暴露观测维度、动作维度。actor 看普通观测，critic 额外看到脚部接触等“特权信息”。
- `model.py` — `ActorCritic`：一个普通 PyTorch 模型，actor 输出高斯动作分布，critic 估状态价值；带在线观测归一化。三个版本共用它。

## 三个版本（从简到繁）

| 脚本 | 算法 | 关键机制 | 缺什么 |
|---|---|---|---|
| `train_v1_reinforce.py` | REINFORCE | 策略梯度 + 回报基线 | 无 critic、无 GAE、无裁剪、数据只用一遍 |
| `train_v2_a2c.py` | A2C | 加 **critic 基线 + GAE 优势** | 仍无 ratio 裁剪、单轮 minibatch、固定学习率 |
| `train_v3_ppo.py` | PPO | 加 **clip + 多轮 minibatch 复用 + KL 自适应学习率 + value clip** | —（最完整） |

每个脚本都是标准的 Lightning 结构：`Dataset`（在线采一段 rollout）→ `LightningDataModule` → `LightningModule`（算 loss）→ `trainer.fit(model, data)`。要改的超参就近写成变量，没有命令行参数层。

## 第四版（进阶）：换成 off-policy 的 SAC（讲16 §4.3）

前三版是一条 **on-policy** 演进链——每轮采一段数据、算完梯度就扔。`train_v4_sac.py` 在**同一个任务、同一份奖励、同一套四件套**下把算法换成 **off-policy 的 SAC**，专门看清 on-policy → off-policy 的分水岭。差异集中在三件事：

- **经验回放（replay buffer）**：几千个并行环境的转移汇进一个大池子、训练时随机抽样——数据从"采一段用一遍"变成"存下来反复抽"。对照 v1–v3 的 `Rollout*Dataset` 与 v4 的 `G1WalkReplayDataset`，**off-policy 的全部区别在数据这一层直接看得见**。
- **双 Q + 软目标 + 随机策略**：actor 输出一个"挤压高斯"、每步采样即探索；两个 Q 取小治高估，目标网络每步软更新。
- **自动温度 α（带下限）**：熵权重被自动调到让策略熵匹配目标熵。

配方取 FastSAC/FastTD3 里对课堂有意义的三件：并行采样进同一个 buffer、大 batch、较高的更新采样比（UTD）。三个招牌坑代码里各用一行按住：观测不归一化，actor 的 tanh 会饱和、梯度冻结（`RunningNorm`）；奖励尺度太小，熵项压过 Q 值、策略只探索不迈步（`reward_scale`）；高 UTD 下自动温度会把 α 压到接近 0、熵正则失效、动作被顶到极限而发散（梯度裁剪 + 温度下限）。

**对照结果**（评测口径统一：256 环境 × 300 步、确定性动作；PPO 用同口径重测）：

| 版本 | 算法 | 评测 reward | 摔倒率 | 备注 |
|---|---|---|---|---|
| v3 | PPO（on-policy） | 0.097 | 0.0% | 3000 迭代 ≈ 295M 环境步、约 100 分钟墙钟 |
| **v4** | **SAC（off-policy）** | **0.026** | **~1%** | 约 7.4M 环境步 / 约 4 分钟即稳定走起来 |

两张对照曲线在 `result/1_1_g1_walk_rl/`（`sac-vs-ppo-envsteps.png` / `sac-vs-ppo-walltime.png`）。要点：

- **样本效率**：SAC 用约 2–7M 环境步就到 10⁻² 的 reward 水平并稳定行走，而 PPO 的确定性策略到**最早存档的** 20M 步才测到相当水平（20M 步之前没有 checkpoint，PPO 实际何时越过这一水平未知，故这里只作定性结论"起步阶段更省交互"，不给倍数）——off-policy「每条经验反复用」在起步阶段确实更省。
- **绝对高度**：SAC 停在 10⁻² 量级（和 v1/v2 评测同档），**没追上 PPO 的 0.097**。原因有二：这套奖励是给 PPO 调的，off-policy 品味不同（讲义 §4.2 已点明"原样搬不一定最优"）；课堂版只保留最核心三件，没上 FastSAC 的 n-step / 分布式 critic 等提速件。
- **过训**：后期 SAC 动作会慢慢顶大、reward 从峰值回落（off-policy 常见不稳），所以取"会走"的最佳 checkpoint（约 iter 900）作证据。

## 怎么跑

```bash
# 三个版本各自独立训练（单卡即可）
python rl/1_rl_basics/1_1_g1_walk_rl/train_v1_reinforce.py
python rl/1_rl_basics/1_1_g1_walk_rl/train_v2_a2c.py
python rl/1_rl_basics/1_1_g1_walk_rl/train_v3_ppo.py
python rl/1_rl_basics/1_1_g1_walk_rl/train_v4_sac.py       # 进阶：off-policy SAC，约 15 分钟

# 训完后录对照视频（读各自最终 checkpoint，输出到 result/1_1_g1_walk_rl/）
python rl/1_rl_basics/1_1_g1_walk_rl/rollout.py
```

训练曲线在 W&B（project `rl_class`）；权重存到 `DATASETS_ROOT/models/trained/xbotics_rl_g1_walk/<run>/`。

> 说明：脚本默认 `max_iterations=3000`，是为了课堂能在合理时间内看出三者差距。要训出更利落的行走，把它调大（mjlab 官方配方约 3 万迭代）即可，算法本身不用改。

## 进阶：同一套 PPO 升级到“动作跟随”

会了按速度指令行走，下一步就是让 G1 **逐帧贴住一段真实动作**（武术 / 舞蹈）。那需要的算法**还是这里的 v3 PPO**，只把环境从“速度指令”换成“参考动作跟踪”。见 `rl/1_rl_basics/1_3_g1_motion_tracking/`：那里的 `train_v3_ppo.py` 就是同一套 PPO 跑动作跟随，`train_v1_reinforce.py` / `train_v2_a2c.py` 则是同样的简单版本对照——再次验证“简单算法在更难的任务上更跟不动”。
