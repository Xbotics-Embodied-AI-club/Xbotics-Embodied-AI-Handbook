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

## 怎么跑

```bash
# 三个版本各自独立训练（单卡即可）
python rl/1_rl_basics/1_1_g1_walk_rl/train_v1_reinforce.py
python rl/1_rl_basics/1_1_g1_walk_rl/train_v2_a2c.py
python rl/1_rl_basics/1_1_g1_walk_rl/train_v3_ppo.py

# 训完后录对照视频（读各自最终 checkpoint，输出到 result/1_1_g1_walk_rl/）
python rl/1_rl_basics/1_1_g1_walk_rl/rollout.py
```

训练曲线在 W&B（project `rl_class`）；权重存到 `DATASETS_ROOT/models/trained/xbotics_rl_g1_walk/<run>/`。

> 说明：脚本默认 `max_iterations=3000`，是为了课堂能在合理时间内看出三者差距。要训出更利落的行走，把它调大（mjlab 官方配方约 3 万迭代）即可，算法本身不用改。

## 进阶：同一套 PPO 升级到“动作跟随”

会了按速度指令行走，下一步就是让 G1 **逐帧贴住一段真实动作**（武术 / 舞蹈）。那需要的算法**还是这里的 v3 PPO**，只把环境从“速度指令”换成“参考动作跟踪”。见 `rl/1_rl_basics/1_3_g1_motion_tracking/`：那里的 `train_v3_ppo.py` 就是同一套 PPO 跑动作跟随，`train_v1_reinforce.py` / `train_v2_a2c.py` 则是同样的简单版本对照——再次验证“简单算法在更难的任务上更跟不动”。
