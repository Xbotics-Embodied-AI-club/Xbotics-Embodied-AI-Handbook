# 3_2 SO101 连续控制：DDPG → TD3 → SAC → 视觉分布式 SAC

第16讲的连续控制阶梯，承接 3_1 的值学习地基（CartPole 上的 Q-learning/DQN 都是
离散动作，对每个动作打分再 argmax）。SO101 机械臂的动作是连续的，argmax 没法穷举了，
四级由简到繁依次解决"怎么在连续动作空间里做 off-policy 学习"这个问题：

| 级别 | 文件 | 相对上一级的核心变化 |
|---|---|---|
| v3 DDPG | `train_v3_ddpg.py` | 确定性策略梯度：Actor 直接吐动作，沿 Q 的梯度往上爬 |
| v4 TD3 | `train_v4_td3.py` | 治 DDPG 的 Q 高估：双 Q 取 min + 延迟策略更新 + 目标策略平滑 |
| v5 SAC | `train_v5_sac.py` | 治 DDPG/TD3 的探索不足：随机策略 + 最大熵 + 自动温度，成功率从 0 → 0.99 |
| v6 视觉分布式 SAC | `train_v6_squint.py` | 治 SAC 标量 Q 样本效率有限：Critic 换成 C51 分布式 + 输入换成 16px 视觉，success≈0.99 但更快更稳 |

四个文件**各自自包含**（本级的网络 `nn.Module` + `LightningModule` 更新 + 回放池 +
`trainer.fit` 都在同一个文件里，无共享 `model.py`），阶梯对照的重点就在每一级"改了
什么"，diff 直接体现在文件里，不用来回跳文件找。SO101 环境不在本模块定义，统一从
`platform/so101_sim` 消费（`state_rl_env`/`visual_rl_env`）——只写/走算法侧。

```bash
cd code
python rl/3_offpolicy/3_2_so101_offpolicy/train_v3_ddpg.py    # v3/v4/v5 可直接跑
python rl/3_offpolicy/3_2_so101_offpolicy/train_v6_squint.py  # ⚠️ 现在跑不起来，见下
```

> ⚠️ **v6 现在直接跑会报错。** `train_v6_squint.py` 的 `CNNEncoder` 与 `ReplayBuffer`
> 按单相机 **3 通道**写死，而 `platform/so101_sim` 现存的三个分发场景都是双相机
> （`top` + `wrist`）**6 通道**输出，通道数对不上。v3/v4/v5 走 `state_rl_env`（只吃关节
> 状态、不过渲染管线），不受影响。要跑 v6 得先把编码器与回放池改成吃 6 通道，
> 见文末「待重新整训」。

v3/v4/v5 从关节状态（`obs_mode="state"`）学习，公平预算 500 iter 对照：DDPG 全程震荡
（mean_reward≈0.28，success_once 始终 0）、TD3 治住震荡但仍是 0（mean_reward≈0.52）、
SAC 换随机策略 + 最大熵后 success_once 从 iter~480 起稳定落在 0.93～0.99——是这条阶梯
上第一次真正意义的"解决"，也印证了探索不足才是 DDPG/TD3 学不动的病根。v6 换视觉输入 +
C51 分布式 Critic，success≈0.99，且能顺手产 VLA 课要的数据。

> 上面这组对照数据是在已下线的单相机 `SO101ReachCube-v1` 任务上跑出来的，四个文件现在
> 的 `TASK` 已改指向 `platform/so101_sim` 现存的 KIT 分发场景（`SO101PickPlaceCube40-v1`
> 等）；重跑不会复现这组数值，本模块的算法阶梯待重新整训（见文末说明）。

## v6 顺带产出的 VLA 数据：`datagen/`

v6 训好的策略不只是教学终点，rollout 出的成功轨迹换个外观就是 VLA 课用的仿真数据，
独立收在 `datagen/`（不与训练算法混在一起）。VLA 课讲9、讲12 会先给学员看这批仿真
数据，只点明"是视觉 RL 从零训出来的"，把来源埋成伏笔；到本模块（讲16）才把 SAC 一路
升级到 squint 的完整算法讲透，学员这时能自己跑通 `datagen/` 复现数据来源，前后呼应
形成一个完整 callback。

| 文件 | 做什么 |
|---|---|
| `datagen/rollout.py` | 加载 `train_v6_squint.py` 训好的 `sac_ckpt.pt`（`{"encoder","actor"}`）跑成功轨迹，`RecordEpisode` 录成 h5（128px 图 + 完整 `env_states`；策略看的是 16px 降采样视图） |
| `datagen/replay.py` | 回放 `env_states` **重渲成目标外观/相机**（黑臂/白底贴近真机）。相机位姿/FOV/外观内联，真机标定后改这里再重跑即可，不必重训 |
| `datagen/to_lerobot.py` | h5 → LeRobotDataset：复用 ManiSkill 官方 convert，仅补 `noisy_qpos→qpos` 别名 |
| `datagen/gen_dataset.py` | 一条命令串起 rollout → replay → to_lerobot |

```bash
cd code
python rl/3_offpolicy/3_2_so101_offpolicy/train_v6_squint.py       # 先训 v6，产 sac_ckpt.pt
python rl/3_offpolicy/3_2_so101_offpolicy/datagen/gen_dataset.py   # 再产数据集（改内联 TASK；外观在 datagen/replay.py）
```

> ⚠️ 这条链现在**整条都跑不通**：第一步 v6 因上面那个 3/6 通道问题报错，产不出
> `sac_ckpt.pt`；第二步 `datagen/rollout.py` 直接 `from train_v6_squint import CNNEncoder`
> 复用同一个 3 通道编码器，就算有 ckpt 也会在同一处挂。两处要一起改成按环境实际
> 通道数构造。

产物（ckpt / h5 / 数据集）统一落 `DATASETS_ROOT` 下，不入代码仓。

## 结果

- v3/v4/v5/v6 四级同 `SO101ReachCube-v1` 阶梯对照见上；v6 从零训到
  `success_once ≈ 0.99`（该任务已下线，历史结果不会重跑复现，见上方说明）。
- `datagen/` 全链产出 64 集 / 3200 帧 / 128px LeRobotDataset（lerobot 0.5.1 加载通过）；
  黑臂白底换色目视核验通过。

## 待重新整训

`so101_sim` 重构后三个分发场景改为 KIT 双相机（`SO101PickPlaceCube40-v1` 等），
上面记录的阶梯对照结果全部来自已下线的单相机 `SO101ReachCube-v1`。四个训练脚本已
机械收敛到新接口（`state_rl_env`/`visual_rl_env`，`TASK = "SO101PickPlaceCube40-v1"`），
但尚未在新场景上重新跑出对照数据——v6 现在**在新场景上直接跑会报错**：`CNNEncoder`
第一层是 `nn.Conv2d(3, 32, 4, stride=2)`、`ReplayBuffer` 的 `rgb` 缓冲也按 3 通道开，
而 `visual_rl_env` 经 `FlattenRGBDObservationWrapper` 把 top/wrist 两路 RGB 沿通道维
拼成 6 通道。要重训得先把这两处改成按环境实际通道数构造（`run_training` 已经用
`observation_space["state"].shape[-1]` 推状态维，图像通道数同法可得），`datagen/rollout.py`
复用同一个编码器，要一起改。`ColorJitterWrapper` 对 6 通道的行为也要顺带确认一遍。
本模块的算法阶梯待与真机 HIL-SERL（`3_3`）一起重新整训。
