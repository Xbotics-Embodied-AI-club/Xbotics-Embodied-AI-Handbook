# 3_2 SO101 连续控制：DDPG → TD3 → SAC → squint 分布式 SAC

组3 讲16 的连续控制阶梯，承接 3_1 的值学习地基（CartPole 上的 Q-learning/DQN 都是
离散动作，对每个动作打分再 argmax）。SO101 机械臂的动作是连续的，argmax 没法穷举了，
四级由简到繁依次解决"怎么在连续动作空间里做 off-policy 学习"这个问题：

| 级别 | 文件 | 相对上一级的核心变化 |
|---|---|---|
| v3 DDPG | `train_v3_ddpg.py` | 确定性策略梯度：Actor 直接吐动作，沿 Q 的梯度往上爬 |
| v4 TD3 | `train_v4_td3.py` | 治 DDPG 的 Q 高估：双 Q 取 min + 延迟策略更新 + 目标策略平滑 |
| v5 SAC | `train_v5_sac.py` | 治 DDPG/TD3 的探索不足：随机策略 + 最大熵 + 自动温度，成功率从 0 → 0.99 |
| v6 squint | `train_v6_squint.py` | 治 SAC 标量 Q 样本效率有限：Critic 换成 C51 分布式 + 输入换成 16px 视觉，success≈0.99 但更快更稳 |

四个文件**各自自包含**（本级的网络 `nn.Module` + `LightningModule` 更新 + 回放池 +
`trainer.fit` 都在同一个文件里，无共享 `model.py`），阶梯对照的重点就在每一级"改了
什么"，diff 直接体现在文件里，不用来回跳文件找。SO101 环境不在本模块定义，统一从
`platform/so101_sim`（`so101_sim.make_train_env`）消费——只写/走算法侧。

```bash
cd experiments
python rl/3_offpolicy/3_2_so101_offpolicy/train_v3_ddpg.py    # 依次跑通 v3→v4→v5→v6
python rl/3_offpolicy/3_2_so101_offpolicy/train_v6_squint.py  # 改内联 TASK 换任务
```

v3/v4/v5 从关节状态（`obs_mode="state"`）学 `SO101ReachCube-v1`，公平预算 500 iter
对照：DDPG 全程震荡（mean_reward≈0.28，success_once 始终 0）、TD3 治住震荡但仍是
0（mean_reward≈0.52）、SAC 换随机策略 + 最大熵后 success_once 从 iter~480 起稳定
落在 0.93～0.99——是这条阶梯上第一次真正意义的"解决"，也印证了探索不足才是
DDPG/TD3 学不动的病根。v6 换视觉输入 + C51 分布式 Critic，success≈0.99，且能顺手
产 VLA 课要的数据。

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
cd experiments
python rl/3_offpolicy/3_2_so101_offpolicy/train_v6_squint.py       # 先训 v6，产 sac_ckpt.pt
python rl/3_offpolicy/3_2_so101_offpolicy/datagen/gen_dataset.py   # 再产数据集（改内联 TASK；外观在 datagen/replay.py）
```

产物（ckpt / h5 / 数据集）统一落 `DATASETS_ROOT` 下，不入代码仓。

## 结果

- v3/v4/v5/v6 四级同 `SO101ReachCube-v1` 阶梯对照见上；v6 squint 从零训到
  `success_once ≈ 0.99`。
- `datagen/` 全链产出 64 集 / 3200 帧 / 128px LeRobotDataset（lerobot 0.5.1 加载通过）；
  黑臂白底换色目视核验通过。
