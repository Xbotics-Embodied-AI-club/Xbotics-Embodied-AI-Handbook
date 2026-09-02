# RL 实验代码

强化学习课程段的全部演示代码。按**主题组**组织，每个组是一讲的完整材料（代码 +
数据 + 结果），整个组目录打包拿走即可运行与阅读。

## 目录架构

```
rl/
├── 1_rl_basics/            # 组1 RL 基础：同任务三算法对照（REINFORCE→A2C→PPO）
│   ├── 1_0_video_to_g1_reference/   #   人类视频 → G1 参考动作（前置工具链，只产数据）
│   ├── 1_1_g1_walk_rl/     #   G1 行走
│   ├── 1_3_g1_motion_tracking/      #   G1 动作跟随（BeyondMimic）
│   ├── data/               #   组内共享数据（按内容命名）
│   └── result/             #   各模块结果（json 摘要 + 演示视频）
├── 2_grpo_posttraining/    # 组2 GRPO 后训练：让模型自我提升
│   ├── 2_1_grpo_vlm_counting/       #   GRPO 微调小 VLM 学数数
│   ├── data/
│   └── result/
└── 3_offpolicy/            # 组3 Off-policy：值学习地基→连续控制→真机落地，六级依次提升
    ├── 3_1_cartpole_value_rl/       #   值学习入门：Q-learning→DQN（CartPole 贯穿两级）
    ├── 3_2_so101_offpolicy/         #   SO101 连续控制：DDPG→TD3→SAC→视觉分布式 SAC（datagen/ 顺手产 VLA 数据）
    └── 3_3_hilserl_so101/           #   SO-101 真机人在环 HIL-SERL（待建）
```

> 组1 与组3 编号例外：组内序号 = **教学顺序**（非建立顺序）。
> 组3 是为体现"由简到繁"的算法递进；组1 是因为 `1_0` 只产数据、不产结论，
> 它是动作跟随实验的前置，排在算法阶梯之前才对得上讲义的一级标题。

约定：

- **双编号 `<组>_<序>_<名字>`**：组号是主题、组内序号默认是建立顺序，只增不改；
  例外见上——序号改动只在"讲义结构与代码结构对不上"时发生，且两边同轮改完。
- **一个一级标题对一个模块**：讲14 的 §4↔`1_0`、§5/§6/§7↔`1_1` 的 v1/v2/v3、§8↔`1_3`。
  读者读到哪一节，打开哪一个目录即可；这层对应写在讲义 4.5 节的代码地图里。
- 模块内代码引组内数据一律用相对路径（`../data/...`），组目录整体迁移后仍可运行。
- 面向课堂走读的训练/评测入口都配有同名 `.ipynb`（中文分节，与 `.py` 内容一致）。
- 大体积产物（训练 checkpoint、预处理中间量）不入库，统一落
  `DATASETS_ROOT/models/trained/` 下（环境变量由 `.env` 提供）。

## 课程讲次映射

| 组 / 模块 | 内容 | 课程对应 |
|---|---|---|
| `1_rl_basics/1_0_video_to_g1_reference` | 人类视频 → G1 参考动作工具链 | 讲14 §4 |
| `1_rl_basics/1_1_g1_walk_rl` | REINFORCE→A2C→PPO 让 G1 行走 | 讲14 §5 / §6 / §7 |
| `1_rl_basics/1_3_g1_motion_tracking` | 同三算法对照 · G1 动作跟随 | 讲14 §8 |
| `2_grpo_posttraining/2_1_grpo_vlm_counting` | GRPO 微调小 VLM 学数数 | 讲15 |
| `3_offpolicy/3_1_cartpole_value_rl` | 值学习入门 Q-learning→DQN（CartPole 贯穿） | 讲16 |
| `3_offpolicy/3_2_so101_offpolicy` | SO101 连续控制 DDPG→TD3→SAC→视觉分布式 SAC；`datagen/` 顺手产 VLA 数据 | 讲16（讲14 作案例引用） |
| `3_offpolicy/3_3_hilserl_so101`（待建） | 真机人在环学接触型任务（HIL-SERL） | 讲16 |

> 讲次调整只改这张表，不动目录。

## 环境

全部模块共用 `code/pyproject.toml` 的统一 uv 环境，GPU 工作站统一用 `gpu_x86` extra（组1 mjlab 训练、组2 GRPO 全包）：

```bash
cd code
uv sync --extra gpu_x86
```
