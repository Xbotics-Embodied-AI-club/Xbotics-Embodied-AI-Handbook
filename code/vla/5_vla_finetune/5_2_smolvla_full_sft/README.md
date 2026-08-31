# 5_2 全量微调 SmolVLA → SO-101 仿真抓取（讲12）

用 `rl/3_offpolicy/3_2_so101_offpolicy` 生成的 **SO-101 仿真数据集**，从社区预训练权重
`lerobot/smolvla_base`（SmolVLM2-500M 底座 + flow-matching 动作专家）出发做**全参数微调**
（无 LoRA / 无 PEFT），再在同一仿真里评测成功率。这样就合上一整条闭环：

**仿真环境 → RL 训专家 → 专家 rollout 生成数据集 → VLA 全量微调 → 仿真评测**，全程无需真机。

## 文件

| 文件 | 作用 |
|---|---|
| `train_smolvla.sh` | 微调入口：`lerobot-train --policy.path=lerobot/smolvla_base` 全量微调 |
| `smolvla_eval.sh` | 评测入口：`lerobot-eval --env.type=so101_sim` 跑 20 episode，报 pc_success + 录像 |
| `plot_loss.py` | 从训练日志抓 loss 画收敛曲线到 `result/loss_curve.png` |
| `smolvla_finetune.ipynb` | 逐行走读：命令 + 机制讲解 + 结果展示 |
| `result/` | 训练与评测的产物落点（loss 曲线、成功率、rollout 抽帧）；跑完上面两个脚本才会生成 |

## 数据准备

数据集是一个标准 `LeRobotDataset`，由 `rl/3_offpolicy/3_2_so101_offpolicy/datagen/gen_dataset.py`
产出（SAC 专家在 ReachCube 上 rollout，转成 LeRobot 格式），落
`$DATASETS_ROOT/so101_sim/_gen/SO101ReachCube-v1/dataset`。

> `SO101ReachCube-v1` 是 `so101_sim` 重构前 vendored squint 的单相机任务，现已下线（换成了
> KIT 双相机的 `SO101PickPlaceCube40-v1` 等三个分发场景）。这份数据集与训出的 checkpoint
> 是**已存在于共享存储上的历史产物**，路径按原样保留，不随环境改名——重新生成数据集需要先在
> 现存场景上把 `rl/3_offpolicy/3_2_so101_offpolicy` 的 RL 阶梯重新训通（该模块正等待整训，
> 见其 README）。

- 单目相机 `observation.images.base_camera`（128×128），6 维关节 `observation.state` / `action`，20 fps。
- 想要更多集数：改 `gen_dataset.py` 里的 `N_EPISODES` 重跑一条命令即可（前提是 `SO101ReachCube-v1`
  仍能注册；重构后的 `so101_sim` 已不再提供它，见上）。

## 训练

```bash
cd code
uv sync --extra gpu_x86
bash vla/5_vla_finetune/5_2_smolvla_full_sft/train_smolvla.sh
```

**关键点 —— `rename_map` 对齐相机名**：`smolvla_base` 预训练权重按三路相机 `camera1/2/3`
命名，本仿真数据集只有一路 `base_camera`。把 `base_camera` 改名映射到 `camera1` 即可对齐；
缺 `camera2/3` 允许——特征校验只要求数据提供的相机是权重期望相机的**子集**。
不做这步会直接报 `Feature mismatch`。

产物落 `$DATASETS_ROOT/models/trained/so101_sim_smolvla/SO101ReachCube-v1/`（不进 git）。
画收敛曲线：`uv run python vla/5_vla_finetune/5_2_smolvla_full_sft/plot_loss.py`。

## 评测

```bash
bash vla/5_vla_finetune/5_2_smolvla_full_sft/smolvla_eval.sh
```

`so101_sim` 评测环境由 `code/platform/lerobot` 的 0004 补丁注册；评测端同样用 `rename_map`
把环境输出的 `base_camera` 映射到 checkpoint 的 `camera1`。

> `--env.task` 现在填的是 `so101_sim` 现存的 `SO101PickPlaceCube40-v1`（ReachCube 已下线，
> 传给仿真器的环境 id 必须是它能注册的场景）。这个 checkpoint 是在 ReachCube 数据上训的，
> 换到 PickPlaceCube40 上评测属于任务不匹配，pc_success≈0 是预期结果，不代表策略退化。

## 结果

ReachCube（"reach the red cube"），16 集 / 800 帧，batch 64，20000 步全量微调
（该任务已下线，见上方数据准备一节；下方结果是训练当时留下的历史记录）。

| 指标 | 数值 |
|---|---|
| 最终 flow-matching loss | 跑完 `plot_loss.py` 后见 `result/loss_curve.png` |
| 评测 episode 数 | 20 |
| pc_success | 跑完 `smolvla_eval.sh` 后见 `result/eval_smolvla/eval_info.json` |

> `result/` 下的产物**不随仓库分发**（大文件不进 git）：收敛曲线来自 `plot_loss.py`，
> rollout 视频与逐 episode 结果来自 `smolvla_eval.sh`，都要自己跑一遍才有。

> 环境说明：本仓库统一走 `code/pyproject.toml` 的 uv 环境。若你的 venv 不在默认位置，
> 把 `UV_PROJECT_ENVIRONMENT` 指过去，上面的 `uv run` 命令即可原样复用。
