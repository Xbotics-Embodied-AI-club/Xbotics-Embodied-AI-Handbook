# 仿真遥操采集（无真机兜底线）

没有真机械臂时，用 SO-101 **仿真器**顶替 `2_3` 的真机 Leader–Follower 臂。

关键是**不换命令、只换机器人**：仿真器已经登记成一个 lerobot 机器人，于是

```bash
# 真机
lerobot-record --robot.type=so101_follower --teleop.type=keyboard --dataset.repo_id=...
# 仿真
lerobot-record --robot.type=so101_sim      --teleop.type=keyboard --dataset.repo_id=...
```

两条命令除 `--robot.type` 之外逐字相同 —— 同一个 `lerobot-record`、同一个键盘遥操器、
同一个数据集写入器。所以单位、帧率、动作语义、字段名**不需要事后对齐**，
它们没有第二个实现，也就无处走偏。下游 ACT / VLA 读进来分不出数据来自真机还是仿真。

> 这一节曾是另一套东西：自己起 ManiSkill 环境、自己接 pynput、用 ManiSkill 的
> `RecordEpisode` 录 h5、再 `convert_to_lerobot` 转格式。那条平行实现与真机线差了四项 ——
> 归一化增量而非绝对位置、原生弧度而非真机口径、20fps 而非 30、128×128 而非标定的
> 640×480 —— 四项都不报错，只让下游静默学错。**平行实现是这些差异的根源。**

## 文件

- `teleop_record.py` — 把那条标准命令拼出来并执行；参数（场景 / 集数 / 时长）在文件顶部
- `teleop_record.ipynb` — 同一份内容的中文分节讲解版
- `README.md` — 本文件

## 按键

键位由 lerobot 自带的 `keyboard` 遥操器定义，真机与仿真完全一致；具体键位随 lerobot
版本演进，运行时窗口里会打印当前键位表，以那里为准 —— 这里不抄一份，抄下来就会过期。

## 运行

在 `code/` 下（环境走统一 `pyproject.toml` 的 `gpu_x86` extra）：

```bash
uv run python vla/2_data_collection/2_5_sim_teleop_record/teleop_record.py
```

想换场景改 `TASK`（三个 SO-101 分发场景之一）；想改集数或每集时长改 `NUM_EPISODES`
与 `EPISODE_TIME_S`。这些字段名与真机线的 `lerobot-record` 参数同名。

## 输出

数据集落 `DATASETS_ROOT/so101_sim/_teleop/<TASK>/`，不入代码仓。
目录即标准 `LeRobotDataset`（`data/` parquet + `videos/` mp4 + `meta/` info/stats/episodes）：

| 字段 | 取值 |
|---|---|
| `action` · `observation.state` | f32×6，五个臂关节**度**、夹爪**0~100 行程百分比** |
| `action` 语义 | 绝对关节位置目标 |
| 相机 | `observation.images.top` 与 `observation.images.wrist`，480×640 |
| `fps` | 30 |
| `robot_type` | `so_follower` |

夹爪与臂关节单位不同，是因为真机就是这样：`lerobot-record` 走 `so_follower`，
而它把 gripper 写死为 `MotorNormMode.RANGE_0_100`。仿真沿用同一套，不是巧合也不是妥协。
