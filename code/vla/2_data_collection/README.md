# 组2 · 操作数据闭环

模仿学习的上限由数据决定。本组按「接硬件 → 采数据 → 读懂数据」（环境统一走 `code/pyproject.toml`，`2_1` 空号已废弃）的顺序走完
一条采集闭环：

| 模块 | 干什么 |
|---|---|
| `2_2_so101_setup/` | SO-101 主从臂串口、相机的 udev 绑定脚本 |
| `2_3_teleop_record/` | 主从臂遥操作、`lerobot-record` 录制与 Rerun 可视化 |
| `2_4_lerobot_dataset/` | LeRobot 数据集格式走读（parquet/mp4/meta 三支柱） |
| `2_5_sim_teleop_record/` | 无真机兜底线：SO-101 仿真器 + 键盘遥操，采出与实物线同格式的数据集 |
| `2_6_public_so101_dataset/` | 公开数据线：下载 9 个任务的公开 SO-101 数据集、合并成多任务训练集、开训前审计 |

采出的数据集按 repo_id 落 `$HF_LEROBOT_HOME`，不进仓库。
