# 公开 SO-101 数据集：下载 → 合并 → 审计

对应第9讲《操作数据闭环》4.6 节末尾那条路：手上还没有自采数据时，先用一份别人采好的
公开数据集，把「拿到数据 → 合并 → 开训前检查」整条链走通。

数据源是 ModelScope 上的 `zhuzhuangtian/so101-pick-place-tasks`：SO-ARM101 真机遥操采的
9 个 pick-place 任务，LeRobot v3.0 格式，合计 2200 集 / 784963 帧、两路相机（top + wrist）。

## 文件

| 文件 | 干什么 |
|---|---|
| `download_dataset.py` | 拉 9 个任务子集，逐份核对 `meta/info.json` 的版本 / 集数 / 帧数，对不上直接停 |
| `merge_datasets.py` | 用官方 `aggregate_datasets` 合成一个多任务训练集，合并后再核一遍计数并真加载一次 |
| `audit_dataset.py` | 开训前的四项审计，全过才开训 |

## 运行

在 `code/` 下按顺序跑（数据落 `DATASETS_ROOT`，不进仓库）：

```bash
uv run python vla/2_data_collection/2_6_public_so101_dataset/download_dataset.py
uv run python vla/2_data_collection/2_6_public_so101_dataset/merge_datasets.py
uv run python vla/2_data_collection/2_6_public_so101_dataset/audit_dataset.py
```

## 审计查什么

1. **动作口径**：逐关节比 `action` 与 `observation.state` 的平均差。差值都在噪声量级就是绝对
   口径，某个关节冒出固定偏移说明录制时开了锚定——这正是第9讲 2.2.2 节讲的那个坑，口径
   判错了部署就会把臂推到训练分布外。
2. **任务配比与集长**：集长的 min / median / max，以及短于 50 帧、装不下一个 action chunk 的集。
3. **stats 是否退化**：`std` 接近 0 的维度会让归一化除零或把整维压成常量，必须开训前发现。
4. **dataloader 真迭代**：光看 `meta/` 不算数，要按训练那条路径真把两路视频解出来。

结果写成 `DATASETS_ROOT/so101/outputs/audit.json`，随 checkpoint 一起存档——部署侧要照同一
份口径配置。
