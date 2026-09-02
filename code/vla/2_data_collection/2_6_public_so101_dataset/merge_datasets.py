"""把 9 个单任务数据集合并成一个多任务训练集。

对应第9讲《操作数据闭环》4.6 节末尾的公开数据集路线，接在 download_dataset.py 之后。

合并用官方命令 `lerobot-edit-dataset --operation.type merge`，与 `lerobot-record`
`lerobot-replay` `lerobot-train` 同一家族。它先强校验 9 份的 fps / robot_type /
features 完全一致（不一致直接抛错，而不是静默错位），然后重映射每份的 task_index
到合并后的统一任务表、重建全局 index 与 episode_index、重算 stats。

★为什么是这条命令，而不是自己写合并：
  合并这件事有唯一官方入口，自己写就多出一个数据写入者。两个写入者产出的
  数据集在细节上必然分岔（分片边界、统计量、任务表顺序），而分岔不报错 ——
  只在训练读到不同数值时才显形，那时已经查不回来了。

★为什么不是 `MultiLeRobotDataset`（运行时挂多份、不落盘）：
  它读得进来，但训练入口吃不下。`lerobot.datasets.factory.make_dataset` 对
  非 str 的 `repo_id` 直接 `raise NotImplementedError`，多数据集那条分支是
  死代码。所以训练需要的是一份真正合并好的数据集。

不做任何上采样。9 个任务各 200-300 集本来就均匀，合并即平衡；反过来向某个任务偏斜
上采样会伤害泛化——真机上对比过，偏斜版明显不如平衡版。
"""

import json
import os
import subprocess
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset

TASKS = [
    "pick_up_a_battery_and_place_in_the_bin",
    "pick_up_a_can_and_place_in_the_bin",
    "pick_up_a_cube_and_place_in_the_bin",
    "pick_up_a_eraser_and_place_in_the_bin",
    "pick_up_a_golf_and_place_in_the_bin",
    "pick_up_a_medicine_bottle_and_place_in_the_bin",
    "pick_up_a_plush_toy_and_place_in_the_bin",
    "Stack_the_cube_on_the_can",
    "Stack_the_smaller_cube_on_the_larger_one",
]

RAW_DIR = Path(os.environ["DATASETS_ROOT"]) / "so101" / "datasets" / "raw"
MERGED_DIR = Path(os.environ["DATASETS_ROOT"]) / "so101" / "datasets" / "merged_9task"
MERGED_REPO_ID = "so101/pickplace_9task"

roots = [RAW_DIR / task for task in TASKS]
for root in roots:
    if not (root / "meta" / "info.json").exists():
        raise SystemExit(f"缺数据集：{root}")

# 列表参数按 draccus 的写法给：`--operation.repo_ids='[a,b,c]'`。
# 顺序即合并后的集顺序，两个列表必须一一对应。
print(f"合并 {len(TASKS)} 个任务 -> {MERGED_DIR}", flush=True)
subprocess.run(
    ["lerobot-edit-dataset",
     "--operation.type=merge",
     f"--operation.repo_ids=[{','.join(f'so101/{t}' for t in TASKS)}]",
     f"--operation.roots=[{','.join(str(r) for r in roots)}]",
     f"--new_repo_id={MERGED_REPO_ID}",
     f"--new_root={MERGED_DIR}"],
    check=True,
)

# 计数核对：合并后必须等于 9 份之和
info = json.loads((MERGED_DIR / "meta" / "info.json").read_text())
print(
    f"合并结果：{info['total_episodes']}ep / {info['total_frames']}frames / "
    f"{info['total_tasks']}tasks / fps={info['fps']} / robot={info['robot_type']}",
    flush=True,
)
if (info["total_episodes"], info["total_frames"], info["total_tasks"]) != (2200, 784963, 9):
    raise SystemExit("合并计数与 2200ep / 784963frames / 9tasks 不符")

# 真实加载一遍：光看 json 不算，能被训练读出来才算
dataset = LeRobotDataset(MERGED_REPO_ID, root=MERGED_DIR)
print(f"LeRobotDataset 加载 OK：num_frames={dataset.num_frames} num_episodes={dataset.num_episodes}", flush=True)
sample = dataset[0]
print(f"首帧 keys={sorted(sample)}", flush=True)
print(f"action shape={tuple(sample['action'].shape)} state shape={tuple(sample['observation.state'].shape)}", flush=True)
print("MERGE_VERIFIED", flush=True)
