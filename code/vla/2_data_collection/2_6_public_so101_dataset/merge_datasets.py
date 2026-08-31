"""把 9 个单任务数据集合并成一个多任务训练集。

用 lerobot 官方的 aggregate_datasets：它会先 validate_all_metadata 强校验 9 份的 fps /
robot_type / features 完全一致（不一致直接抛错，而不是静默错位），然后重映射每份的
task_index 到合并后的统一任务表、重建全局 index 与 episode_index、重算 stats。

不做任何上采样。9 个任务各 200-300 集本来就均匀，合并即平衡；反过来向某个任务偏斜
上采样会伤害泛化——这是 piper 项目上用真机验出来的（偏斜版比平衡版明显更差）。
"""

import json
import os
from pathlib import Path

from lerobot.datasets.aggregate import aggregate_datasets
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

print(f"合并 {len(TASKS)} 个任务 -> {MERGED_DIR}", flush=True)
aggregate_datasets(
    repo_ids=[f"so101/{task}" for task in TASKS],
    aggr_repo_id=MERGED_REPO_ID,
    roots=roots,
    aggr_root=MERGED_DIR,
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
