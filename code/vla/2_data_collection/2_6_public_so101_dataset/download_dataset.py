"""从 ModelScope 下载 SO-ARM101 pick-place 数据集的 9 个任务子集。

数据源 zhuzhuangtian/so101-pick-place-tasks 是 LeRobot v3.0 格式，9 个任务各自成一份完整
数据集、直接放在仓库根（README 里写的 merged_datasets/ 子目录实际不存在，别照着找）。

下载完逐任务核对 meta/info.json：codebase_version、集数、帧数都要和标称对上，
对不上就停——后面合并与训练全建立在这些计数上。
"""

import os
from pathlib import Path

from modelscope.hub.snapshot_download import snapshot_download

DATASET_ID = "zhuzhuangtian/so101-pick-place-tasks"

# 标称集数 / 帧数（来自数据集卡片，逐任务核对用）
EXPECTED = {
    "pick_up_a_battery_and_place_in_the_bin": (200, 56295),
    "pick_up_a_can_and_place_in_the_bin": (250, 82792),
    "pick_up_a_cube_and_place_in_the_bin": (300, 106085),
    "pick_up_a_eraser_and_place_in_the_bin": (300, 109899),
    "pick_up_a_golf_and_place_in_the_bin": (250, 77409),
    "pick_up_a_medicine_bottle_and_place_in_the_bin": (300, 116342),
    "pick_up_a_plush_toy_and_place_in_the_bin": (200, 80811),
    "Stack_the_cube_on_the_can": (200, 74425),
    "Stack_the_smaller_cube_on_the_larger_one": (200, 80905),
}

# 数据落训练机的大文件工作区（集群要求大文件放 /work，不放 home）
DOWNLOAD_DIR = Path(os.environ["DATASETS_ROOT"]) / "so101" / "datasets" / "raw"

DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

# 只要 9 个任务目录，仓库根的 .gitattributes / 检查脚本等不要
allow_patterns = [f"{task}/*" for task in EXPECTED]

print(f"下载 {DATASET_ID} -> {DOWNLOAD_DIR}", flush=True)
local_dir = snapshot_download(
    DATASET_ID,
    repo_type="dataset",
    local_dir=str(DOWNLOAD_DIR),
    allow_patterns=allow_patterns,
)
print(f"下载完成：{local_dir}", flush=True)

# 逐任务核对 info.json
import json

total_episodes = 0
total_frames = 0
for task, (want_ep, want_fr) in EXPECTED.items():
    info = json.loads((DOWNLOAD_DIR / task / "meta" / "info.json").read_text())
    version = info["codebase_version"]
    got_ep = info["total_episodes"]
    got_fr = info["total_frames"]
    ok = version == "v3.0" and got_ep == want_ep and got_fr == want_fr
    print(f"{'OK ' if ok else 'BAD'} {task}: {version} {got_ep}ep {got_fr}frames", flush=True)
    if not ok:
        raise SystemExit(f"{task} 与标称不符：期望 v3.0 {want_ep}ep {want_fr}frames")
    total_episodes += got_ep
    total_frames += got_fr

print(f"\n合计 {total_episodes} episodes / {total_frames} frames", flush=True)
if (total_episodes, total_frames) != (2200, 784963):
    raise SystemExit("合计与标称 2200ep / 784963frames 不符")
print("DOWNLOAD_VERIFIED", flush=True)
