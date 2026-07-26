"""一条命令产出一个 SO101 仿真 LeRobotDataset。

薄薄地把三步串起来：加载 v6 策略 rollout → 按外观换色重渲（replay）→ 转成 LeRobotDataset。
改下面三个值（任务 / 集数 / 任务描述），外观在 replay.py 里改，然后
`python datagen/gen_dataset.py`。策略 ckpt 由上一级 `train_v6_squint.py` 先训好。
"""

import os
from pathlib import Path

import so101_sim  # noqa: F401  注册任务

from rollout import rollout
from replay import replay, APPEARANCE
from to_lerobot import to_lerobot

TASK = "SO101ReachCube-v1"
N_EPISODES = 64
TASK_DESCRIPTION = "reach the red cube"

datasets_root = Path(os.environ["DATASETS_ROOT"])
ckpt = datasets_root / "models" / "trained" / "so101_sim_sac" / TASK / "sac_ckpt.pt"  # train_v6_squint.py 的产物
work = datasets_root / "so101_sim" / "_gen" / TASK

h5 = rollout(TASK, ckpt, N_EPISODES, work)
if APPEARANCE != "default":
    h5 = replay(TASK, h5, work / "replay.h5", work)
to_lerobot(h5, work / "dataset", task_name=TASK_DESCRIPTION)
print(f"\ndataset -> {work / 'dataset'}  (外观={APPEARANCE})")
