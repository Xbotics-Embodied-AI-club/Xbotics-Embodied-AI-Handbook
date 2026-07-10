"""把 h5 轨迹转成 LeRobotDataset。

直接复用 ManiSkill 官方 `convert_to_lerobot`（未改一行）。唯一要搭的桥：squint 的机器人
状态存在 `obs/agent/noisy_qpos`（带 sim2real 噪声），而官方转换只找 `qpos`——转换前给
h5 补一个 `qpos` 别名（同一份数据）即可，官方脚本照常工作。
"""

import os
import subprocess
import sys
from pathlib import Path

import h5py

FPS = 20
IMAGE_SIZE = "128"


def _alias_qpos(h5_path: Path) -> None:
    with h5py.File(h5_path, "a") as f:
        for traj in f:
            agent = f[traj].get("obs/agent")
            if agent is not None and "qpos" not in agent and "noisy_qpos" in agent:
                agent["qpos"] = agent["noisy_qpos"][:]


def to_lerobot(h5_path: Path, out_dir: Path, task_name: str) -> Path:
    _alias_qpos(h5_path)
    subprocess.run(
        [
            sys.executable, "-m", "mani_skill.trajectory.convert_to_lerobot",
            f"--traj-path={h5_path}",
            f"--output-dir={out_dir}",
            f"--fps={FPS}",
            f"--image-size={IMAGE_SIZE}",
            f"--task-name={task_name}",
        ],
        check=True,
    )
    print(f"LeRobotDataset -> {out_dir}")
    return out_dir


TASK = "SO101ReachCube-v1"

if __name__ == "__main__":
    work = Path(os.environ["DATASETS_ROOT"]) / "so101_sim" / "_gen" / TASK
    to_lerobot(work / "rollout.h5", work / "dataset", task_name="reach the red cube")
