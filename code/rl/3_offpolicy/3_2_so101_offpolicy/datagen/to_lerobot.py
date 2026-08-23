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

# 默认 20，对应普通任务的 control_freq。`*Slow-v1` 任务是 30Hz（为对齐真机），
# 转它们的轨迹必须显式传 fps=30——标签和环境不一致会让回放倍速失真。
FPS = 20
IMAGE_SIZE = "128"


def _alias_qpos(h5_path: Path) -> None:
    with h5py.File(h5_path, "a") as f:
        for traj in f:
            agent = f[traj].get("obs/agent")
            if agent is not None and "qpos" not in agent and "noisy_qpos" in agent:
                agent["qpos"] = agent["noisy_qpos"][:]


def to_lerobot(h5_path: Path, out_dir: Path, task_name: str,
               image_size: str = IMAGE_SIZE, fps: int = FPS) -> Path:
    """转格式。`image_size` 传 "WxH"（如 KIT 双相机的 "640x480"）或单个数字（正方形）。

    `fps` 必须等于采这批轨迹的环境的 control_freq。
    """
    _alias_qpos(h5_path)
    subprocess.run(
        [
            sys.executable, "-m", "mani_skill.trajectory.convert_to_lerobot",
            f"--traj-path={h5_path}",
            f"--output-dir={out_dir}",
            f"--fps={fps}",
            f"--image-size={image_size}",
            f"--task-name={task_name}",
        ],
        check=True,
    )
    print(f"LeRobotDataset -> {out_dir}")
    return out_dir


TASK = "SO101PickPlaceCube40-v1"

if __name__ == "__main__":
    work = Path(os.environ["DATASETS_ROOT"]) / "so101_sim" / "_gen" / TASK
    to_lerobot(work / "rollout.h5", work / "dataset",
              task_name="pick up the red cube and place it in the bin")
