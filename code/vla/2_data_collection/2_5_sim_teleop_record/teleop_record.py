"""无真机兜底线：用 SO-101 仿真器 + 键盘遥操，采出与实物线**同一条命令**产的数据集。

对应第9讲《操作数据闭环》4.5 节「仿真方案：SO-101 仿真器键盘 / 手柄」。

前一模块 `2_3_teleop_record` 用真的主从臂 + `lerobot-record` 采数据。没有机械臂时，
本模块**不换命令、只换机器人**：仿真器已经登记成一个 lerobot 机器人，于是

    真机： lerobot-record --robot.type=so101_follower --teleop.type=keyboard ...
    仿真： lerobot-record --robot.type=so101_sim      --teleop.type=keyboard ...

两条命令除 `--robot.type` 之外逐字相同。这一点是关键：**口径不需要事后对齐，
因为两边走的是同一条代码路径** —— 同一个 `lerobot-record`、同一个键盘遥操器、
同一个数据集写入器。单位、帧率、动作语义、字段名都没有第二个实现，也就无处走偏。

早先这里是另一套东西：自己起 ManiSkill 环境、自己接 pynput、用 ManiSkill 的
`RecordEpisode` 录 h5、再用 `convert_to_lerobot` 转格式。那条平行实现与真机线差了
四项 —— 归一化增量而不是绝对位置、原生弧度而不是真机口径、20fps 而不是 30、
128×128 而不是标定的 640×480 —— 而它的注释还写着「与真机数据集逐字段一致」。
四项里任何一项都不报错，只让下游静默学错。**平行实现是这些差异的根源，不是它们的表现。**

★ `--robot.discover_packages_path=so101_sim` 是 lerobot 的插件发现口：它 import 本包
  从而完成机器人注册。lerobot 侧不需要为此改任何代码。
"""

import os
import subprocess
from pathlib import Path

# 场景：换任务改这一行。三个分发场景见 so101_sim 的 README。
TASK = "SO101PickPlaceCube40-v1"
# 数据集标识与落点。与真机线同一套命名，下游分不出这批数据来自真机还是仿真。
REPO_ID = "so101_sim/teleop_cube40"
SINGLE_TASK = "pick up the cube and place it in the bin"
# 采多少集、每集多长。与真机线的默认一致。
NUM_EPISODES = 5
EPISODE_TIME_S = 20
RESET_TIME_S = 5


def record_cmd(root: Path) -> list[str]:
    """拼出那条标准命令。

    Args:
        root: 数据集落盘目录。

    Returns:
        可直接交给 subprocess 的命令行。
    """
    return [
        "lerobot-record",
        # ── 机器人：真机换成 --robot.type=so101_follower --robot.port=/dev/tty... 即可 ──
        "--robot.type=so101_sim",
        "--robot.discover_packages_path=so101_sim",
        f"--robot.task={TASK}",
        # ── 遥操：lerobot 自带的键盘遥操器，与真机线用的是同一个 ──
        "--teleop.type=keyboard",
        # ── 数据集：fps 不写死在这里，用真机那条线的同一个值 ──
        f"--dataset.repo_id={REPO_ID}",
        f"--dataset.root={root}",
        f"--dataset.single_task={SINGLE_TASK}",
        f"--dataset.num_episodes={NUM_EPISODES}",
        f"--dataset.episode_time_s={EPISODE_TIME_S}",
        f"--dataset.reset_time_s={RESET_TIME_S}",
        "--dataset.push_to_hub=false",
        "--display_data=true",
    ]


def main() -> int:
    """跑那条命令。

    Returns:
        子进程的退出码。
    """
    root = Path(os.environ["DATASETS_ROOT"]) / "so101_sim" / "_teleop" / TASK
    cmd = record_cmd(root)
    print("键盘遥操采集，走的是驱动真机的那条命令：\n")
    print("    " + " \\\n        ".join(cmd) + "\n")
    # check=False：采集被 Ctrl-C 或 Esc 打断是正常收尾，不该当异常抛出；
    # 退出码原样透出，调用方要判就自己判。
    return subprocess.run(cmd, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
