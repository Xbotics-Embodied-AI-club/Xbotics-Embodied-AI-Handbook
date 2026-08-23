"""无真机兜底线：用 SO-101 仿真器 + 键盘遥操，采出与实物线同格式的 LeRobot 数据集。

前一模块 `2_3_teleop_record` 用真的主从臂 + `lerobot-record` 采数据；没有机械臂时，
本模块用 SO-101 仿真器顶替：人用键盘直接驱动仿真里的从臂，每一步把相机图、关节位置、
动作都记下来，最后转成和实物线一模一样的 `LeRobotDataset`（128px 图 + 6 维 state + 6 维 action），
下游 ACT / VLA 训练分不出这批数据到底来自真机还是仿真。

它和 `rl/3_offpolicy` 里那条「RL 专家自动采数据」的生产线是同一套骨架，只换了动作的来源：
那边动作由训练好的策略网络给出，这边动作由键盘实时给出——录制器、h5 结构、官方格式转换
三步完全复用，所以格式天然对齐。
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import cv2
import gymnasium as gym
import h5py

import so101_sim  # import 即向 ManiSkill 注册 3 个 SO101 仿真场景

from mani_skill.utils.wrappers.record import RecordEpisode


# ── 键盘 → 6 维关节增量动作 ────────────────────────────────────────────────
# SO-101 从臂有 6 个关节：底座旋转 shoulder_pan、大臂 shoulder_lift、小臂 elbow_flex、
# 腕俯仰 wrist_flex、腕滚转 wrist_roll、夹爪 gripper。仿真的控制模式是关节增量
# （pd_joint_target_delta_pos）：每步的动作是归一化到 [-1, 1] 的「这一步各关节朝哪个
# 方向走、走多满」。于是键盘遥操就很直接——按住某个键，就给对应关节一个固定大小的增量。
KEY_TO_JOINT = {
    "a": (0, +1), "d": (0, -1),   # 底座左转 / 右转
    "w": (1, +1), "s": (1, -1),   # 大臂抬起 / 落下
    "e": (2, +1), "q": (2, -1),   # 小臂前伸 / 收回
    "r": (3, +1), "f": (3, -1),   # 腕部上仰 / 下俯
    "t": (4, +1), "g": (4, -1),   # 腕部左滚 / 右滚
    "c": (5, +1), "v": (5, -1),   # 夹爪张开 / 闭合
}
STEP = 0.6   # 每步增量占满格的比例；调大动得更快、调小更精细


class Keyboard:
    """后台监听键盘，随时报出「此刻该发的 6 维增量动作」。

    pynput 在一个后台线程里维护「当前按住了哪些键」的集合，采集主循环每步读一次这个集合，
    把按住的键按上表翻译成关节增量。另外约定：回车键表示「这一集采完了」，Esc 表示「整轮结束」。
    """

    def __init__(self):
        from pynput import keyboard

        self._pressed = set()
        self.end_episode = False
        self.stop = False

        def on_press(key):
            char = getattr(key, "char", None)
            if char:
                self._pressed.add(char.lower())
            elif key == keyboard.Key.enter:
                self.end_episode = True
            elif key == keyboard.Key.esc:
                self.stop = True

        def on_release(key):
            char = getattr(key, "char", None)
            if char:
                self._pressed.discard(char.lower())

        keyboard.Listener(on_press=on_press, on_release=on_release).start()

    def action(self):
        a = np.zeros(6, dtype=np.float32)
        for char in list(self._pressed):
            if char in KEY_TO_JOINT:
                joint, sign = KEY_TO_JOINT[char]
                a[joint] = sign * STEP
        return a


# ── 起一个带录制器的 SO-101 仿真 ──────────────────────────────────────────
TASK = "SO101PickPlaceCube40-v1"   # 把 4cm 方块抓起来放进料盒；换场景改这一行
FPS = 20                     # 仿真控制频率，也是数据集回放帧率


def make_recorder(out_dir):
    """num_envs=1 的 SO-101 仿真，外面套 ManiSkill 的 RecordEpisode。

    每次 reset 开一集，每步自动把 128px 相机图、关节位置、动作都写进 h5。这正是 RL 数据
    生产线用的同一个录制器，所以录出来的 h5 结构一致，后面能直接喂给官方格式转换。
    """
    env = gym.make(
        TASK, num_envs=1, obs_mode="rgb+segmentation", render_mode="all",
        sim_backend="gpu", domain_randomization=False, reconfiguration_freq=1,
        sensor_configs=dict(width=128, height=128),
    )
    return RecordEpisode(
        env, output_dir=str(out_dir), save_trajectory=True, save_video=False,
        trajectory_name="teleop", max_steps_per_video=50,
    )


# ── h5 → LeRobotDataset（复用 ManiSkill 官方转换，与实物线同格式）─────────────
IMAGE_SIZE = "128"


def to_lerobot(h5_path, out_dir):
    """把录好的 h5 转成 LeRobotDataset。

    直接调用 ManiSkill 官方的 `convert_to_lerobot`（一行没改）。唯一要搭的小桥：仿真的
    关节位置存在 `obs/agent/noisy_qpos`（带 sim2real 噪声），而官方转换只认 `qpos`——转换前
    给每条轨迹补一个同名别名即可。转出的 action / observation.state / observation.images
    三支柱与真机数据集逐字段一致。
    """
    with h5py.File(h5_path, "a") as f:
        for traj in f:
            agent = f[traj]["obs/agent"]
            if "qpos" not in agent:
                agent["qpos"] = agent["noisy_qpos"][:]
    subprocess.run(
        [
            sys.executable, "-m", "mani_skill.trajectory.convert_to_lerobot",
            f"--traj-path={h5_path}",
            f"--output-dir={out_dir}",
            f"--fps={FPS}",
            f"--image-size={IMAGE_SIZE}",
            f"--task-name=pick up the cube and place it in the bin",
        ],
        check=True,
    )
    return out_dir


# ── 键盘遥操采集：reset → 逐步遥操 → 回车存一集 → 攒 N 集 → 转数据集 ──────────
N_EPISODES = 5        # 想采多少集；每集最多走 50 步，回车可提前结束、Esc 整轮停
EPISODE_STEPS = 50


def teleop():
    work = Path(os.environ["DATASETS_ROOT"]) / "so101_sim" / "_teleop" / TASK
    rec = make_recorder(work)
    kb = Keyboard()
    print("键盘遥操：wasd 底座/大臂，qe 小臂，rf 腕俯仰，tg 腕滚转，cv 夹爪；回车存一集，Esc 结束")

    for ep in range(N_EPISODES):
        obs, _ = rec.reset(seed=ep)
        for _ in range(EPISODE_STEPS):
            obs, _, terminated, truncated, _ = rec.step(kb.action()[None])
            frame = obs["sensor_data"]["top"]["rgb"][0].cpu().numpy()
            cv2.imshow("SO-101 遥操", cv2.resize(frame[:, :, ::-1], (512, 512)))
            cv2.waitKey(1)
            if kb.end_episode or bool(terminated[0]) or bool(truncated[0]):
                kb.end_episode = False
                break
            time.sleep(1.0 / FPS)
        print(f"第 {ep + 1} 集采完")
        if kb.stop:
            break

    cv2.destroyAllWindows()
    rec.close()   # flush 出 h5
    dataset = to_lerobot(work / "teleop.h5", work / "dataset")
    print(f"数据集已生成 -> {dataset}")


# ── 脚本回放自检：用内置动作序列代替键盘，跑通「采集 → 转换 → 加载」整条链 ────────
def selfcheck():
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    work = Path(os.environ["DATASETS_ROOT"]) / "so101_sim" / "_teleop_selfcheck" / TASK
    rec = make_recorder(work)

    # 一段确定性的够取动作：大臂下压、小臂前伸、末尾合爪；不依赖键盘，可无人值守跑完整条链。
    for ep in range(2):
        rec.reset(seed=ep)
        for step in range(30):
            action = np.zeros(6, dtype=np.float32)
            action[1] = -0.5           # 大臂下压
            action[2] = +0.5           # 小臂前伸
            action[5] = -0.8 if step > 20 else +0.4   # 先张爪再合爪
            rec.step(action[None])
    rec.close()

    dataset_dir = to_lerobot(work / "teleop.h5", work / "dataset")
    dataset = LeRobotDataset("so101_sim/teleop_selfcheck", root=str(dataset_dir))
    print("episodes:", dataset.meta.total_episodes)
    print("frames:", dataset.meta.total_frames)
    print("fps:", dataset.meta.fps)
    print("features:", list(dataset.meta.features.keys()))
    print("cameras:", dataset.meta.camera_keys)
    sample = dataset[0]
    print("action:", tuple(sample["action"].shape), "state:", tuple(sample["observation.state"].shape))


RUN = "teleop"   # 想不接键盘、只验证整条链是否跑通，改成 "selfcheck"

if __name__ == "__main__":
    (teleop if RUN == "teleop" else selfcheck)()
