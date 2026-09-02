"""参考动作文件的读取与校验。

参考动作由 `1_0_video_to_g1_reference/` 那条管线产出（视频 → GVHMR → GMR → npz）。
本文件只负责把它读进来、逐项校验形状，并提供训练前预览用的小工具。

讲义对应：第14讲 4.4 节（管线）与第 8 节（用它训练）。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class MotionClip:
    """一段已经重定向到 Unitree G1 的参考动作。"""

    fps: float
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    body_pos_w: torch.Tensor
    body_quat_w: torch.Tensor
    body_lin_vel_w: torch.Tensor
    body_ang_vel_w: torch.Tensor

    @property
    def num_frames(self) -> int:
        """这段动作一共多少帧。

        Returns:
            帧数。
        """
        return int(self.joint_pos.shape[0])

    @property
    def duration_s(self) -> float:
        """这段动作有多长（秒）。

        Returns:
            时长，按帧数除以帧率算。
        """
        return self.num_frames / self.fps

    def joint_snapshot(self, frame_index: int, num_joints: int = 6) -> list[float]:
        """取某一帧前几个关节的角度，训练前用来肉眼确认动作文件不是一团乱码。

        Args:
            frame_index: 帧号，越界会被夹到有效范围内。
            num_joints: 取前几个关节。

        Returns:
            保留四位小数的关节角列表。
        """

        frame_index = min(max(frame_index, 0), self.num_frames - 1)
        return [round(float(value), 4) for value in self.joint_pos[frame_index, :num_joints].detach().cpu()]

    @classmethod
    def load(cls, motion_file: str | Path, device: str | torch.device = "cpu") -> "MotionClip":
        """从 npz 文件读一段参考动作，并逐项校验形状。

        宁可在训练开始前把文件读坏的情况直接抛出来，也不要让一个形状不对的
        参考动作悄悄进入训练、几小时后才发现学的是一团乱码。

        Args:
            motion_file: `build_motion_npz.py` 产出的 npz 路径。
            device: 张量放到哪个设备上。

        Returns:
            校验通过的 `MotionClip`。

        Raises:
            ValueError: 缺少必需字段，或某个张量的形状与 29 关节的 G1 对不上。
        """
        data = np.load(motion_file)
        required = [
            "fps",
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
        ]
        missing = [key for key in required if key not in data.files]
        if missing:
            raise ValueError(f"motion file missing required keys: {missing}")

        joint_pos = torch.as_tensor(data["joint_pos"], dtype=torch.float32, device=device)
        joint_vel = torch.as_tensor(data["joint_vel"], dtype=torch.float32, device=device)
        body_pos_w = torch.as_tensor(data["body_pos_w"], dtype=torch.float32, device=device)
        body_quat_w = torch.as_tensor(data["body_quat_w"], dtype=torch.float32, device=device)
        body_lin_vel_w = torch.as_tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
        body_ang_vel_w = torch.as_tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)

        if joint_pos.ndim != 2 or joint_pos.shape[1] != 29:
            raise ValueError(f"joint_pos must have shape (T, 29), got {tuple(joint_pos.shape)}")
        if joint_vel.shape != joint_pos.shape:
            raise ValueError(f"joint_vel must have shape {tuple(joint_pos.shape)}, got {tuple(joint_vel.shape)}")
        if body_pos_w.ndim != 3 or body_pos_w.shape[-1] != 3:
            raise ValueError(f"body_pos_w must have shape (T, B, 3), got {tuple(body_pos_w.shape)}")
        if body_quat_w.shape[:2] != body_pos_w.shape[:2] or body_quat_w.shape[-1] != 4:
            raise ValueError(f"body_quat_w must have shape (T, B, 4), got {tuple(body_quat_w.shape)}")
        if body_lin_vel_w.shape != body_pos_w.shape or body_ang_vel_w.shape != body_pos_w.shape:
            raise ValueError("body velocity tensors must match body_pos_w shape")

        return cls(
            fps=float(np.asarray(data["fps"]).reshape(-1)[0]),
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            body_pos_w=body_pos_w,
            body_quat_w=body_quat_w,
            body_lin_vel_w=body_lin_vel_w,
            body_ang_vel_w=body_ang_vel_w,
        )


def describe_motion_dataset(motion_file: str | Path) -> dict[str, str | int | float | list[float]]:
    """读一遍参考动作文件，汇总成一份可打印的概况。

    Args:
        motion_file: 参考动作 npz 路径。

    Returns:
        含帧数、帧率、时长与几帧关节角样本的字典。
    """
    motion = MotionClip.load(motion_file, device="cpu")
    middle_frame = motion.num_frames // 2
    return {
        "motion_file": str(motion_file),
        "frames": motion.num_frames,
        "fps": motion.fps,
        "duration_s": round(motion.duration_s, 2),
        "joint_dim": int(motion.joint_pos.shape[1]),
        "body_count": int(motion.body_pos_w.shape[1]),
        "first_joint_pos": motion.joint_snapshot(0),
        "middle_joint_pos": motion.joint_snapshot(middle_frame),
        "last_joint_pos": motion.joint_snapshot(motion.num_frames - 1),
    }


def print_motion_dataset_preview(motion_file: str | Path) -> None:
    """把参考动作的概况打到终端，开训前肉眼确认数据没读错。

    Args:
        motion_file: 参考动作 npz 路径。
    """
    preview = describe_motion_dataset(motion_file)
    print("Motion dataset preview")
    for key, value in preview.items():
        print(f"{key}: {value}")
