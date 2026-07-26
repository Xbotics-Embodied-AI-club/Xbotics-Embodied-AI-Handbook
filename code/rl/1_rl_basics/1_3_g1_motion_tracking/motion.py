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
        return int(self.joint_pos.shape[0])

    @property
    def duration_s(self) -> float:
        return self.num_frames / self.fps

    def joint_snapshot(self, frame_index: int, num_joints: int = 6) -> list[float]:
        """取一小段关节角，方便在训练前看清 motion 文件里的动作数值。"""

        frame_index = min(max(frame_index, 0), self.num_frames - 1)
        return [round(float(value), 4) for value in self.joint_pos[frame_index, :num_joints].detach().cpu()]

    @classmethod
    def load(cls, motion_file: str | Path, device: str | torch.device = "cpu") -> "MotionClip":
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
