from __future__ import annotations

from dataclasses import replace
from time import monotonic

from ..models import MotionResult, Observation, Pose, TaskConfig
from .base import RobotBackend


class MockBackend(RobotBackend):
    """A deterministic teaching backend with no external dependencies."""

    def __init__(self) -> None:
        self.connected = False
        self.task: TaskConfig | None = None
        self.eef_pose = Pose(0.20, 0.0, 0.30)
        self.object_pose = Pose(0.0, 0.0, 0.0)
        self.gripper_width = 0.08
        self.object_attached = False
        self.object_from_eef = (0.0, 0.0, 0.0)
        self.collision = False

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def reset_task(self, task: TaskConfig) -> None:
        self.task = task
        self.eef_pose = Pose(0.20, 0.0, 0.30)
        self.object_pose = task.object_pose
        self.gripper_width = task.gripper_open_width
        self.object_attached = False
        self.collision = False

    def get_observation(self) -> Observation:
        return Observation(
            timestamp=monotonic(),
            eef_pose=self.eef_pose,
            gripper_width=self.gripper_width,
            object_pose=self.object_pose,
            object_visible=True,
            object_attached=self.object_attached,
            collision=self.collision,
        )

    def move_pose(
        self,
        target: Pose,
        speed: float,
        timeout: float = 8.0,
    ) -> MotionResult:
        if not self.connected:
            return MotionResult(False, "backend is not connected")
        if speed <= 0.0 or timeout <= 0.0:
            return MotionResult(False, "speed and timeout must be positive")
        if self.collision:
            return MotionResult(False, "collision flag is active")

        self.eef_pose = target
        if self.object_attached:
            dx, dy, dz = self.object_from_eef
            self.object_pose = target.shifted(dx, dy, dz)
        return MotionResult(True)

    def set_gripper(
        self,
        width: float,
        force: float = 0.5,
        timeout: float = 3.0,
    ) -> MotionResult:
        if self.task is None:
            return MotionResult(False, "task has not been reset")
        if width < 0.0 or timeout <= 0.0:
            return MotionResult(False, "invalid gripper command")

        self.gripper_width = width
        closing = width <= self.task.gripper_close_width + 0.01
        opening = width >= self.task.gripper_open_width - 0.01

        if closing and not self.object_attached:
            if self.eef_pose.distance_to(self.object_pose) <= 0.11:
                self.object_attached = True
                self.object_from_eef = (
                    self.object_pose.x - self.eef_pose.x,
                    self.object_pose.y - self.eef_pose.y,
                    self.object_pose.z - self.eef_pose.z,
                )
        elif opening and self.object_attached:
            self.object_attached = False
            self.object_pose = replace(
                self.object_pose,
                z=self.task.place_pose.z,
            )

        return MotionResult(True)

    def stop(self) -> None:
        self.collision = False
