from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import MotionResult, Observation, Pose, TaskConfig


class RobotBackend(ABC):
    """Common interface shared by hardware and simulation backends."""

    @abstractmethod
    def connect(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def disconnect(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def reset_task(self, task: TaskConfig) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_observation(self) -> Observation:
        raise NotImplementedError

    @abstractmethod
    def move_pose(
        self,
        target: Pose,
        speed: float,
        timeout: float = 8.0,
    ) -> MotionResult:
        raise NotImplementedError

    @abstractmethod
    def set_gripper(
        self,
        width: float,
        force: float = 0.5,
        timeout: float = 3.0,
    ) -> MotionResult:
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        raise NotImplementedError
