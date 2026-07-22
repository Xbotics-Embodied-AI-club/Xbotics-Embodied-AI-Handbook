from __future__ import annotations

from collections.abc import Callable
from time import monotonic
from typing import Any

import numpy as np

from ..models import MotionResult, Observation, Pose, TaskConfig
from .base import RobotBackend


class ManiSkillBackend(RobotBackend):
    """Adapter for a single CPU ManiSkill environment.

    pose_to_actions converts one absolute target pose into a short sequence
    of actions compatible with the selected ManiSkill controller.
    extract_observation converts the environment observation into the common
    Observation data class used by the state machine.
    """

    def __init__(
        self,
        env: Any,
        pose_to_actions: Callable[[Pose, float], list[np.ndarray]],
        extract_observation: Callable[[Any], Observation],
        gripper_action: Callable[[float], np.ndarray],
    ) -> None:
        self.env = env
        self.pose_to_actions = pose_to_actions
        self.extract_observation = extract_observation
        self.gripper_action = gripper_action
        self.last_obs: Any = None
        self.task: TaskConfig | None = None

    def connect(self) -> None:
        self.last_obs, _ = self.env.reset(seed=0)

    def disconnect(self) -> None:
        self.env.close()

    def reset_task(self, task: TaskConfig) -> None:
        self.task = task
        self.last_obs, _ = self.env.reset()

    def get_observation(self) -> Observation:
        return self.extract_observation(self.last_obs)

    def _step(self, action: np.ndarray) -> MotionResult:
        obs, _, terminated, truncated, info = self.env.step(action)
        self.last_obs = obs
        if bool(info.get("collision", False)):
            return MotionResult(False, "collision")
        if bool(terminated) or bool(truncated):
            return MotionResult(False, "episode ended")
        return MotionResult(True)

    def move_pose(
        self,
        target: Pose,
        speed: float,
        timeout: float = 8.0,
    ) -> MotionResult:
        del timeout
        for action in self.pose_to_actions(target, speed):
            result = self._step(action)
            if not result.success:
                return result
        return MotionResult(True)

    def set_gripper(
        self,
        width: float,
        force: float = 0.5,
        timeout: float = 3.0,
    ) -> MotionResult:
        del force, timeout
        return self._step(self.gripper_action(width))

    def stop(self) -> None:
        if self.env.action_space is not None:
            zero = np.zeros(self.env.action_space.shape, dtype=np.float32)
            self._step(zero)
