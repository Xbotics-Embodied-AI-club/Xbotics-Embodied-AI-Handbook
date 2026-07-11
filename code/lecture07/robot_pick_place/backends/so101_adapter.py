from __future__ import annotations

from collections.abc import Callable, Mapping
from time import monotonic, sleep
from typing import Any

from ..models import MotionResult, Observation, Pose, TaskConfig
from .base import RobotBackend


class SO101Backend(RobotBackend):
    """Adapter around a connected LeRobot-style SO-101 robot object.

    The injected robot object must provide connect(), disconnect(),
    get_observation(), and send_action(). Pose planning and forward
    kinematics are injected because their implementation depends on the
    selected URDF, calibration, and motion-planning stack.
    """

    def __init__(
        self,
        robot: Any,
        plan_pose: Callable[[Pose, float], list[dict[str, float]]],
        read_eef_pose: Callable[[Mapping[str, Any]], Pose],
        detect_object: Callable[[Mapping[str, Any]], Pose | None],
        gripper_key: str = "gripper.pos",
    ) -> None:
        self.robot = robot
        self.plan_pose = plan_pose
        self.read_eef_pose = read_eef_pose
        self.detect_object = detect_object
        self.gripper_key = gripper_key
        self.task: TaskConfig | None = None

    def connect(self) -> None:
        self.robot.connect()

    def disconnect(self) -> None:
        self.robot.disconnect()

    def reset_task(self, task: TaskConfig) -> None:
        self.task = task

    def get_observation(self) -> Observation:
        raw = self.robot.get_observation()
        object_pose = self.detect_object(raw)
        return Observation(
            timestamp=monotonic(),
            eef_pose=self.read_eef_pose(raw),
            gripper_width=float(raw.get(self.gripper_key, 0.0)),
            object_pose=object_pose,
            object_visible=object_pose is not None,
            object_attached=False,
            collision=bool(raw.get("collision", False)),
            raw=dict(raw),
        )

    def move_pose(
        self,
        target: Pose,
        speed: float,
        timeout: float = 8.0,
    ) -> MotionResult:
        deadline = monotonic() + timeout
        try:
            joint_actions = self.plan_pose(target, speed)
        except Exception as exc:
            return MotionResult(False, f"planning failed: {exc}")

        for action in joint_actions:
            if monotonic() > deadline:
                self.stop()
                return MotionResult(False, "motion timeout")
            self.robot.send_action(action)
            sleep(0.02)
        return MotionResult(True)

    def set_gripper(
        self,
        width: float,
        force: float = 0.5,
        timeout: float = 3.0,
    ) -> MotionResult:
        del force, timeout
        try:
            self.robot.send_action({self.gripper_key: width})
        except Exception as exc:
            return MotionResult(False, f"gripper command failed: {exc}")
        return MotionResult(True)

    def stop(self) -> None:
        current = self.robot.get_observation()
        hold = {
            key: value
            for key, value in current.items()
            if key.endswith(".pos")
        }
        if hold:
            self.robot.send_action(hold)
