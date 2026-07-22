from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum, auto
from math import sqrt
from typing import Any


@dataclass(frozen=True)
class Pose:
    """Position and quaternion in the robot base frame."""

    x: float
    y: float
    z: float
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0
    qw: float = 1.0

    def shifted(self, dx: float, dy: float, dz: float) -> "Pose":
        return Pose(
            self.x + dx,
            self.y + dy,
            self.z + dz,
            self.qx,
            self.qy,
            self.qz,
            self.qw,
        )

    def distance_to(self, other: "Pose") -> float:
        return sqrt(
            (self.x - other.x) ** 2
            + (self.y - other.y) ** 2
            + (self.z - other.z) ** 2
        )

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class ActionTargets:
    pre_grasp: Pose
    grasp: Pose
    lift: Pose
    pre_place: Pose
    place: Pose
    retreat: Pose


@dataclass
class TaskConfig:
    name: str
    object_pose: Pose
    place_pose: Pose
    grasp_quaternion: tuple[float, float, float, float]
    approach_direction: tuple[float, float, float]
    grasp_offset: tuple[float, float, float]
    pre_grasp_distance: float
    lift_height: float
    place_clearance: float
    retreat_height: float
    gripper_open_width: float
    gripper_close_width: float
    object_width: float
    fast_speed: float = 0.35
    slow_speed: float = 0.08
    position_tolerance: float = 0.01
    lift_threshold: float = 0.05
    place_xy_tolerance: float = 0.04
    max_retries: int = 3


@dataclass(frozen=True)
class Observation:
    timestamp: float
    eef_pose: Pose
    gripper_width: float
    object_pose: Pose | None
    object_visible: bool
    object_attached: bool
    collision: bool = False
    raw: dict[str, Any] | None = None


@dataclass(frozen=True)
class MotionResult:
    success: bool
    reason: str = ""


class State(Enum):
    INIT = auto()
    MOVE_PRE_GRASP = auto()
    APPROACH = auto()
    CLOSE_GRIPPER = auto()
    VERIFY_GRASP = auto()
    LIFT = auto()
    VERIFY_LIFT = auto()
    MOVE_PRE_PLACE = auto()
    LOWER_PLACE = auto()
    OPEN_GRIPPER = auto()
    VERIFY_PLACE = auto()
    RETREAT = auto()
    DONE = auto()
    RECOVER = auto()
    SAFE_EXIT = auto()


class FailureCode(Enum):
    NONE = auto()
    MOTION_FAILED = auto()
    GRASP_FAILED = auto()
    LIFT_FAILED = auto()
    PLACE_FAILED = auto()
    SENSOR_FAILED = auto()
    COLLISION = auto()
