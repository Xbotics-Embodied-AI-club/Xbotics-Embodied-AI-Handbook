from __future__ import annotations

from .models import Pose, TaskConfig


def cube_task() -> TaskConfig:
    """Red cube: vertical top grasp, move from A to B."""

    return TaskConfig(
        name="cube_a_to_b",
        object_pose=Pose(0.28, 0.10, 0.04),
        place_pose=Pose(0.28, -0.12, 0.04),
        grasp_quaternion=(0.0, 0.0, 0.0, 1.0),
        approach_direction=(0.0, 0.0, -1.0),
        grasp_offset=(0.0, 0.0, 0.055),
        pre_grasp_distance=0.12,
        lift_height=0.13,
        retreat_height=0.12,
        gripper_open_width=0.075,
        gripper_close_width=0.035,
        object_width=0.04,
    )


def bottle_task() -> TaskConfig:
    """Vertical bottle: horizontal side approach, place in a bin."""

    return TaskConfig(
        name="bottle_to_bin",
        object_pose=Pose(0.30, 0.08, 0.12),
        place_pose=Pose(0.30, -0.16, 0.12),
        grasp_quaternion=(0.0, 0.7071, 0.0, 0.7071),
        approach_direction=(-1.0, 0.0, 0.0),
        grasp_offset=(0.065, 0.0, 0.03),
        pre_grasp_distance=0.11,
        lift_height=0.16,
        retreat_height=0.14,
        gripper_open_width=0.095,
        gripper_close_width=0.058,
        object_width=0.065,
        slow_speed=0.06,
        place_xy_tolerance=0.05,
    )


def get_task(name: str) -> TaskConfig:
    tasks = {
        "cube": cube_task,
        "bottle": bottle_task,
    }
    try:
        return tasks[name]()
    except KeyError as exc:
        choices = ", ".join(sorted(tasks))
        raise ValueError(f"unknown task {name!r}; choose from {choices}") from exc
