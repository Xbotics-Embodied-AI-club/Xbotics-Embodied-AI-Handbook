from __future__ import annotations

from dataclasses import replace

from .models import FailureCode, TaskConfig


def adjust_task(
    task: TaskConfig,
    failure: FailureCode,
    retry_index: int,
) -> TaskConfig:
    """Return a modified task configuration for the next attempt."""

    if retry_index <= 0:
        return task

    if failure in {FailureCode.MOTION_FAILED, FailureCode.COLLISION}:
        return replace(
            task,
            pre_grasp_distance=task.pre_grasp_distance + 0.02,
            slow_speed=max(0.03, task.slow_speed * 0.8),
        )

    if failure in {FailureCode.GRASP_FAILED, FailureCode.LIFT_FAILED}:
        gx, gy, gz = task.grasp_offset
        return replace(
            task,
            grasp_offset=(gx, gy, gz + 0.005),
            gripper_close_width=max(0.0, task.gripper_close_width - 0.003),
            slow_speed=max(0.03, task.slow_speed * 0.85),
        )

    if failure is FailureCode.PLACE_FAILED:
        return replace(
            task,
            place_pose=task.place_pose.shifted(0.0, 0.0, 0.01),
            retreat_height=task.retreat_height + 0.02,
        )

    return task
