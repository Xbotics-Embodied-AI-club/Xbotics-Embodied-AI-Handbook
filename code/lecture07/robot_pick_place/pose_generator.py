from __future__ import annotations

from math import sqrt

from .models import ActionTargets, Pose, TaskConfig


def _normalize(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    length = sqrt(sum(value * value for value in vector))
    if length < 1e-9:
        raise ValueError("approach_direction must be non-zero")
    return tuple(value / length for value in vector)


def generate_targets(task: TaskConfig) -> ActionTargets:
    """Generate the six key poses from the object and place poses."""

    ax, ay, az = _normalize(task.approach_direction)
    ox, oy, oz = task.grasp_offset
    qx, qy, qz, qw = task.grasp_quaternion

    grasp = Pose(
        task.object_pose.x + ox,
        task.object_pose.y + oy,
        task.object_pose.z + oz,
        qx,
        qy,
        qz,
        qw,
    )
    pre_grasp = grasp.shifted(
        -ax * task.pre_grasp_distance,
        -ay * task.pre_grasp_distance,
        -az * task.pre_grasp_distance,
    )
    lift = grasp.shifted(0.0, 0.0, task.lift_height)

    place = Pose(
        task.place_pose.x + ox,
        task.place_pose.y + oy,
        task.place_pose.z + oz,
        qx,
        qy,
        qz,
        qw,
    )
    pre_place = place.shifted(0.0, 0.0, task.place_clearance)
    retreat = place.shifted(0.0, 0.0, task.retreat_height)

    return ActionTargets(
        pre_grasp=pre_grasp,
        grasp=grasp,
        lift=lift,
        pre_place=pre_place,
        place=place,
        retreat=retreat,
    )
