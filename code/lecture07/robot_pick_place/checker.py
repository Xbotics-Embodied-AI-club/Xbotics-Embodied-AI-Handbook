from __future__ import annotations

from math import hypot

from .models import Observation, Pose, TaskConfig


def grasp_success(obs: Observation, task: TaskConfig) -> bool:
    if not obs.object_visible or obs.object_pose is None:
        return False
    width_is_plausible = (
        task.gripper_close_width - 0.015
        <= obs.gripper_width
        <= task.object_width + 0.03
    )
    object_is_near = obs.eef_pose.distance_to(obs.object_pose) <= 0.12
    return obs.object_attached or (width_is_plausible and object_is_near)


def lift_success(
    obs: Observation,
    initial_object_pose: Pose,
    task: TaskConfig,
) -> bool:
    if obs.object_pose is None:
        return False
    height_gain = obs.object_pose.z - initial_object_pose.z
    object_is_near = obs.eef_pose.distance_to(obs.object_pose) <= 0.14
    return height_gain >= task.lift_threshold and object_is_near


def place_success(obs: Observation, task: TaskConfig) -> bool:
    if obs.object_pose is None or obs.object_attached:
        return False
    xy_error = hypot(
        obs.object_pose.x - task.place_pose.x,
        obs.object_pose.y - task.place_pose.y,
    )
    return xy_error <= task.place_xy_tolerance
