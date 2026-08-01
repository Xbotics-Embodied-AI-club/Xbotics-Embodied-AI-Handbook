"""Scene-derived task configurations and the five key action poses for MuJoCo.

The Lecture 07 state machine is platform-agnostic: it consumes a
``robot_pick_place.TaskConfig`` and produces five target poses with
``robot_pick_place.pose_generator.generate_targets``.  This module builds that
``TaskConfig`` from the actual MuJoCo scene (object positions, gripper opening
measured on the SO-101 model, and a grasp offset calibrated against the real
jaw geometry), so the five poses are executable by the simulated arm.

The grasp offsets below were found empirically on the SO-101 MuJoCo model
(``HOME_QPOS`` wrist orientation, ``wxyz=(0, 0, 1, 0)``): they place the two jaw
collision geoms on opposite sides of the task object so that both jaws contact
it when the gripper closes (see ``StickyGraspAssist``).
"""

from __future__ import annotations

from dataclasses import replace

import mujoco
import numpy as np

from mujoco_tasks.envs.scene import HOME_QPOS, TABLE_TOP_Z
from mujoco_tasks.motion.control import quat_from_axis_angle, quat_mul
from mujoco_tasks.motion.grasp_to_gripper import (
    GRASP_TO_EEF_ROTATION,
    GRIPPER_WIDTH_INTERCEPT,
    GRIPPER_WIDTH_SLOPE,
    measure_grasp_pose,
)

from robot_pick_place.models import ActionTargets, Pose, TaskConfig
from robot_pick_place.pose_generator import generate_targets

# Default gripper opening (metres) at the home gripper joint angle, derived from
# the linear width -> joint-angle calibration used by ``grasp_width_to_gripper_qpos``.
DEFAULT_OPEN_WIDTH = (float(HOME_QPOS[-1]) - GRIPPER_WIDTH_INTERCEPT) / GRIPPER_WIDTH_SLOPE

# Midpoint between the two jaw collision geoms in the EEF frame, measured on the
# SO-101 model with the gripper fully closed (the bottle close width clamps to
# the joint limit).  Used to place the EEF so the closed jaws straddle the object.
JAW_MIDPOINT_EEF_CLOSED = np.array([-0.0057, 0.0001, -0.0407], dtype=np.float64)

# ``grasp_offset`` is EEF_origin - object_center in the robot base (world) frame.
# The EEF quaternion is ``(qx, qy, qz, qw) = (0, 1, 0, 0)`` (wxyz = (0, 0, 1, 0),
# a 180 deg rotation about Y, R = diag(-1, 1, -1)): the wrist points downward
# while the jaw opening stays horizontal.
TASK_LAYOUTS: dict[str, dict[str, object]] = {
    "cube": {
        "name": "cube_a_to_b",
        "object_center": (0.22, -0.12, TABLE_TOP_Z + 0.025),  # red A zone
        # Released slightly above the table so IK residual never sinks the cube
        # into the surface (which would pop it out and slide it on settle).
        # Place pose Y is nudged past the B-zone centre so the cube (released a
        # few mm short of the pose) lands centred in the blue frame.
        # Place pose is nudged past the B-zone centre so the cube (released a
        # few mm short of the pose) lands centred in the blue frame.
        # The released cube hangs ~1.5 cm off the place EEF (grasp offset), so
        # the pose is compensated to make the cube land centred in the blue B
        # zone at (0.22, 0.12).
        "place_center": (0.2047, 0.1246, TABLE_TOP_Z + 0.033),  # blue B zone
        "approach_direction": (0.0, 0.0, -1.0),
        "grasp_quaternion": (0.0, 1.0, 0.0, 0.0),
        # Empirically calibrated on the rotating-jaw gripper so that both
        # fingers pinch the cube (90 deg roll).  The fixed jaw's inner surface
        # is flush with the cube's +Y face (its mesh stays fully outside the
        # cube) and the moving jaw closes from the -Y side onto it.  The X
        # offset is as close to the cube centre as the wedge geometry allows.
        "grasp_offset": (-0.015, 0.022, -0.025),
        # Extra roll of the gripper about its own pointing axis (deg).  The
        # jaws then pinch the cube's Y faces instead of its X faces.
        "grasp_roll_deg": 90.0,
        "settle_time": 0.2,
        "pre_grasp_distance": 0.08,
        "lift_height": 0.10,
        "retreat_height": 0.10,
        "gripper_close_width": 0.049,
        "object_width": 0.05,
        "place_xy_tolerance": 0.04,
        "slow_speed": 0.08,
        "center_offset": (0.0, 0.0, 0.025),
        "object_body": "cube",
        "free_joint": "cube_free",
        "object_geom": "cube_geom",
    },
    "bottle": {
        "name": "bottle_to_bin",
        "object_center": (0.22, -0.12, TABLE_TOP_Z + 0.055),
        # Place pose Y is offset past the box centre so the bottle (released a
        # few cm short of the pose due to the grasp offset) lands near the box
        # middle instead of its near edge.
        # Compensated so the released bottle lands centred in the box at
        # (0.22, 0.12).
        # Raised so the transit endpoint keeps the bottle hovering above the
        # box (the arm holds it high instead of pressing it toward the table);
        # the release drops it into the bin.
        "place_center": (0.2497, 0.168, TABLE_TOP_Z + 0.088),  # box centre
        # Horizontal radial approach: the grasp +Z axis stays parallel to the
        # table and points back toward the base; the gripper moves radially
        # outward (+X/+Y) to pinch the bottle body.  generate_targets
        # normalizes this vector, so it only needs the correct direction.
        "approach_direction": (0.22, -0.12, 0.0),
        # Empirically calibrated so the two fingers wrap the bottle body.
        "grasp_offset": (0.018, -0.015, 0.0104),
        "pre_grasp_distance": 0.04,
        "lift_height": 0.10,
        "retreat_height": 0.10,
        "gripper_close_width": 0.038,
        "object_width": 0.038,
        "place_xy_tolerance": 0.07,
        "slow_speed": 0.06,
        "settle_time": 0.8,
        "center_offset": (0.0, 0.0, 0.055),
        "object_body": "bottle",
        "free_joint": "bottle_free",
        "object_geom": "bottle_body_geom",
    },
}


def task_layout(task: str) -> dict[str, object]:
    """Return the scene constants used to build a task configuration."""

    try:
        return TASK_LAYOUTS[task]
    except KeyError as exc:
        raise ValueError(f"unknown MuJoCo task {task!r}; choose from {sorted(TASK_LAYOUTS)}") from exc


def object_body_info(task: str) -> dict[str, str]:
    """Return MuJoCo names for the task object body / free joint / geom."""

    layout = task_layout(task)
    return {
        "body": str(layout["object_body"]),
        "free_joint": str(layout["free_joint"]),
        "geom": str(layout["object_geom"]),
        "center_offset": tuple(layout["center_offset"]),
    }


def measured_open_width(model, data) -> float:
    """Measure the jaw opening (m) on the compiled SO-101 model at HOME_QPOS."""

    grasp = measure_grasp_pose(
        model,
        data,
        np.asarray(HOME_QPOS, dtype=np.float64),
        score=1.0,
    )
    return float(grasp.width)


def _quat_to_mat(quat: np.ndarray) -> np.ndarray:
    matrix = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(matrix, np.asarray(quat, dtype=np.float64).reshape(4))
    return matrix.reshape(3, 3)


def radial_horizontal_grasp(center: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    """EEF pose for a horizontal radial grasp whose grasp +Z is parallel to the table.

    The grasp frame follows the repo's GraspNet convention (approach along
    ``-Z_grasp``, jaw width along ``X_grasp``, plane normal ``Y_grasp``):

    - ``Z_grasp`` is horizontal, pointing from the object back toward the base;
    - ``Y_grasp`` is vertical, so the jaw opening stays in the table plane;
    - the EEF pose is derived from the calibrated jaw midpoint at the closed
      gripper opening, so the two jaws land symmetrically on the bottle body.
    """

    center = np.asarray(center, dtype=np.float64)
    radial = np.array([center[0], center[1], 0.0], dtype=np.float64)
    radial_norm = float(np.linalg.norm(radial))
    if radial_norm < 1e-9:
        raise ValueError("object must not sit on the base axis for a radial grasp")
    radial = radial / radial_norm

    z_grasp = -radial
    y_grasp = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    x_grasp = np.cross(y_grasp, z_grasp)
    r_grasp = np.stack([x_grasp, y_grasp, z_grasp], axis=1)
    r_eef = r_grasp @ GRASP_TO_EEF_ROTATION

    quat = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, r_eef.reshape(-1))
    eef_pos = center - r_eef @ JAW_MIDPOINT_EEF_CLOSED
    return eef_pos, quat


def make_task_config(
    task: str,
    model=None,
    data=None,
    *,
    open_width: float | None = None,
    grasp_roll_deg: float | None = None,
) -> TaskConfig:
    """Build a ``TaskConfig`` whose five poses are executable in the MuJoCo scene."""

    layout = task_layout(task)
    if open_width is None:
        open_width = measured_open_width(model, data) if model is not None else DEFAULT_OPEN_WIDTH

    object_pose = Pose(*tuple(layout["object_center"]))
    place_pose = Pose(*tuple(layout["place_center"]))
    if task == "bottle":
        _, quat = radial_horizontal_grasp(layout["object_center"])
        grasp_quaternion = (float(quat[1]), float(quat[2]), float(quat[3]), float(quat[0]))
        grasp_offset = tuple(layout["grasp_offset"])
    else:
        grasp_quaternion = tuple(layout["grasp_quaternion"])
        grasp_offset = tuple(layout["grasp_offset"])
        if task == "cube":
            roll_deg = float(
                grasp_roll_deg
                if grasp_roll_deg is not None
                else layout.get("grasp_roll_deg", 0.0)
            )
            roll_rad = np.deg2rad(roll_deg)
            if abs(roll_rad) > 1e-9:
                base_wxyz = np.array(
                    [grasp_quaternion[3], *grasp_quaternion[:3]],
                    dtype=np.float64,
                )
                rotated = quat_mul(
                    base_wxyz,
                    quat_from_axis_angle(np.array([0.0, 0.0, 1.0]), roll_rad),
                )
                grasp_quaternion = (
                    float(rotated[1]),
                    float(rotated[2]),
                    float(rotated[3]),
                    float(rotated[0]),
                )
    return TaskConfig(
        name=str(layout["name"]),
        object_pose=object_pose,
        place_pose=place_pose,
        grasp_quaternion=grasp_quaternion,
        approach_direction=tuple(layout["approach_direction"]),
        grasp_offset=grasp_offset,
        pre_grasp_distance=float(layout["pre_grasp_distance"]),
        lift_height=float(layout["lift_height"]),
        retreat_height=float(layout["retreat_height"]),
        gripper_open_width=float(open_width),
        gripper_close_width=float(layout["gripper_close_width"]),
        object_width=float(layout["object_width"]),
        place_xy_tolerance=float(layout["place_xy_tolerance"]),
        slow_speed=float(layout["slow_speed"]),
    )


def make_five_targets(task_config: TaskConfig) -> ActionTargets:
    """Generate the five key poses (pre_grasp/grasp/lift/place/retreat)."""

    targets = generate_targets(task_config)
    if task_config.name == "bottle_to_bin":
        layout = task_layout("bottle")
        object_center = (
            task_config.object_pose.x,
            task_config.object_pose.y,
            task_config.object_pose.z,
        )
        matches_mujoco = all(
            abs(a - b) < 1e-6
            for a, b in zip(object_center, layout["object_center"])
        )
        if matches_mujoco:
            # The box sits on the mirror azimuth, so the placement pose uses the
            # mirrored radial orientation (reachable on the +Y side).
            place_center = np.array(
                (
                    task_config.place_pose.x,
                    task_config.place_pose.y,
                    task_config.place_pose.z,
                ),
                dtype=np.float64,
            )
            _, quat_place = radial_horizontal_grasp(tuple(place_center))
            # The bottle hangs from the EEF at the grasp-frame relative offset
            # that was calibrated for the grasp side; place the EEF so the
            # bottle rests at place_center once the gripper rotates to the
            # mirrored orientation.
            grasp_quat = task_config.grasp_quaternion
            quat_grasp = np.array(
                [grasp_quat[3], grasp_quat[0], grasp_quat[1], grasp_quat[2]],
                dtype=np.float64,
            )
            rel_eef = _quat_to_mat(quat_grasp).T @ np.asarray(
                task_config.grasp_offset, dtype=np.float64
            )
            eef_place = place_center - _quat_to_mat(quat_place) @ rel_eef
            qx, qy, qz, qw = quat_place[1], quat_place[2], quat_place[3], quat_place[0]
            place = Pose(
                float(eef_place[0]),
                float(eef_place[1]),
                float(eef_place[2]),
                float(qx),
                float(qy),
                float(qz),
                float(qw),
            )
            retreat = Pose(
                float(eef_place[0]),
                float(eef_place[1]),
                float(eef_place[2] + task_config.retreat_height),
                float(qx),
                float(qy),
                float(qz),
                float(qw),
            )
            targets = replace(targets, place=place, retreat=retreat)
    return targets


__all__ = [
    "DEFAULT_OPEN_WIDTH",
    "TASK_LAYOUTS",
    "make_five_targets",
    "make_task_config",
    "measured_open_width",
    "object_body_info",
    "task_layout",
]
