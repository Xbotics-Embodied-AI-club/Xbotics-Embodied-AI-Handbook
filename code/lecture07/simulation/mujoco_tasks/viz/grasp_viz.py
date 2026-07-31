"""MuJoCo 3D visualization for GraspNet-style parallel-jaw grasps."""

from __future__ import annotations

from collections.abc import Sequence

import mujoco
import numpy as np

from mujoco_tasks.motion.grasp_pose import GraspPose

_FINGER_WIDTH = 0.004
_TAIL_LENGTH = 0.025
_DEPTH_BASE = 0.02
_DEFAULT_JAW_HEIGHT = 0.004
_GRASP_RGBA = (1.0, 0.0, 0.0, 0.85)
_GRASP_AXIS_LENGTH = 0.025
_GRASP_VIZ_WIDTH_SCALE = 0.85
# Visual-only shift along grasp -Z (wrist-roll local -Z) to align with the jaw opening.
_GRASP_VIZ_CENTER_OFFSET_IN_WRIST = np.array([0.0, 0.0, -0.02], dtype=np.float64)


def _rotation_to_quat(rotation: np.ndarray) -> np.ndarray:
    quat = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, np.asarray(rotation, dtype=np.float64).reshape(9))
    return quat


def _gripper_part_boxes(width: float, depth: float, jaw_height: float) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return (local_center, half_size) pairs in the wrist-roll grasp frame.

    The gripper geometry lies in the local X-Z plane: width along X, approach along ``-Z``.
    """

    finger_half = _FINGER_WIDTH / 2.0
    jaw_half = jaw_height / 2.0
    finger_depth = depth + _DEPTH_BASE + _FINGER_WIDTH
    finger_center_z = (depth - _DEPTH_BASE - _FINGER_WIDTH) / 2.0

    left_center = np.array(
        [-width / 2.0 - finger_half, 0.0, -finger_center_z],
        dtype=np.float64,
    )
    right_center = np.array(
        [width / 2.0 + finger_half, 0.0, -finger_center_z],
        dtype=np.float64,
    )
    bottom_center = np.array(
        [0.0, 0.0, _DEPTH_BASE + finger_half],
        dtype=np.float64,
    )
    tail_center = np.array(
        [0.0, 0.0, _DEPTH_BASE + _FINGER_WIDTH + _TAIL_LENGTH / 2.0],
        dtype=np.float64,
    )

    finger_half_size = np.array([finger_half, jaw_half, finger_depth / 2.0], dtype=np.float64)
    bottom_half_size = np.array([width / 2.0, jaw_half, finger_half], dtype=np.float64)
    tail_half_size = np.array([finger_half, jaw_half, _TAIL_LENGTH / 2.0], dtype=np.float64)

    return [
        (left_center, finger_half_size),
        (right_center, finger_half_size),
        (bottom_center, bottom_half_size),
        (tail_center, tail_half_size),
    ]


def _add_grasp_gripper_local(
    body: mujoco.MjsBody,
    *,
    prefix: str,
    width: float,
    depth: float,
    jaw_height: float,
    rgba: tuple[float, float, float, float] = _GRASP_RGBA,
) -> None:
    """Add grasp gripper boxes in the parent body's local frame."""

    parts = ("left", "right", "bottom", "tail")
    for part_name, (local_center, half_size) in zip(
        parts,
        _gripper_part_boxes(width, depth, jaw_height),
    ):
        geom = body.add_geom(
            name=f"{prefix}_{part_name}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=local_center.tolist(),
            size=half_size.tolist(),
            rgba=list(rgba),
        )
        geom.contype = 0
        geom.conaffinity = 0


def _add_grasp_axis_local(
    body: mujoco.MjsBody,
    *,
    prefix: str,
    length: float = _GRASP_AXIS_LENGTH,
    radius: float = 0.0015,
) -> None:
    """Draw grasp-frame XYZ axes at the body origin (local X/Y/Z)."""

    axes = (
        ("x", [length, 0.0, 0.0], [1.0, 0.15, 0.15, 0.9]),
        ("y", [0.0, length, 0.0], [0.15, 1.0, 0.15, 0.9]),
        ("z", [0.0, 0.0, length], [0.15, 0.35, 1.0, 0.9]),
    )
    for axis_name, endpoint, rgba in axes:
        geom = body.add_geom(
            name=f"{prefix}_axis_{axis_name}",
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            fromto=[0.0, 0.0, 0.0, *endpoint],
            size=[radius, 0.0, 0.0],
            rgba=rgba,
        )
        geom.contype = 0
        geom.conaffinity = 0


def home_grasp_visual_params(
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> tuple[np.ndarray, float, float, float]:
    """Return grasp-viz size and wrist-local center offset measured at ``HOME_QPOS``."""

    from mujoco_tasks.envs.scene import HOME_QPOS, JOINT_NAMES, WRIST_ROLL_BODY_NAME
    from mujoco_tasks.motion.grasp_to_gripper import measure_grasp_center, measure_grasp_pose

    joint_ids = [model.joint(name).qposadr[0] for name in JOINT_NAMES]
    data.qpos[joint_ids] = HOME_QPOS
    mujoco.mj_forward(model, data)

    grasp = measure_grasp_pose(model, data, np.asarray(HOME_QPOS, dtype=np.float64))
    wrist_id = model.body(WRIST_ROLL_BODY_NAME).id
    wrist_rotation = data.xmat[wrist_id].reshape(3, 3)
    center_in_wrist = wrist_rotation.T @ (measure_grasp_center(model, data) - data.xpos[wrist_id])
    center_in_wrist = center_in_wrist + _GRASP_VIZ_CENTER_OFFSET_IN_WRIST
    viz_width = grasp.width * _GRASP_VIZ_WIDTH_SCALE
    return center_in_wrist, viz_width, grasp.depth, grasp.height


def add_grasp_visualization_on_wrist_roll(
    scene_spec: mujoco.MjSpec,
    wrist_roll_body_name: str,
    *,
    center_in_wrist: np.ndarray,
    width: float,
    depth: float = 0.02,
    height: float = _DEFAULT_JAW_HEIGHT,
    show_axes: bool = True,
    prefix: str = "grasp",
) -> None:
    """Attach a red grasp visualization that moves with the wrist-roll body."""

    grasp_body = scene_spec.body(wrist_roll_body_name).add_body(
        name=f"{prefix}_vis",
        pos=np.asarray(center_in_wrist, dtype=np.float64).reshape(3).tolist(),
    )
    _add_grasp_gripper_local(
        grasp_body,
        prefix=prefix,
        width=width,
        depth=depth,
        jaw_height=height,
    )
    if show_axes:
        _add_grasp_axis_local(grasp_body, prefix=prefix)


def add_grasp_visualizations(
    scene_spec: mujoco.MjSpec,
    grasps: Sequence[GraspPose],
    *,
    show_axes: bool = True,
) -> None:
    """Attach visual-only grasp grippers to the scene world body (static, for tests)."""

    for index, grasp in enumerate(grasps):
        body = scene_spec.worldbody.add_body(name=f"grasp_vis_{index}", pos=[0.0, 0.0, 0.0])
        prefix = f"grasp_{index}"
        rotation = grasp.rotation_matrix
        quat = _rotation_to_quat(rotation)
        rgba = grasp.score_rgba()
        parts = ("left", "right", "bottom", "tail")

        for part_name, (local_center, half_size) in zip(
            parts,
            _gripper_part_boxes(grasp.width, grasp.depth, grasp.height),
        ):
            world_center = grasp.translation + rotation @ local_center
            geom = body.add_geom(
                name=f"{prefix}_{part_name}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=world_center.tolist(),
                quat=quat.tolist(),
                size=half_size.tolist(),
                rgba=list(rgba),
            )
            geom.contype = 0
            geom.conaffinity = 0

        if show_axes:
            center = grasp.translation
            axes = (
                ("x", rotation[:, 0], [1.0, 0.15, 0.15, 0.9]),
                ("y", rotation[:, 1], [0.15, 1.0, 0.15, 0.9]),
                ("z", rotation[:, 2], [0.15, 0.35, 1.0, 0.9]),
            )
            for axis_name, direction, axis_rgba in axes:
                endpoint = center + direction * _GRASP_AXIS_LENGTH
                geom = body.add_geom(
                    name=f"{prefix}_axis_{axis_name}",
                    type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                    fromto=[*center, *endpoint],
                    size=[0.0015, 0.0, 0.0],
                    rgba=axis_rgba,
                )
                geom.contype = 0
                geom.conaffinity = 0


def demo_grasps() -> list[GraspPose]:
    """Single red grasp derived from the SO-101 home pose."""

    import mujoco

    from mujoco_tasks.envs.scene import HOME_QPOS, build_grasp_viz_model
    from mujoco_tasks.motion.grasp_to_gripper import measure_grasp_pose

    model = build_grasp_viz_model(show_grasp=False)
    data = mujoco.MjData(model)
    grasp = measure_grasp_pose(model, data, np.asarray(HOME_QPOS, dtype=np.float64), score=1.0)
    return [grasp]
