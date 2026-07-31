"""Map GraspNet ``GraspPose`` targets to the SO-101 MuJoCo gripper (EEF + jaw width)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import numpy as np

if TYPE_CHECKING:
    from .grasp_pose import GraspPose

# Calibrated on the Lecture 07 SO-101 model at ``HOME_QPOS``:
# R_grasp == R_wrist_roll; R_eef = R_grasp @ GRASP_TO_EEF_ROTATION
GRASP_TO_EEF_ROTATION = np.array(
    [
        [-1.0, 0.0, 3e-06],
        [0.0, 1.0, 0.0],
        [-3e-06, 0.0, -1.0],
    ],
    dtype=np.float64,
)

# Vector from the EEF origin to the grasp center on the wrist-roll Z axis, in the grasp frame.
EEF_TO_GRASP_CENTER_IN_GRASP = np.array([0.0079, 0.0002, 0.0309], dtype=np.float64)

# Linear fit: gripper_qpos = GRIPPER_WIDTH_SLOPE * jaw_width + GRIPPER_WIDTH_INTERCEPT
GRIPPER_WIDTH_SLOPE = 48.52321976849343
GRIPPER_WIDTH_INTERCEPT = -2.1163345586561357

GRIPPER_JAW_GEOM_NAMES = (
    "so101_gripper_fixed_jaw_collision",
    "so101_gripper_moving_jaw_collision",
)


def _rotation_to_quat(rotation: np.ndarray) -> np.ndarray:
    quat = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, np.asarray(rotation, dtype=np.float64).reshape(9))
    return quat


def wrist_roll_rotation_world(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Return the wrist-roll body rotation (columns = wrist X, Y, Z in world frame)."""

    from mujoco_tasks.envs.scene import WRIST_ROLL_BODY_NAME

    return data.xmat[model.body(WRIST_ROLL_BODY_NAME).id].reshape(3, 3).copy()


def wrist_roll_axis_world(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Unit vector of grasp +Z: wrist-roll +Z on ``so101_gripper_link``."""

    return wrist_roll_rotation_world(model, data)[:, 2]


def grasp_to_eef_pose(grasp: GraspPose) -> tuple[np.ndarray, np.ndarray]:
    """Convert a GraspNet grasp to the SO-101 EEF position and quaternion."""

    rotation_grasp = grasp.rotation_matrix
    rotation_eef = rotation_grasp @ GRASP_TO_EEF_ROTATION
    position = grasp.translation - rotation_grasp @ EEF_TO_GRASP_CENTER_IN_GRASP
    return position, _rotation_to_quat(rotation_eef)


def grasp_width_to_gripper_qpos(
    width: float,
    *,
    gripper_range: tuple[float, float] | None = None,
) -> float:
    """Map a GraspNet jaw opening width (meters) to the SO-101 gripper joint angle."""

    joint_qpos = GRIPPER_WIDTH_SLOPE * float(width) + GRIPPER_WIDTH_INTERCEPT
    if gripper_range is None:
        return joint_qpos
    return float(np.clip(joint_qpos, gripper_range[0], gripper_range[1]))


def measure_grasp_center(
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> np.ndarray:
    """Return the fixed-jaw center projected onto the wrist-roll Z axis line."""

    from mujoco_tasks.envs.scene import WRIST_ROLL_BODY_NAME

    wrist_id = model.body(WRIST_ROLL_BODY_NAME).id
    wrist_pos = data.xpos[wrist_id]
    wrist_z = data.xmat[wrist_id].reshape(3, 3)[:, 2]
    fixed_pos = data.geom_xpos[model.geom(GRIPPER_JAW_GEOM_NAMES[0]).id]
    return wrist_pos + wrist_z * np.dot(fixed_pos - wrist_pos, wrist_z)


def measure_grasp_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_qpos: np.ndarray,
    *,
    score: float = 1.0,
    depth: float = 0.02,
    jaw_height: float = 0.004,
) -> GraspPose:
    """Build a ``GraspPose`` locked to the wrist-roll frame on ``so101_gripper_link``.

    ``rotation_matrix`` equals the wrist-roll body rotation exactly:
    - X, Z span the grasp plane (wrist-roll X-Z plane)
    - Y is the wrist-roll Y axis (plane normal)
    - Z is the wrist-roll rotation axis
    Approach toward the object is ``-Z``.
    """

    from .grasp_pose import GraspPose

    from mujoco_tasks.envs.scene import JOINT_NAMES

    joint_ids = [model.joint(name).qposadr[0] for name in JOINT_NAMES]
    data.qpos[joint_ids] = joint_qpos
    mujoco.mj_forward(model, data)

    fixed_id = model.geom(GRIPPER_JAW_GEOM_NAMES[0]).id
    moving_id = model.geom(GRIPPER_JAW_GEOM_NAMES[1]).id
    width_vector = data.geom_xpos[moving_id] - data.geom_xpos[fixed_id]
    width = float(np.linalg.norm(width_vector))

    return GraspPose(
        score=score,
        width=width,
        height=jaw_height,
        depth=depth,
        rotation_matrix=wrist_roll_rotation_world(model, data),
        translation=measure_grasp_center(model, data),
    )


def grasp_alignment_error(
    grasp: GraspPose,
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> float:
    """Euclidean error between the grasp center and the wrist-roll-axis projection."""

    return float(np.linalg.norm(measure_grasp_center(model, data) - grasp.translation))
