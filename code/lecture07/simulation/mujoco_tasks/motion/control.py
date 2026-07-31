"""Differential IK utilities for SO-101 (adapted from Isaac Lab factory_ur5e_control)."""

from __future__ import annotations

from typing import Literal

import numpy as np

def axis_angle_from_quat(quat: np.ndarray) -> np.ndarray:
    """Convert a unit quaternion (w, x, y, z) to axis-angle."""

    quat = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(quat)
    if norm < 1e-12:
        return np.zeros(3, dtype=np.float64)
    quat = quat / norm
    w = float(np.clip(quat[0], -1.0, 1.0))
    angle = 2.0 * np.arccos(w)
    sin_half = np.sqrt(max(0.0, 1.0 - w * w))
    if sin_half < 1e-8:
        return np.zeros(3, dtype=np.float64)
    return quat[1:4] / sin_half * angle


def quat_conjugate(quat: np.ndarray) -> np.ndarray:
    out = np.asarray(quat, dtype=np.float64).reshape(4).copy()
    out[1:] *= -1.0
    return out


def quat_mul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    import mujoco

    out = np.empty(4, dtype=np.float64)
    mujoco.mju_mulQuat(out, np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64))
    return out


def quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Build a unit quaternion (w, x, y, z) from axis-angle."""

    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    axis = axis / norm
    half = 0.5 * float(angle)
    sin_half = np.sin(half)
    return np.array([np.cos(half), *(axis * sin_half)], dtype=np.float64)


def apply_delta_rotation(
    quat: np.ndarray,
    axis: np.ndarray,
    angle: float,
    *,
    frame: Literal["tool", "world"] = "tool",
) -> np.ndarray:
    """Apply a small rotation to a quaternion in the tool or world frame."""

    delta = quat_from_axis_angle(axis, angle)
    current = np.asarray(quat, dtype=np.float64).reshape(4)
    if frame == "tool":
        rotated = quat_mul(current, delta)
    elif frame == "world":
        rotated = quat_mul(delta, current)
    else:
        raise ValueError(f"Unsupported rotation frame: {frame}")
    return rotated / max(np.linalg.norm(rotated), 1e-12)


def get_pose_error(
    eef_pos: np.ndarray,
    eef_quat: np.ndarray,
    target_pos: np.ndarray,
    target_quat: np.ndarray,
    *,
    rot_error_type: Literal["axis_angle", "quat"] = "axis_angle",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute 3D position error and rotation error in the TCP frame."""

    pos_error = np.asarray(target_pos, dtype=np.float64).reshape(3) - np.asarray(eef_pos, dtype=np.float64).reshape(3)

    current_quat = np.asarray(eef_quat, dtype=np.float64).reshape(4)
    desired_quat = np.asarray(target_quat, dtype=np.float64).reshape(4)
    if np.dot(desired_quat, current_quat) < 0.0:
        desired_quat = -desired_quat

    quat_error = quat_mul(desired_quat, quat_conjugate(current_quat))
    if rot_error_type == "quat":
        return pos_error, quat_error
    if rot_error_type == "axis_angle":
        return pos_error, axis_angle_from_quat(quat_error)
    raise ValueError(f"Unsupported rotation error type: {rot_error_type}")


def get_delta_dof_pos(
    delta_pose: np.ndarray,
    jacobian: np.ndarray,
    *,
    dls_lambda: float = 0.25,
) -> np.ndarray:
    """Map a TCP pose delta to arm joint deltas using damped least squares."""

    delta_pose = np.asarray(delta_pose, dtype=np.float64).reshape(-1)
    jacobian = np.asarray(jacobian, dtype=np.float64)
    task_dim, _ = jacobian.shape
    if delta_pose.shape[0] != task_dim:
        raise ValueError(f"delta_pose dim {delta_pose.shape[0]} != jacobian rows {task_dim}")

    lambda_matrix = (dls_lambda**2) * np.eye(task_dim, dtype=np.float64)
    jjt = jacobian @ jacobian.T + lambda_matrix
    delta_dof_pos = jacobian.T @ np.linalg.pinv(jjt, rcond=1e-5) @ delta_pose

    if not np.isfinite(delta_dof_pos).all():
        return np.zeros(jacobian.shape[1], dtype=np.float64)
    return delta_dof_pos


def nullspace_joint_delta(
    jacobian: np.ndarray,
    joint_pos: np.ndarray,
    home_qpos: np.ndarray,
    *,
    min_singular_value: float = 0.01,
) -> np.ndarray:
    """Pull redundant joints toward a preferred posture without moving the TCP."""

    jacobian = np.asarray(jacobian, dtype=np.float64)
    task_dim, num_joints = jacobian.shape
    joint_pos = np.asarray(joint_pos, dtype=np.float64).reshape(-1)
    home_qpos = np.asarray(home_qpos, dtype=np.float64).reshape(-1)
    if joint_pos.shape[0] != num_joints or home_qpos.shape[0] != num_joints:
        raise ValueError("joint vectors must match Jacobian column count")

    u_mat, singular_vals, vh_mat = np.linalg.svd(jacobian, full_matrices=True)
    singular_inv = np.zeros((num_joints, task_dim), dtype=np.float64)
    for idx in range(min(task_dim, singular_vals.shape[0])):
        singular_inv[idx, idx] = 1.0 / singular_vals[idx] if singular_vals[idx] > min_singular_value else 0.0
    jacobian_pinv = vh_mat.T @ singular_inv @ u_mat.T
    null_proj = np.eye(num_joints, dtype=np.float64) - jacobian_pinv @ jacobian

    q_home_err = home_qpos - joint_pos
    q_home_err = (q_home_err + np.pi) % (2.0 * np.pi) - np.pi
    return null_proj @ q_home_err
