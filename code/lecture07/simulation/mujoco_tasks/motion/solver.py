"""MuJoCo differential IK solver for the Lecture 07 SO-101 arm."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import mujoco
import numpy as np

from ..envs.scene import ARM_JOINT_NAMES, HOME_QPOS, JOINT_NAMES

from .control import get_delta_dof_pos, get_pose_error, nullspace_joint_delta


@dataclass(slots=True)
class IKConfig:
    ik_gain: float = 0.4
    null_gain: float = 0.6
    max_joint_step: float = 0.05
    pos_tolerance: float = 0.002
    rot_tolerance: float = 0.35
    max_steps: int = 200
    min_singular_value: float = 0.01
    dls_lambda: float = 0.25
    position_only: bool = False
    pos_weight: float = 1.0
    rot_weight: float = 0.35


@dataclass(slots=True)
class IKStepResult:
    joint_target: np.ndarray
    pos_error: float
    rot_error: float
    converged: bool


class SO101IKSolver:
    """Jacobian-based differential IK for the 5-DOF SO-101 arm in MuJoCo."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        eef_site_name: str = "eef_site",
        config: IKConfig | None = None,
    ) -> None:
        self.model = model
        self.data = data
        self.config = config or IKConfig()
        self.eef_site_id = model.site(eef_site_name).id
        self._arm_qpos_ids = np.array(
            [model.joint(name).qposadr[0] for name in ARM_JOINT_NAMES],
            dtype=np.int32,
        )
        self._arm_dof_ids = np.array(
            [model.joint(name).dofadr[0] for name in ARM_JOINT_NAMES],
            dtype=np.int32,
        )
        self._joint_qpos_ids = np.array(
            [model.joint(name).qposadr[0] for name in JOINT_NAMES],
            dtype=np.int32,
        )
        self._home_qpos = np.asarray(HOME_QPOS[: len(ARM_JOINT_NAMES)], dtype=np.float64)

    def arm_joint_positions(self) -> np.ndarray:
        return self.data.qpos[self._arm_qpos_ids].copy()

    def eef_pose(self) -> tuple[np.ndarray, np.ndarray]:
        pos = self.data.site_xpos[self.eef_site_id].copy()
        mat = self.data.site_xmat[self.eef_site_id].reshape(3, 3)
        quat = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quat, mat.reshape(-1))
        return pos, quat

    def jacobian(self) -> np.ndarray:
        jac_pos = np.zeros((3, self.model.nv), dtype=np.float64)
        jac_rot = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jacSite(self.model, self.data, jac_pos, jac_rot, self.eef_site_id)
        arm_cols = self._arm_dof_ids
        return np.vstack([jac_pos[:, arm_cols], jac_rot[:, arm_cols]])

    def pose_error(self, target_pos: np.ndarray, target_quat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        eef_pos, eef_quat = self.eef_pose()
        return get_pose_error(eef_pos, eef_quat, target_pos, target_quat)

    def solve_delta(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
    ) -> np.ndarray:
        pos_error, axis_angle_error = self.pose_error(target_pos, target_quat)
        jacobian = self.jacobian()
        joint_pos = self.arm_joint_positions()

        if self.config.position_only:
            delta_pose = pos_error
            task_jacobian = jacobian[:3, :]
        else:
            delta_pose = np.concatenate(
                [
                    self.config.pos_weight * pos_error,
                    self.config.rot_weight * axis_angle_error,
                ]
            )
            task_jacobian = jacobian

        delta_task = get_delta_dof_pos(
            delta_pose,
            task_jacobian,
            dls_lambda=self.config.dls_lambda,
        )
        delta_null = nullspace_joint_delta(
            task_jacobian,
            joint_pos,
            self._home_qpos,
            min_singular_value=self.config.min_singular_value,
        )
        joint_delta = self.config.ik_gain * delta_task + self.config.null_gain * delta_null
        joint_delta = np.clip(joint_delta, -self.config.max_joint_step, self.config.max_joint_step)
        return joint_delta

    def clip_to_limits(self, joint_target: np.ndarray) -> np.ndarray:
        clipped = np.asarray(joint_target, dtype=np.float64).copy()
        for idx, joint_name in enumerate(ARM_JOINT_NAMES):
            low, high = self.model.joint(joint_name).range
            clipped[idx] = np.clip(clipped[idx], low, high)
        return clipped

    def apply_joint_target(self, joint_target: np.ndarray, *, gripper_qpos: float | None = None) -> None:
        """Teleport the arm to a joint target (kinematic IK, no contact forces)."""

        joint_target = self.clip_to_limits(joint_target)
        self.data.qpos[self._arm_qpos_ids] = joint_target
        self.data.qvel[self._arm_dof_ids] = 0.0
        self.data.ctrl[: len(ARM_JOINT_NAMES)] = joint_target
        if gripper_qpos is not None:
            gripper_dof = self.model.joint(JOINT_NAMES[-1]).dofadr[0]
            self.data.qpos[self._joint_qpos_ids[-1]] = gripper_qpos
            self.data.qvel[gripper_dof] = 0.0
            self.data.ctrl[len(ARM_JOINT_NAMES)] = gripper_qpos
        mujoco.mj_forward(self.model, self.data)

    def apply_ctrl_target(
        self,
        joint_target: np.ndarray,
        *,
        gripper_qpos: float | None = None,
        slew_alpha: float = 1.0,
        gripper_slew_alpha: float | None = None,
    ) -> None:
        """Command the position actuators without overwriting simulated joint state."""

        joint_target = self.clip_to_limits(joint_target)
        if slew_alpha >= 1.0:
            self.data.ctrl[: len(ARM_JOINT_NAMES)] = joint_target
        else:
            current = self.data.ctrl[: len(ARM_JOINT_NAMES)].copy()
            self.data.ctrl[: len(ARM_JOINT_NAMES)] = current + slew_alpha * (joint_target - current)
        if gripper_qpos is not None:
            gripper_alpha = slew_alpha if gripper_slew_alpha is None else gripper_slew_alpha
            gripper_idx = len(ARM_JOINT_NAMES)
            if gripper_alpha >= 1.0:
                self.data.ctrl[gripper_idx] = gripper_qpos
            else:
                current_gripper = self.data.ctrl[gripper_idx]
                self.data.ctrl[gripper_idx] = current_gripper + gripper_alpha * (gripper_qpos - current_gripper)

    def plan_joint_target(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        *,
        iterations: int = 24,
        gripper_qpos: float | None = None,
    ) -> np.ndarray:
        """Run differential IK on the current pose and return a joint-space setpoint."""

        del gripper_qpos
        saved_qpos = self.data.qpos[self._arm_qpos_ids].copy()
        saved_qvel = self.data.qvel[self._arm_dof_ids].copy()
        joint_target = saved_qpos.copy()
        try:
            for _ in range(max(1, iterations)):
                self.data.qpos[self._arm_qpos_ids] = joint_target
                mujoco.mj_forward(self.model, self.data)
                joint_delta = self.solve_delta(target_pos, target_quat)
                joint_target = self.clip_to_limits(joint_target + joint_delta)
            return joint_target
        finally:
            self.data.qpos[self._arm_qpos_ids] = saved_qpos
            self.data.qvel[self._arm_dof_ids] = saved_qvel
            mujoco.mj_forward(self.model, self.data)

    def step_toward_pose(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        *,
        gripper_qpos: float | None = None,
        mode: Literal["kinematic", "actuator"] = "kinematic",
    ) -> IKStepResult:
        joint_pos = self.arm_joint_positions()
        joint_delta = self.solve_delta(target_pos, target_quat)
        joint_target = self.clip_to_limits(joint_pos + joint_delta)
        if mode == "kinematic":
            self.apply_joint_target(joint_target, gripper_qpos=gripper_qpos)
        else:
            self.apply_ctrl_target(joint_target, gripper_qpos=gripper_qpos)

        pos_error, axis_angle_error = self.pose_error(target_pos, target_quat)
        pos_err_norm = float(np.linalg.norm(pos_error))
        rot_err_norm = float(np.linalg.norm(axis_angle_error))
        if self.config.position_only:
            converged = pos_err_norm < self.config.pos_tolerance
        else:
            converged = pos_err_norm < self.config.pos_tolerance and rot_err_norm < self.config.rot_tolerance
        return IKStepResult(
            joint_target=joint_target,
            pos_error=pos_err_norm,
            rot_error=rot_err_norm,
            converged=converged,
        )

    def move_to_pose(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        *,
        gripper_qpos: float | None = None,
    ) -> IKStepResult:
        last_result = IKStepResult(
            joint_target=self.arm_joint_positions(),
            pos_error=np.inf,
            rot_error=np.inf,
            converged=False,
        )
        for _ in range(self.config.max_steps):
            last_result = self.step_toward_pose(target_pos, target_quat, gripper_qpos=gripper_qpos)
            if last_result.converged:
                break
        return last_result
