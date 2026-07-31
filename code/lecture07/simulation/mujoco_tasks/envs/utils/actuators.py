"""Position-servo actuators for the Lecture 07 SO-101 MuJoCo model."""

from __future__ import annotations

from math import sqrt

import mujoco

from .collision import ROBOT_PREFIX

ARM_ACTUATOR_KP = 600.0
GRIPPER_ACTUATOR_KP = 100.0
JOINT_ACTUATOR_FRC_LIMIT = 1.0


def configure_robot_actuators(model: mujoco.MjModel) -> None:
    """Raise URDF effort limits so position servos are not clipped at +/-10 Nm."""

    for joint_id in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or ""
        if not joint_name.startswith(ROBOT_PREFIX):
            continue
        limit = JOINT_ACTUATOR_FRC_LIMIT
        model.jnt_actfrcrange[joint_id] = [-limit, limit]
        model.jnt_actfrclimited[joint_id] = 1


def actuator_kv(kp: float) -> float:
    """Critical damping for a position servo: kv = 2 * sqrt(kp)."""

    return 2.0 * sqrt(kp)


def add_position_actuator(spec: mujoco.MjSpec, joint_name: str, kp: float) -> None:
    """Add a joint position servo: force = kp * (ctrl - qpos) - kv * qvel."""

    kv = actuator_kv(kp)
    actuator = spec.add_actuator()
    actuator.name = f"act_{joint_name}"
    actuator.target = joint_name
    actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
    actuator.dyntype = mujoco.mjtDyn.mjDYN_NONE
    actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
    actuator.biastype = mujoco.mjtBias.mjBIAS_AFFINE
    actuator.gainprm[0] = kp
    actuator.biasprm[1] = -kp
    actuator.biasprm[2] = -kv
