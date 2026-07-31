"""Contact-triggered sticky grasp assistance for keyboard teleoperation."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from ..envs.utils.collision import GRIPPER_JAW_COLLISION_GEOM_NAMES, ROBOT_PREFIX


@dataclass(frozen=True, slots=True)
class StickyGraspEvent:
    action: str
    task: str

    def format(self) -> str:
        return f"STICKY_GRASP_{self.action.upper()} task={self.task}"


class StickyGraspAssist:
    """Kinematically hold an object after both jaws have established contact."""

    _OBJECT_GEOMS = {
        "cube": ("cube_geom",),
        "bottle": ("bottle_body_geom",),
    }

    def __init__(
        self,
        model: mujoco.MjModel,
        task: str,
        *,
        contact_steps: int = 3,
        min_penetration: float = 0.001,
        release_open_delta_rad: float = 0.10,
    ) -> None:
        if task not in self._OBJECT_GEOMS:
            raise ValueError(f"unsupported task {task!r}")
        if contact_steps < 1:
            raise ValueError("contact_steps must be positive")
        if min_penetration < 0.0:
            raise ValueError("min_penetration must be non-negative")

        self.model = model
        self.task = task
        self.contact_steps = contact_steps
        self.min_penetration = min_penetration
        self.release_open_delta_rad = release_open_delta_rad
        self._jaw_geom_ids = frozenset(
            model.geom(f"{ROBOT_PREFIX}{name}").id for name in GRIPPER_JAW_COLLISION_GEOM_NAMES
        )
        self._object_geom_ids = tuple(model.geom(name).id for name in self._OBJECT_GEOMS[task])
        free_joint = model.joint(f"{task}_free")
        self._object_qpos_adr = int(free_joint.qposadr[0])
        self._object_dof_adr = int(free_joint.dofadr[0])
        self._eef_site_id = model.site("eef_site").id
        self._gripper_qpos_adr = int(model.joint(f"{ROBOT_PREFIX}gripper").qposadr[0])

        self._contact_count = 0
        self._attached = False
        self._grasp_width_rad = 0.0
        self._relative_pos = np.zeros(3, dtype=np.float64)
        self._relative_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self._saved_collision: dict[int, tuple[int, int]] = {}

    @property
    def attached(self) -> bool:
        return self._attached

    @property
    def grasp_width_rad(self) -> float:
        """Actual gripper joint angle measured when the object was attached."""

        return self._grasp_width_rad

    def clamp_gripper_target(self, gripper_target: float) -> float:
        """Prevent further closing past the measured grasp width while attached."""

        return max(gripper_target, self._grasp_width_rad) if self._attached else gripper_target

    def observe_contacts(self, data: mujoco.MjData, gripper_target: float) -> StickyGraspEvent | None:
        """Attach after both jaws penetrate the same object consecutively."""

        if self._attached:
            return None

        penetrating_jaws = {
            jaw_id
            for contact in data.contact[: data.ncon]
            if float(contact.dist) <= -self.min_penetration
            for jaw_id in self._contact_jaws(contact.geom1, contact.geom2)
        }
        self._contact_count = self._contact_count + 1 if penetrating_jaws == self._jaw_geom_ids else 0
        if self._contact_count < self.contact_steps:
            return None

        self._attach(data, gripper_target)
        return StickyGraspEvent("attached", self.task)

    def follow_or_release(self, data: mujoco.MjData, gripper_target: float) -> StickyGraspEvent | None:
        """Follow the end effector while attached; detach after opening the gripper."""

        if not self._attached:
            return None
        if gripper_target >= self._grasp_width_rad + self.release_open_delta_rad:
            self._detach(data)
            return StickyGraspEvent("released", self.task)

        eef_pos, eef_quat = self._eef_pose(data)
        rotation = _quat_to_mat(eef_quat)
        qpos = data.qpos
        qpos[self._object_qpos_adr : self._object_qpos_adr + 3] = eef_pos + rotation @ self._relative_pos
        qpos[self._object_qpos_adr + 3 : self._object_qpos_adr + 7] = _quat_mul(eef_quat, self._relative_quat)
        data.qvel[self._object_dof_adr : self._object_dof_adr + 6] = 0.0
        mujoco.mj_forward(self.model, data)
        return None

    def _contact_jaws(self, geom1: int, geom2: int) -> tuple[int, ...]:
        if geom1 in self._jaw_geom_ids and geom2 in self._object_geom_ids:
            return (geom1,)
        if geom2 in self._jaw_geom_ids and geom1 in self._object_geom_ids:
            return (geom2,)
        return ()

    def _attach(self, data: mujoco.MjData, gripper_target: float) -> None:
        del gripper_target
        eef_pos, eef_quat = self._eef_pose(data)
        object_pos = data.qpos[self._object_qpos_adr : self._object_qpos_adr + 3].copy()
        object_quat = data.qpos[self._object_qpos_adr + 3 : self._object_qpos_adr + 7].copy()
        self._relative_pos = _quat_to_mat(eef_quat).T @ (object_pos - eef_pos)
        self._relative_quat = _quat_mul(_quat_conjugate(eef_quat), object_quat)
        self._grasp_width_rad = float(data.qpos[self._gripper_qpos_adr])
        self._contact_count = 0
        self._attached = True
        self._saved_collision = {
            geom_id: (
                int(self.model.geom_contype[geom_id]),
                int(self.model.geom_conaffinity[geom_id]),
            )
            for geom_id in self._object_geom_ids
        }
        for geom_id in self._object_geom_ids:
            self.model.geom_contype[geom_id] = 0
            self.model.geom_conaffinity[geom_id] = 0

    def _detach(self, data: mujoco.MjData) -> None:
        for geom_id, (contype, conaffinity) in self._saved_collision.items():
            self.model.geom_contype[geom_id] = contype
            self.model.geom_conaffinity[geom_id] = conaffinity
        self._saved_collision.clear()
        data.qvel[self._object_dof_adr : self._object_dof_adr + 6] = 0.0
        self._attached = False
        self._contact_count = 0
        mujoco.mj_forward(self.model, data)

    def _eef_pose(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        quat = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quat, data.site_xmat[self._eef_site_id])
        return data.site_xpos[self._eef_site_id].copy(), quat


def _quat_conjugate(quat: np.ndarray) -> np.ndarray:
    result = np.asarray(quat, dtype=np.float64).copy()
    result[1:] *= -1.0
    return result


def _quat_mul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    result = np.empty(4, dtype=np.float64)
    mujoco.mju_mulQuat(result, left, right)
    return result / max(np.linalg.norm(result), 1e-12)


def _quat_to_mat(quat: np.ndarray) -> np.ndarray:
    matrix = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(matrix, quat)
    return matrix.reshape(3, 3)
