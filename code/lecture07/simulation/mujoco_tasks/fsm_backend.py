"""Concrete MuJoCo backend for the Lecture 07 pick-place state machine.

``MuJoCoFSMBackend`` implements the ``RobotBackend`` interface on top of the
real SO-101 MuJoCo scene:

- ``move_pose`` runs differential IK (``SO101IKSolver``) and steps the physics
  simulation with position-servo actuators, so the arm moves smoothly;
- ``set_gripper`` maps the target jaw width (m) to the gripper joint angle and
  closes/opens it while ``StickyGraspAssist`` detects both-jaw contact and
  attaches/releases the task object;
- ``get_observation`` reports the EEF pose, measured jaw opening, the free-joint
  object pose, the sticky attach state, and any unintended collision.

The state machine in ``robot_pick_place.state_machine`` is unchanged: it only
sees this backend through the common interface.
"""

from __future__ import annotations

from time import monotonic, sleep
from typing import Any

import mujoco
import numpy as np

from mujoco_tasks.envs.scene import HOME_QPOS, JOINT_NAMES, TABLE_TOP_Z
from mujoco_tasks.motion import IKConfig, SO101IKSolver
from mujoco_tasks.motion.grasp_to_gripper import (
    GRIPPER_JAW_GEOM_NAMES,
    grasp_width_to_gripper_qpos,
)
from mujoco_tasks.motion.sticky_grasp import StickyGraspAssist
from mujoco_tasks.pose_targets import object_body_info, task_layout

from robot_pick_place.backends.base import RobotBackend
from robot_pick_place.models import MotionResult, Observation, Pose, TaskConfig

GRIPPER_JOINT = "so101_gripper"
GRIPPER_ACTUATOR = "act_so101_gripper"
DT = 0.002


def _quat_nlerp(qa: np.ndarray, qb: np.ndarray, t: float) -> np.ndarray:
    """Normalised linear interpolation between two unit quaternions (wxyz)."""

    qa = np.asarray(qa, dtype=np.float64)
    qb = np.asarray(qb, dtype=np.float64)
    if float(np.dot(qa, qb)) < 0.0:
        qb = -qb
    interp = qa + float(t) * (qb - qa)
    norm = float(np.linalg.norm(interp))
    return interp / norm if norm > 1e-12 else qa


class MuJoCoFSMBackend(RobotBackend):
    """Run the pick-place state machine against the real MuJoCo simulation."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        task: str,
        *,
        sticky_grasp: bool = True,
        viewer: Any = None,
        verbose: bool = False,
        physics_substeps: int = 8,
        move_timeout: float = 30.0,
        settle_time: float = 0.7,
        pos_tolerance: float = 0.07,
        rot_tolerance: float = 0.5,
    ) -> None:
        self.model = model
        self.data = data
        self.task_name = task
        self.viewer = viewer
        self.verbose = verbose
        self.sticky_grasp = sticky_grasp
        self.physics_substeps = physics_substeps
        self.move_timeout = move_timeout
        self.settle_time = float(task_layout(task).get("settle_time", settle_time))
        self.pos_tolerance = pos_tolerance
        self.rot_tolerance = rot_tolerance

        self.solver = SO101IKSolver(
            model,
            data,
            config=IKConfig(
                max_steps=1500,
                pos_tolerance=0.012,
                rot_tolerance=0.18,
                rot_weight=0.5,
                max_joint_step=0.03,
            ),
        )
        self._joint_qpos_ids = np.array(
            [model.joint(name).qposadr[0] for name in JOINT_NAMES],
            dtype=np.int32,
        )
        self._joint_dof_ids = np.array(
            [model.joint(name).dofadr[0] for name in JOINT_NAMES],
            dtype=np.int32,
        )
        self._arm_qpos_ids = self._joint_qpos_ids[:5]
        self._arm_dof_ids = self._joint_dof_ids[:5]
        self._gripper_range = tuple(model.joint(GRIPPER_JOINT).range)
        self._gripper_actuator_id = model.actuator(GRIPPER_ACTUATOR).id
        self._jaw_ids = tuple(model.geom(name).id for name in GRIPPER_JAW_GEOM_NAMES)

        info = object_body_info(task)
        self._object_free_joint = info["free_joint"]
        self._object_geom = info["geom"]
        self._object_center_offset = np.asarray(info["center_offset"], dtype=np.float64)
        self._free_qpos_adr = int(model.joint(info["free_joint"]).qposadr[0])
        self._free_dof_adr = int(model.joint(info["free_joint"]).dofadr[0])

        self.task: TaskConfig | None = None
        self._sticky: StickyGraspAssist | None = None
        self._last_collision = False
        self._connected = False

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    def connect(self) -> None:
        self._connected = True
        self._hold_home()

    def disconnect(self) -> None:
        self._connected = False

    def _hold_home(self) -> None:
        self.data.qpos[self._joint_qpos_ids] = HOME_QPOS
        self.data.qvel[self._joint_dof_ids] = 0.0
        self.data.ctrl[:] = HOME_QPOS
        mujoco.mj_forward(self.model, self.data)

    def reset_task(self, task: TaskConfig) -> None:
        self.task = task
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self._joint_qpos_ids] = HOME_QPOS
        self.data.qvel[self._joint_dof_ids] = 0.0

        # Place the object at the task's initial pose (free joint origin below
        # the measured centre by the object's centre offset).
        obj_pos = np.asarray(
            [task.object_pose.x, task.object_pose.y, task.object_pose.z],
            dtype=np.float64,
        )
        self.data.qpos[self._free_qpos_adr : self._free_qpos_adr + 3] = (
            obj_pos - self._object_center_offset
        )
        self.data.qpos[self._free_qpos_adr + 3 : self._free_qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[self._free_dof_adr : self._free_dof_adr + 6] = 0.0

        # The linear width->qpos calibration is wrong for this rotating-jaw
        # gripper (it was fit to the parallel-jaw convention).  Open the
        # gripper to its real mechanical maximum so the fingers fully clear the
        # task object before the approach.
        open_qpos = self._gripper_range[1]
        self.data.qpos[self._joint_qpos_ids[-1]] = open_qpos
        self.data.ctrl[self._gripper_actuator_id] = open_qpos
        mujoco.mj_forward(self.model, self.data)

        self._last_collision = False
        self._sticky = (
            StickyGraspAssist(self.model, self.task_name)
            if self.sticky_grasp
            else None
        )

    # ------------------------------------------------------------------ #
    # observations
    # ------------------------------------------------------------------ #
    def get_observation(self) -> Observation:
        eef_pos, eef_quat = self.solver.eef_pose()
        obj_pose = self._object_pose()
        return Observation(
            timestamp=monotonic(),
            eef_pose=Pose(
                float(eef_pos[0]),
                float(eef_pos[1]),
                float(eef_pos[2]),
                float(eef_quat[1]),
                float(eef_quat[2]),
                float(eef_quat[3]),
                float(eef_quat[0]),
            ),
            gripper_width=self._measure_gripper_width(),
            object_pose=obj_pose,
            object_visible=True,
            object_attached=bool(self._sticky is not None and self._sticky.attached),
            collision=self._last_collision,
        )

    def _measure_gripper_width(self) -> float:
        return float(
            np.linalg.norm(
                self.data.geom_xpos[self._jaw_ids[0]]
                - self.data.geom_xpos[self._jaw_ids[1]]
            )
        )

    def _object_pose(self) -> Pose:
        qpos = self.data.qpos[self._free_qpos_adr : self._free_qpos_adr + 7]
        center = qpos[:3] + self._object_center_offset
        return Pose(
            float(center[0]),
            float(center[1]),
            float(center[2]),
            float(qpos[4]),
            float(qpos[5]),
            float(qpos[6]),
            float(qpos[3]),
        )

    # ------------------------------------------------------------------ #
    # motion
    # ------------------------------------------------------------------ #
    def move_pose(
        self,
        target: Pose,
        speed: float,
        timeout: float = 8.0,
    ) -> MotionResult:
        if not self._connected:
            return MotionResult(False, "backend is not connected")

        deadline = monotonic() + max(timeout, self.move_timeout * 0.25)
        target_pos = np.array([target.x, target.y, target.z], dtype=np.float64)
        target_quat = np.array(
            [target.qw, target.qx, target.qy, target.qz],
            dtype=np.float64,
        )

        gripper_target = float(self.data.ctrl[self._gripper_actuator_id])
        current_z = float(self.solver.eef_pose()[0][2])
        attached = bool(self._sticky is not None and self._sticky.attached)
        # While the object is held, it must never go below its resting height
        # at the placement target (the task's place pose) during the transit.
        min_object_z = (
            float(self.task.place_pose.z)
            if attached and self.task is not None
            else None
        )

        # Descents (approach / lift -> place) first hover above the target at
        # the current height; then the joint-space drive lowers the arm while a
        # height guard on the held object keeps it above its resting height
        # (the elbow cannot press it toward the table).
        if float(target.z) < current_z - 0.015:
            hover_target = np.array([target_pos[0], target_pos[1], current_z])
            hover_home = self._plan_from_home(hover_target, target_quat)
            if hover_home is not None:
                self._drive_joint_space(
                    hover_home,
                    gripper_target=gripper_target,
                    deadline=deadline,
                    speed=speed,
                    min_object_z=min_object_z,
                )

        home_target = self._plan_from_home(target_pos, target_quat)
        if home_target is not None and self._drive_joint_space(
            home_target,
            gripper_target=gripper_target,
            deadline=deadline,
            speed=speed,
            min_object_z=min_object_z,
        ):
            pass  # drive succeeded; height guard and final check below

        # Final height guard: the home-plan may leave the held object below its
        # resting height; raise the arm so it hovers above the target zone/bin.
        if attached and min_object_z is not None:
            obj = self._object_pose()
            if float(obj.z) < min_object_z - 0.002:
                eef_pos, _ = self.solver.eef_pose()
                offset = float(obj.z - eef_pos[2])
                raise_pos = np.array(
                    [eef_pos[0], eef_pos[1], min_object_z - offset + 0.005],
                    dtype=np.float64,
                )
                raise_home = self._plan_from_home(raise_pos, target_quat)
                if raise_home is not None:
                    self._drive_joint_space(
                        raise_home,
                        gripper_target=gripper_target,
                        deadline=deadline + 2.0,
                        speed=max(float(speed), 0.30),
                        min_object_z=min_object_z,
                    )

        pos_error, rot_error = self.solver.pose_error(target_pos, target_quat)
        if (
            float(np.linalg.norm(pos_error)) < 0.030
            and float(np.linalg.norm(rot_error)) < 0.30
        ):
            return MotionResult(True)

        # Fallback: direct incremental IK.  Unlike interpolated pose tracking,
        # stepping straight at the full target converges for large orientation
        # changes (the bottle's radial poses) instead of drifting away.
        replanned = False
        best_error = float("inf")
        stale_iterations = 0
        for _ in range(600):
            pos_error, rot_error = self.solver.pose_error(target_pos, target_quat)
            pos_norm = float(np.linalg.norm(pos_error))
            rot_norm = float(np.linalg.norm(rot_error))
            if pos_norm < 0.012 and rot_norm < 0.20:
                return MotionResult(True)
            if pos_norm < best_error - 0.0005:
                best_error = pos_norm
                stale_iterations = 0
            else:
                stale_iterations += 1
                if stale_iterations >= 150:
                    if pos_norm < self.pos_tolerance and rot_norm < self.rot_tolerance:
                        return MotionResult(True)
                    if not replanned:
                        replanned = True
                        home_target = self._plan_from_home(target_pos, target_quat)
                        if not self._drive_joint_space(
                            home_target,
                            gripper_target=gripper_target,
                            deadline=deadline + 2.0,
                            speed=max(float(speed), 0.30),
                        ):
                            return MotionResult(False, "IK replan failed")
                        pos_error, rot_error = self.solver.pose_error(
                            target_pos, target_quat
                        )
                        if (
                            float(np.linalg.norm(pos_error)) < self.pos_tolerance
                            and float(np.linalg.norm(rot_error)) < self.rot_tolerance
                        ):
                            return MotionResult(True)
                        best_error = float("inf")
                        stale_iterations = 0
                        continue
                    return MotionResult(
                        False,
                        f"IK did not converge: pos_err={pos_norm:.3f} rot_err={rot_norm:.3f}",
                    )
            self.solver.step_toward_pose(
                target_pos,
                target_quat,
                gripper_qpos=gripper_target,
                mode="kinematic",
            )
            self.data.ctrl[self._gripper_actuator_id] = gripper_target
            if self._sticky is not None and self._sticky.attached:
                self._sticky.follow_or_release(self.data, gripper_target)
            self._last_collision = self._probe_collision()
            self._sync_viewer()
            if self._last_collision:
                return MotionResult(False, "collision")
            if monotonic() > deadline:
                return MotionResult(False, "motion timeout")

        pos_error, rot_error = self.solver.pose_error(target_pos, target_quat)
        pos_norm = float(np.linalg.norm(pos_error))
        rot_norm = float(np.linalg.norm(rot_error))
        if pos_norm < self.pos_tolerance and rot_norm < self.rot_tolerance:
            return MotionResult(True)
        return MotionResult(
            False,
            f"move_pose failed: pos_err={pos_norm:.3f} rot_err={rot_norm:.3f}",
        )

    def _plan_from_home(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
    ) -> np.ndarray:
        """Joint configuration that solves the target pose from the home posture."""

        saved_qpos = self.data.qpos.copy()
        saved_qvel = self.data.qvel.copy()
        try:
            self.data.qpos[self._arm_qpos_ids] = HOME_QPOS[:5]
            self.data.qvel[self._arm_dof_ids] = 0.0
            mujoco.mj_forward(self.model, self.data)
            return self.solver.plan_joint_target(
                target_pos,
                target_quat,
                iterations=150,
            )
        finally:
            self.data.qpos[:] = saved_qpos
            self.data.qvel[:] = saved_qvel
            mujoco.mj_forward(self.model, self.data)

    def _drive_joint_space(
        self,
        joint_target: np.ndarray,
        *,
        gripper_target: float,
        deadline: float,
        speed: float,
        min_object_z: float | None = None,
    ) -> bool:
        """Steer the arm to a joint configuration with collision probing."""

        joint_target = np.asarray(joint_target, dtype=np.float64)
        # Pace the motion with the requested speed: fast (0.35) moves ~0.02
        # rad/step at 25 ms, slow (0.06) ~0.008 rad/step at 45 ms.
        step = float(np.clip(0.01 + float(speed) * 0.05, 0.01, 0.03))
        pace = float(np.clip(0.03 - float(speed) * 0.05, 0.005, 0.03))
        for _ in range(600):
            current = self.data.qpos[self._arm_qpos_ids]
            delta = np.clip(joint_target - current, -step, step)
            if float(np.max(np.abs(delta))) < 1e-4:
                return True
            next_q = current + delta
            if min_object_z is not None:
                # Height guard on the held OBJECT: if this joint step would
                # carry the object below its target height, raise the arm
                # instead of dipping it into the table or box (equivalent to
                # rotating the shoulder first during a transit).
                self.data.qpos[self._arm_qpos_ids] = next_q
                self.data.qvel[self._arm_dof_ids] = 0.0
                mujoco.mj_forward(self.model, self.data)
                obj_z = float(self._object_pose().z)
                if obj_z < min_object_z - 1e-4:
                    eef_pos, eef_quat = self.solver.eef_pose()
                    obj_offset_z = float(self._object_pose().z - eef_pos[2])
                    raise_pos = np.array(
                        [
                            eef_pos[0],
                            eef_pos[1],
                            min_object_z - obj_offset_z + 0.02,
                        ],
                        dtype=np.float64,
                    )
                    self.solver.step_toward_pose(
                        raise_pos,
                        eef_quat,
                        gripper_qpos=gripper_target,
                        mode="kinematic",
                    )
                    self.data.qvel[self._arm_dof_ids] = 0.0
                    self.data.ctrl[: len(self._arm_qpos_ids)] = self.data.qpos[
                        self._arm_qpos_ids
                    ]
                    if self._sticky is not None and self._sticky.attached:
                        self._sticky.follow_or_release(self.data, gripper_target)
                    if self._probe_collision():
                        return False
                    self._sync_viewer()
                    sleep(pace)
                    if monotonic() > deadline:
                        return False
                    continue
            self.data.qpos[self._arm_qpos_ids] = next_q
            self.data.qvel[self._arm_dof_ids] = 0.0
            self.data.ctrl[: len(self._arm_qpos_ids)] = next_q
            if self._sticky is not None and self._sticky.attached:
                self._sticky.follow_or_release(self.data, gripper_target)
            if self._probe_collision():
                return False
            self._sync_viewer()
            sleep(pace)
            if monotonic() > deadline:
                return False
        return True

    def _probe_collision(self) -> bool:
        """Compute contacts at the current pose without moving the simulation."""

        saved_qpos = self.data.qpos.copy()
        saved_qvel = self.data.qvel.copy()
        try:
            mujoco.mj_step(self.model, self.data)
            return self._detect_collision()
        finally:
            self.data.qpos[:] = saved_qpos
            self.data.qvel[:] = saved_qvel
            mujoco.mj_forward(self.model, self.data)

    def set_gripper(
        self,
        width: float,
        force: float = 0.5,
        timeout: float = 8.0,
    ) -> MotionResult:
        del force
        if self.task is None:
            return MotionResult(False, "task has not been reset")

        target_qpos = grasp_width_to_gripper_qpos(
            width,
            gripper_range=self._gripper_range,
        )
        if self.task is not None and width >= self.task.gripper_open_width - 0.005:
            target_qpos = self._gripper_range[1]
        current = float(self.data.qpos[self._joint_qpos_ids[-1]])
        closing = target_qpos < current - 1e-4
        was_attached = bool(self._sticky is not None and self._sticky.attached)
        deadline = monotonic() + max(timeout, 8.0)
        arm_target = self.data.qpos[self._arm_qpos_ids].copy()

        if closing:
            # Physics-driven close with the arm held kinematically.  The
            # gripper command ramps down gradually so the rotating fingers
            # press onto the object instead of sweeping it away; the original
            # StickyGraspAssist attaches once both jaws penetrate reliably.
            stalled = 0
            last_current = current
            command = current
            while command - target_qpos > 1e-4:
                command = max(target_qpos, command - 0.006)
                if self._sticky is not None and self._sticky.attached:
                    command = self._sticky.clamp_gripper_target(command)
                self.data.ctrl[self._gripper_actuator_id] = command
                if self._step_gripper_physics(
                    gripper_target=command,
                    arm_target=arm_target,
                    observe=True,
                ):
                    return MotionResult(False, "collision")
                current = float(self.data.qpos[self._joint_qpos_ids[-1]])
                if abs(current - last_current) < 0.002:
                    stalled += 1
                    if stalled >= 40:
                        break
                else:
                    stalled = 0
                    last_current = current
                if monotonic() > deadline:
                    return MotionResult(False, "gripper timeout")
        else:
            # Physics-driven open with the arm held, so the fingers rotate away
            # from the object instead of clipping through it.
            command = current
            while target_qpos - command > 1e-4:
                command = min(target_qpos, command + 0.006)
                self.data.ctrl[self._gripper_actuator_id] = command
                if self._step_gripper_physics(
                    gripper_target=command,
                    arm_target=arm_target,
                    observe=False,
                ):
                    return MotionResult(False, "collision")
                if monotonic() > deadline:
                    return MotionResult(False, "gripper timeout")

        # After releasing an attached object, let it settle so the place check
        # sees the object resting in the target zone / bin.
        if not closing and was_attached and self._sticky is not None:
            self._settle()
        return MotionResult(True)

    def _step_gripper_physics(
        self,
        *,
        gripper_target: float,
        arm_target: np.ndarray,
        observe: bool,
    ) -> bool:
        """Step the simulation while holding the arm at its kinematic target.

        The arm joints are pinned back to ``arm_target`` after every physics
        step, so the servo reaction of the closing gripper cannot swing the arm;
        the gripper joint itself evolves under its actuator and the fingers
        stop against the task object (real contact, no interpenetration).
        """

        for _ in range(self.physics_substeps):
            if self._sticky is not None and self._sticky.attached:
                self._sticky.follow_or_release(self.data, gripper_target)
            mujoco.mj_step(self.model, self.data)
            self.data.qpos[self._arm_qpos_ids] = arm_target
            self.data.qvel[self._arm_dof_ids] = 0.0
            mujoco.mj_forward(self.model, self.data)
            if observe and self._sticky is not None:
                event = self._sticky.observe_contacts(self.data, gripper_target)
                if event is not None and event.action == "attached":
                    if self.verbose:
                        print(f"[fsm] {event.format()}")
                    self.data.ctrl[self._gripper_actuator_id] = self._sticky.grasp_width_rad
        self._last_collision = self._detect_collision()
        self._sync_viewer()
        return self._last_collision

    def _step_physics(self, *, gripper_target: float) -> bool:
        """Step the simulation and return True on an unintended collision."""

        for _ in range(self.physics_substeps):
            if self._sticky is not None:
                if self._sticky.attached:
                    self._sticky.follow_or_release(self.data, gripper_target)
                event = self._sticky.observe_contacts(self.data, gripper_target)
                if event is not None and event.action == "attached":
                    if self.verbose:
                        print(f"[fsm] {event.format()}")
                    self.data.ctrl[self._gripper_actuator_id] = self._sticky.grasp_width_rad
            mujoco.mj_step(self.model, self.data)
        self._last_collision = self._detect_collision()
        self._sync_viewer()
        return self._last_collision

    def _settle(self) -> None:
        """Step physics so a released object falls into its resting pose."""

        # The object may be released a few mm below the table when the grasp
        # approach stopped short; lift it onto the table first so the settle
        # does not pop it out of the surface and slide it sideways.
        min_origin_z = TABLE_TOP_Z
        if self.data.qpos[self._free_qpos_adr + 2] < min_origin_z:
            self.data.qpos[self._free_qpos_adr + 2] = min_origin_z
            self.data.qvel[self._free_dof_adr : self._free_dof_adr + 6] = 0.0
            mujoco.mj_forward(self.model, self.data)
        steps = int(self.settle_time / DT)
        gripper_target = float(self.data.ctrl[self._gripper_actuator_id])
        for _ in range(steps):
            # Keep following only while the object is still attached; do not
            # re-observe contacts here, otherwise a falling object would be
            # re-attached while passing the jaws.
            if self._sticky is not None and self._sticky.attached:
                self._sticky.follow_or_release(self.data, gripper_target)
            mujoco.mj_step(self.model, self.data)
        self._last_collision = self._detect_collision()
        self._sync_viewer()

    def _detect_collision(self) -> bool:
        """Flag unintended robot-object contacts.

        Robot-robot self contacts are ignored because the SO-101 convex
        collision proxies overlap slightly at the home pose (a model artifact).
        Table/floor and jaw-object contacts are expected, so they are excluded
        as well.
        """

        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            name1 = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1
            ) or ""
            name2 = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2
            ) or ""
            robot_side = name1.startswith("so101_") or name2.startswith("so101_")
            if not robot_side:
                continue
            if name1.startswith("so101_") and name2.startswith("so101_"):
                continue
            pair = {name1, name2}
            if pair & {"tabletop_geom", "floor"}:
                continue
            if any(
                "gripper" in name or "finger" in name or "jaw" in name
                for name in pair
            ):
                continue
            return True
        return False

    def stop(self) -> None:
        """Hold the current joint targets so the arm stops in place."""

        if not self._connected:
            return
        hold = self.data.qpos[self._joint_qpos_ids].copy()
        self.data.ctrl[:] = hold
        for _ in range(self.physics_substeps):
            mujoco.mj_step(self.model, self.data)
        self._sync_viewer()

    def return_home(self, timeout: float = 12.0) -> MotionResult:
        """Smoothly drive the arm back to the home joint posture."""

        if not self._connected:
            return MotionResult(False, "backend is not connected")

        target = np.asarray(HOME_QPOS[:5], dtype=np.float64)
        gripper_target = float(self.data.ctrl[self._gripper_actuator_id])
        deadline = monotonic() + timeout
        for _ in range(600):
            current = self.data.qpos[self._arm_qpos_ids]
            delta = np.clip(target - current, -0.02, 0.02)
            if float(np.max(np.abs(delta))) < 1e-3:
                break
            next_q = current + delta
            self.data.qpos[self._arm_qpos_ids] = next_q
            self.data.qvel[self._arm_dof_ids] = 0.0
            self.data.ctrl[: len(self._arm_qpos_ids)] = next_q
            if self._sticky is not None and self._sticky.attached:
                self._sticky.follow_or_release(self.data, gripper_target)
            mujoco.mj_forward(self.model, self.data)
            self._sync_viewer()
            sleep(0.02)
            if monotonic() > deadline:
                return MotionResult(False, "return home timeout")

        self._hold_home()
        if self._sticky is not None and self._sticky.attached:
            self._sticky.follow_or_release(self.data, float(self.data.ctrl[self._gripper_actuator_id]))
        self._sync_viewer()
        return MotionResult(True)

    def _sync_viewer(self) -> None:
        if self.viewer is not None:
            self.viewer.sync()


__all__ = ["MuJoCoFSMBackend"]
