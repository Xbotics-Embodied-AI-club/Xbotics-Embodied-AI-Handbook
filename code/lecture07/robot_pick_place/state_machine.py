from __future__ import annotations

from dataclasses import dataclass

from .backends.base import RobotBackend
from .checker import grasp_success, lift_success, place_success
from .models import FailureCode, State, TaskConfig
from .pose_generator import generate_targets
from .recovery import adjust_task
from .task_logger import TaskLogger


@dataclass(frozen=True)
class AttemptResult:
    success: bool
    failure: FailureCode
    reason: str


class PickPlaceStateMachine:
    def __init__(
        self,
        backend: RobotBackend,
        task: TaskConfig,
        logger: TaskLogger,
    ) -> None:
        self.backend = backend
        self.task = task
        self.logger = logger
        self.state = State.INIT

    def _enter(self, state: State, **extra: object) -> None:
        self.state = state
        self.logger.log(
            state,
            "enter",
            self.backend.get_observation(),
            **extra,
        )

    def _fail(self, code: FailureCode, reason: str) -> AttemptResult:
        self.logger.log(
            self.state,
            "failure",
            self.backend.get_observation(),
            code=code.name,
            reason=reason,
        )
        return AttemptResult(False, code, reason)

    def run_attempt(self) -> AttemptResult:
        targets = generate_targets(self.task)
        initial_object_pose = self.task.object_pose

        self._enter(State.INIT)
        result = self.backend.set_gripper(self.task.gripper_open_width)
        if not result.success:
            return self._fail(FailureCode.MOTION_FAILED, result.reason)

        self._enter(State.MOVE_PRE_GRASP)
        result = self.backend.move_pose(
            targets.pre_grasp,
            self.task.fast_speed,
        )
        if not result.success:
            return self._fail(FailureCode.MOTION_FAILED, result.reason)

        self._enter(State.APPROACH)
        result = self.backend.move_pose(targets.grasp, self.task.slow_speed)
        if not result.success:
            code = FailureCode.COLLISION if "collision" in result.reason else (
                FailureCode.MOTION_FAILED
            )
            return self._fail(code, result.reason)

        self._enter(State.CLOSE_GRIPPER)
        result = self.backend.set_gripper(self.task.gripper_close_width)
        if not result.success:
            return self._fail(FailureCode.GRASP_FAILED, result.reason)

        self._enter(State.VERIFY_GRASP)
        if not grasp_success(self.backend.get_observation(), self.task):
            return self._fail(FailureCode.GRASP_FAILED, "grasp check failed")

        self._enter(State.LIFT)
        result = self.backend.move_pose(targets.lift, self.task.slow_speed)
        if not result.success:
            return self._fail(FailureCode.LIFT_FAILED, result.reason)

        self._enter(State.VERIFY_LIFT)
        if not lift_success(
            self.backend.get_observation(),
            initial_object_pose,
            self.task,
        ):
            return self._fail(FailureCode.LIFT_FAILED, "lift check failed")

        self._enter(State.MOVE_PRE_PLACE)
        result = self.backend.move_pose(targets.pre_place, self.task.fast_speed)
        if not result.success:
            return self._fail(FailureCode.MOTION_FAILED, result.reason)

        self._enter(State.LOWER_PLACE)
        result = self.backend.move_pose(targets.place, self.task.slow_speed)
        if not result.success:
            return self._fail(FailureCode.PLACE_FAILED, result.reason)

        self._enter(State.OPEN_GRIPPER)
        result = self.backend.set_gripper(self.task.gripper_open_width)
        if not result.success:
            return self._fail(FailureCode.PLACE_FAILED, result.reason)

        self._enter(State.VERIFY_PLACE)
        if not place_success(self.backend.get_observation(), self.task):
            return self._fail(FailureCode.PLACE_FAILED, "place check failed")

        self._enter(State.RETREAT)
        result = self.backend.move_pose(targets.retreat, self.task.slow_speed)
        if not result.success:
            return self._fail(FailureCode.MOTION_FAILED, result.reason)

        self._enter(State.DONE)
        self.logger.log(
            State.DONE,
            "success",
            self.backend.get_observation(),
        )
        return AttemptResult(True, FailureCode.NONE, "")

    def run(self) -> bool:
        current_task = self.task
        for retry_index in range(current_task.max_retries + 1):
            self.task = current_task
            self.backend.reset_task(current_task)
            self.logger.log(
                State.INIT,
                "attempt_start",
                self.backend.get_observation(),
                retry_index=retry_index,
            )
            result = self.run_attempt()
            if result.success:
                return True

            self.backend.stop()
            if retry_index >= current_task.max_retries:
                self._enter(State.SAFE_EXIT, reason=result.reason)
                return False

            self._enter(State.RECOVER, failure=result.failure.name)
            current_task = adjust_task(
                current_task,
                result.failure,
                retry_index + 1,
            )
        return False
