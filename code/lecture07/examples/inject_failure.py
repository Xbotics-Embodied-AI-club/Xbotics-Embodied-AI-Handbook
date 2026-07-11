from __future__ import annotations

from pathlib import Path

from robot_pick_place.backends.mock import MockBackend
from robot_pick_place.config import get_task
from robot_pick_place.models import MotionResult, Pose
from robot_pick_place.state_machine import PickPlaceStateMachine
from robot_pick_place.task_logger import TaskLogger


class FailFirstApproachBackend(MockBackend):
    """The first approach command fails once, then later attempts run normally."""

    def __init__(self) -> None:
        super().__init__()
        self.failed_once = False

    def move_pose(
        self,
        target: Pose,
        speed: float,
        timeout: float = 8.0,
    ) -> MotionResult:
        if not self.failed_once and speed <= 0.10:
            self.failed_once = True
            return MotionResult(False, "injected approach failure")
        return super().move_pose(target, speed, timeout)


def main() -> None:
    task = get_task("cube")
    backend = FailFirstApproachBackend()
    logger = TaskLogger(Path("runs"), f"{task.name}_failure_demo")

    backend.connect()
    try:
        success = PickPlaceStateMachine(backend, task, logger).run()
    finally:
        backend.disconnect()

    print(f"success={success}")
    print(f"log_dir={logger.task_dir}")


if __name__ == "__main__":
    main()
