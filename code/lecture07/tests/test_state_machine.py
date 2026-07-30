from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from robot_pick_place.backends.mock import MockBackend
from robot_pick_place.config import bottle_task, cube_task
from robot_pick_place.models import MotionResult, Pose
from robot_pick_place.state_machine import PickPlaceStateMachine
from robot_pick_place.task_logger import TaskLogger


class AlwaysFailBackend(MockBackend):
    def move_pose(
        self,
        target: Pose,
        speed: float,
        timeout: float = 8.0,
    ) -> MotionResult:
        del target, speed, timeout
        return MotionResult(False, "injected permanent failure")


class StateMachineTest(unittest.TestCase):
    def _run_task(self, task_name: str) -> tuple[bool, list[dict[str, object]]]:
        task = cube_task() if task_name == "cube" else bottle_task()
        with tempfile.TemporaryDirectory() as tmp:
            logger = TaskLogger(tmp, task.name)
            backend = MockBackend()
            backend.connect()
            try:
                success = PickPlaceStateMachine(backend, task, logger).run()
            finally:
                backend.disconnect()
            records = [
                json.loads(line)
                for line in logger.events_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        return success, records

    def test_cube_task_succeeds(self) -> None:
        success, records = self._run_task("cube")
        self.assertTrue(success)
        self.assertTrue(any(record.get("event") == "success" for record in records))

    def test_bottle_task_succeeds(self) -> None:
        success, records = self._run_task("bottle")
        self.assertTrue(success)
        self.assertTrue(any(record.get("state") == "VERIFY_PLACE" for record in records))

    def test_permanent_failure_enters_safe_exit(self) -> None:
        task = cube_task()
        task.max_retries = 1
        with tempfile.TemporaryDirectory() as tmp:
            logger = TaskLogger(tmp, task.name)
            backend = AlwaysFailBackend()
            backend.connect()
            try:
                success = PickPlaceStateMachine(backend, task, logger).run()
            finally:
                backend.disconnect()
            records = [
                json.loads(line)
                for line in logger.events_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        self.assertFalse(success)
        self.assertTrue(any(record.get("state") == "SAFE_EXIT" for record in records))


if __name__ == "__main__":
    unittest.main()
