from __future__ import annotations

import unittest

from robot_pick_place.config import cube_task
from robot_pick_place.models import FailureCode
from robot_pick_place.recovery import adjust_task


class RecoveryTest(unittest.TestCase):
    def test_place_failure_adjusts_place_and_retreat_heights(self) -> None:
        task = cube_task()

        adjusted = adjust_task(task, FailureCode.PLACE_FAILED, retry_index=1)

        self.assertAlmostEqual(adjusted.place_pose.z, task.place_pose.z + 0.01)
        self.assertAlmostEqual(adjusted.retreat_height, task.retreat_height + 0.02)


if __name__ == "__main__":
    unittest.main()
