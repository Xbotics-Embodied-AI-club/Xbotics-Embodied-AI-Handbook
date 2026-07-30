from __future__ import annotations

import unittest

from robot_pick_place.config import bottle_task, cube_task
from robot_pick_place.pose_generator import generate_targets


class PoseGeneratorTest(unittest.TestCase):
    def test_cube_pre_grasp_is_above_grasp(self) -> None:
        task = cube_task()
        targets = generate_targets(task)
        self.assertAlmostEqual(targets.pre_grasp.x, targets.grasp.x)
        self.assertAlmostEqual(targets.pre_grasp.y, targets.grasp.y)
        self.assertGreater(targets.pre_grasp.z, targets.grasp.z)

    def test_bottle_pre_grasp_is_opposite_approach_direction(self) -> None:
        task = bottle_task()
        targets = generate_targets(task)
        self.assertGreater(targets.pre_grasp.x, targets.grasp.x)
        self.assertAlmostEqual(targets.pre_grasp.y, targets.grasp.y)
        self.assertAlmostEqual(targets.pre_grasp.z, targets.grasp.z)

    def test_lift_and_retreat_are_higher_than_contact_poses(self) -> None:
        for task in (cube_task(), bottle_task()):
            targets = generate_targets(task)
            self.assertGreater(targets.lift.z, targets.grasp.z)
            self.assertGreater(targets.retreat.z, targets.place.z)


if __name__ == "__main__":
    unittest.main()
