from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "simulation"))

from mujoco_tasks.envs.scene import build_model
from mujoco_tasks.envs.utils.collision import (  # pyright: ignore[reportMissingImports]
    GRIPPER_CUBE_CONTACT_SOLIMP,
    GRIPPER_CUBE_CONTACT_SOLREF,
    GRIPPER_JAW_COLLISION_GEOM_NAMES,
    GRIPPER_OBJECT_CONTACT_FRICTION,
    GRIPPER_OBJECT_CONTACT_MARGIN,
    ROBOT_PREFIX,
)


class TaskContactPairTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = build_model("cube")

    def test_cube_has_a_pair_for_each_gripper_jaw(self) -> None:
        pairs = {
            (
                self.model.pair_geom1[index],
                self.model.pair_geom2[index],
            )
            for index in range(self.model.npair)
        }
        cube_geom_id = self.model.geom("cube_geom").id

        for jaw_name in GRIPPER_JAW_COLLISION_GEOM_NAMES:
            self.assertIn((self.model.geom(f"{ROBOT_PREFIX}{jaw_name}").id, cube_geom_id), pairs)

    def test_cube_pair_uses_compliant_parameters_and_margin(self) -> None:
        cube_geom_id = self.model.geom("cube_geom").id
        jaw_geom_id = self.model.geom(f"{ROBOT_PREFIX}{GRIPPER_JAW_COLLISION_GEOM_NAMES[0]}").id
        pair_index = next(
            index
            for index in range(self.model.npair)
            if self.model.pair_geom1[index] == jaw_geom_id and self.model.pair_geom2[index] == cube_geom_id
        )

        np.testing.assert_allclose(self.model.pair_solref[pair_index], GRIPPER_CUBE_CONTACT_SOLREF)
        np.testing.assert_allclose(self.model.pair_solimp[pair_index], GRIPPER_CUBE_CONTACT_SOLIMP)
        np.testing.assert_allclose(
            self.model.pair_friction[pair_index],
            GRIPPER_OBJECT_CONTACT_FRICTION,
        )
        self.assertAlmostEqual(self.model.pair_margin[pair_index], GRIPPER_OBJECT_CONTACT_MARGIN)


if __name__ == "__main__":
    unittest.main()
