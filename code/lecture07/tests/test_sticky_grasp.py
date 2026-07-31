from __future__ import annotations

import sys
import unittest
from pathlib import Path

import mujoco

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "simulation"))

from mujoco_tasks.motion.sticky_grasp import StickyGraspAssist  # pyright: ignore[reportMissingImports]


def _grasp_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <site name="eef_site" pos="0 0 0"/>
            <body name="fixed_jaw" pos="-0.05 0 0">
              <geom name="so101_gripper_fixed_jaw_collision"
                    type="box" size="0.03 0.03 0.03"/>
            </body>
            <body name="moving_jaw" pos="0.05 0 0">
              <joint name="so101_gripper" type="hinge"/>
              <geom name="so101_gripper_moving_jaw_collision"
                    type="box" size="0.03 0.03 0.03"/>
            </body>
            <body name="cube">
              <freejoint name="cube_free"/>
              <geom name="cube_geom" type="box" size="0.03 0.03 0.03" mass="0.08"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )


class StickyGraspTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = _grasp_model()
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

    def test_both_jaw_contacts_attach_and_opening_releases(self) -> None:
        assist = StickyGraspAssist(self.model, "cube", contact_steps=1, release_open_delta_rad=0.10)

        event = assist.observe_contacts(self.data, gripper_target=0.0)

        self.assertIsNotNone(event)
        self.assertEqual(event.action, "attached")
        self.assertTrue(assist.attached)
        self.assertEqual(assist.grasp_width_rad, 0.0)
        self.assertEqual(assist.clamp_gripper_target(-0.05), 0.0)
        cube_geom_id = self.model.geom("cube_geom").id
        self.assertEqual(self.model.geom_contype[cube_geom_id], 0)

        self.assertIsNone(assist.follow_or_release(self.data, gripper_target=0.05))
        event = assist.follow_or_release(self.data, gripper_target=0.10)

        self.assertIsNotNone(event)
        self.assertEqual(event.action, "released")
        self.assertFalse(assist.attached)
        self.assertNotEqual(self.model.geom_contype[cube_geom_id], 0)

    def test_insufficient_penetration_does_not_attach(self) -> None:
        assist = StickyGraspAssist(self.model, "cube", contact_steps=1, min_penetration=0.10)

        self.assertIsNone(assist.observe_contacts(self.data, gripper_target=0.0))
        self.assertFalse(assist.attached)


if __name__ == "__main__":
    unittest.main()
