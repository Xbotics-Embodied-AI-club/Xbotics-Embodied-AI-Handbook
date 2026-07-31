from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "simulation"))

import mujoco

from mujoco_tasks.envs.scene import HOME_QPOS, JOINT_NAMES, build_grasp_viz_model
from mujoco_tasks.motion.grasp_pose import GraspPose
from mujoco_tasks.motion.grasp_to_gripper import (
    grasp_alignment_error,
    grasp_to_eef_pose,
    grasp_width_to_gripper_qpos,
    measure_grasp_pose,
    wrist_roll_axis_world,
    wrist_roll_rotation_world,
)
from mujoco_tasks.motion.solver import IKConfig, SO101IKSolver
from mujoco_tasks.viz.grasp_viz import _gripper_part_boxes, demo_grasps


class GraspVisualizationTest(unittest.TestCase):
    def test_grasp_array_round_trip(self) -> None:
        grasp = GraspPose(
            score=0.8,
            width=0.05,
            height=0.004,
            depth=0.02,
            rotation_matrix=np.eye(3),
            translation=np.array([0.1, 0.0, 0.05]),
            object_id=3,
        )
        restored = GraspPose.from_array(grasp.to_array())
        np.testing.assert_allclose(restored.to_array(), grasp.to_array())

    def test_gripper_has_four_parts(self) -> None:
        self.assertEqual(len(_gripper_part_boxes(width=0.04, depth=0.02, jaw_height=0.004)), 4)

    def test_grasp_frame_matches_wrist_roll_frame(self) -> None:
        """Grasp rotation must equal the wrist-roll body rotation exactly."""
        model = build_grasp_viz_model()
        data = mujoco.MjData(model)
        grasp = measure_grasp_pose(model, data, np.asarray(HOME_QPOS, dtype=np.float64))
        wrist_rotation = wrist_roll_rotation_world(model, data)
        np.testing.assert_allclose(grasp.rotation_matrix, wrist_rotation, atol=1e-9)
        roll_axis = wrist_roll_axis_world(model, data)
        self.assertGreater(abs(float(np.dot(grasp.rotation_matrix[:, 2], roll_axis))), 0.99)

    def test_build_model_with_robot_mounted_grasp(self) -> None:
        model = build_grasp_viz_model(show_grasp=True)
        self.assertGreaterEqual(model.ngeom, 12)
        self.assertIsNotNone(model.geom("grasp_left"))
        self.assertEqual(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom"), -1)

    def test_grasp_maps_to_gripper_with_small_center_error(self) -> None:
        model = build_grasp_viz_model()
        data = mujoco.MjData(model)
        solver = SO101IKSolver(
            model,
            data,
            config=IKConfig(max_steps=1200, pos_tolerance=0.004, rot_tolerance=0.08),
        )
        gripper_range = tuple(model.joint(JOINT_NAMES[-1]).range)

        for grasp in demo_grasps():
            target_pos, target_quat = grasp_to_eef_pose(grasp)
            gripper_qpos = grasp_width_to_gripper_qpos(grasp.width, gripper_range=gripper_range)
            result = solver.move_to_pose(target_pos, target_quat, gripper_qpos=gripper_qpos)
            center_error = grasp_alignment_error(grasp, model, data)
            self.assertLess(result.pos_error, 0.02, msg=f"grasp score={grasp.score}")
            self.assertLess(center_error, 0.02, msg=f"grasp score={grasp.score}")


if __name__ == "__main__":
    unittest.main()
