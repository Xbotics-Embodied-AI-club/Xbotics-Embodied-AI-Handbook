from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "simulation"))

from mujoco_tasks.envs.scene import HOME_QPOS, JOINT_NAMES, build_model
from mujoco_tasks.motion import IKConfig, SO101IKSolver, apply_delta_rotation, get_delta_dof_pos, get_pose_error
import mujoco


class SO101IKTest(unittest.TestCase):
  @classmethod
  def setUpClass(cls) -> None:
    cls.model = build_model("cube")
    cls.data = mujoco.MjData(cls.model)
    for idx, name in enumerate(JOINT_NAMES):
      cls.data.qpos[cls.model.joint(name).qposadr[0]] = HOME_QPOS[idx]
    mujoco.mj_forward(cls.model, cls.data)
    cls.solver = SO101IKSolver(
      cls.model,
      cls.data,
      config=IKConfig(max_steps=300, position_only=True),
    )

  def test_pose_error_is_zero_at_current_pose(self) -> None:
    pos, quat = self.solver.eef_pose()
    pos_error, rot_error = get_pose_error(pos, quat, pos, quat)
    self.assertLess(np.linalg.norm(pos_error), 1e-9)
    self.assertLess(np.linalg.norm(rot_error), 1e-9)

  def test_delta_dof_pos_uses_damped_least_squares(self) -> None:
    jacobian = self.solver.jacobian()
    delta_pose = np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    delta_q = get_delta_dof_pos(delta_pose, jacobian)
    self.assertEqual(delta_q.shape, (5,))
    self.assertTrue(np.isfinite(delta_q).all())

  def test_move_to_pose_reduces_position_error(self) -> None:
    home_pos, home_quat = self.solver.eef_pose()
    target_pos = home_pos + np.array([0.02, 0.0, 0.03], dtype=np.float64)
    result = self.solver.move_to_pose(
      target_pos,
      home_quat,
      gripper_qpos=HOME_QPOS[-1],
    )
    final_pos, _ = self.solver.eef_pose()
    initial_error = np.linalg.norm(target_pos - home_pos)
    final_error = np.linalg.norm(target_pos - final_pos)
    self.assertLess(final_error, initial_error)
    self.assertLess(result.pos_error, 0.02)

  def test_move_to_pose_reduces_rotation_error(self) -> None:
    solver = SO101IKSolver(
      self.model,
      self.data,
      config=IKConfig(max_steps=300, position_only=False, rot_weight=0.35),
    )
    home_pos, home_quat = solver.eef_pose()
    target_quat = apply_delta_rotation(
      home_quat,
      np.array([0.0, 1.0, 0.0]),
      0.35,
      frame="tool",
    )
    result = solver.move_to_pose(home_pos, target_quat, gripper_qpos=HOME_QPOS[-1])
    _, final_quat = solver.eef_pose()
    _, initial_rot = get_pose_error(home_pos, home_quat, home_pos, target_quat)
    _, final_rot = get_pose_error(home_pos, final_quat, home_pos, target_quat)
    self.assertLess(np.linalg.norm(final_rot), np.linalg.norm(initial_rot))
    self.assertLess(result.rot_error, np.linalg.norm(initial_rot))


if __name__ == "__main__":
  unittest.main()
