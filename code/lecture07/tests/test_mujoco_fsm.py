"""Integration tests for the MuJoCo pick-place state machine."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import mujoco

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "simulation"))

from mujoco_tasks.envs.scene import HOME_QPOS, JOINT_NAMES, build_model
from mujoco_tasks.fsm_backend import MuJoCoFSMBackend
from mujoco_tasks.pose_targets import make_five_targets, make_task_config

from mujoco_pick_place import patch_state_machine_pose_generation
from robot_pick_place.state_machine import PickPlaceStateMachine
from robot_pick_place.task_logger import TaskLogger


class MuJoCoFSMTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        patch_state_machine_pose_generation()

    def _run(self, task: str) -> bool:
        model = build_model(task)
        data = mujoco.MjData(model)
        for idx, name in enumerate(JOINT_NAMES):
            data.qpos[model.joint(name).qposadr[0]] = HOME_QPOS[idx]
        mujoco.mj_forward(model, data)
        task_config = make_task_config(task, model, data)

        with tempfile.TemporaryDirectory() as tmp:
            logger = TaskLogger(tmp, task_config.name)
            backend = MuJoCoFSMBackend(model, data, task)
            backend.connect()
            try:
                success = PickPlaceStateMachine(backend, task_config, logger).run()
            finally:
                backend.disconnect()
            records = [
                json.loads(line)
                for line in logger.events_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertTrue(any(record.get("event") == "success" for record in records))
        return success

    def test_cube_task_runs_in_mujoco(self) -> None:
        self.assertTrue(self._run("cube"))

    def test_bottle_task_runs_in_mujoco(self) -> None:
        self.assertTrue(self._run("bottle"))

    def test_five_targets_follow_the_scene_geometry(self) -> None:
        import numpy as np

        for task in ("cube", "bottle"):
            task_config = make_task_config(task)
            targets = make_five_targets(task_config)
            self.assertGreater(targets.lift.z, targets.grasp.z)
            self.assertGreater(targets.retreat.z, targets.place.z)
            pre_grasp = np.array(
                [targets.pre_grasp.x, targets.pre_grasp.y, targets.pre_grasp.z]
            )
            grasp = np.array([targets.grasp.x, targets.grasp.y, targets.grasp.z])
            self.assertAlmostEqual(
                float(np.linalg.norm(pre_grasp - grasp)),
                task_config.pre_grasp_distance,
            )

    def test_pose_markers_attach_to_the_scene(self) -> None:
        from mujoco_tasks.viz.pose_viz import build_pose_viz_model

        model = build_pose_viz_model("cube")
        self.assertIsNotNone(model.geom("graspnet_grasp_left"))
        self.assertIsNotNone(model.geom("graspnet_place_tail"))
        self.assertIsNotNone(model.geom("graspnet_grasp_axis_z"))


if __name__ == "__main__":
    unittest.main()
