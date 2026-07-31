from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "simulation"))

from mujoco_tasks.try_ik import (  # pyright: ignore[reportMissingImports]
    KEY_PAGE_DOWN,
    KEY_X,
    HeldKeyRepeater,
    TeleopState,
    _on_key,
)


class KeyboardTeleopTest(unittest.TestCase):
    def test_default_position_step_is_five_millimetres(self) -> None:
        state = TeleopState(
            target_pos=np.zeros(3),
            target_quat=np.array([1.0, 0.0, 0.0, 0.0]),
            gripper_qpos=0.0,
            home_pos=np.zeros(3),
            home_quat=np.array([1.0, 0.0, 0.0, 0.0]),
        )

        for _ in range(3):
            _on_key(KEY_PAGE_DOWN, state)

        np.testing.assert_allclose(state.target_pos, [0.0, 0.0, -0.015])

    def test_position_step_remains_configurable(self) -> None:
        state = TeleopState(
            target_pos=np.zeros(3),
            target_quat=np.array([1.0, 0.0, 0.0, 0.0]),
            gripper_qpos=0.0,
            home_pos=np.zeros(3),
            home_quat=np.array([1.0, 0.0, 0.0, 0.0]),
            pos_step=0.002,
        )

        _on_key(KEY_PAGE_DOWN, state)

        np.testing.assert_allclose(state.target_pos, [0.0, 0.0, -0.002])

    def test_held_key_repeater_applies_initial_and_periodic_steps(self) -> None:
        held_keys = {ord("Q")}
        repeater = HeldKeyRepeater(repeat_period=0.1)

        np.testing.assert_allclose(repeater.poll_translation(0.0, held_keys.__contains__), [0.0, 0.0, -1.0])
        np.testing.assert_allclose(repeater.poll_translation(0.05, held_keys.__contains__), [0.0, 0.0, 0.0])
        np.testing.assert_allclose(repeater.poll_translation(0.10, held_keys.__contains__), [0.0, 0.0, -1.0])

        held_keys.clear()
        np.testing.assert_allclose(repeater.poll_translation(0.11, held_keys.__contains__), [0.0, 0.0, 0.0])
        held_keys.add(ord("Q"))
        np.testing.assert_allclose(repeater.poll_translation(0.12, held_keys.__contains__), [0.0, 0.0, -1.0])

    def test_held_gripper_repeater_uses_radian_direction(self) -> None:
        held_keys = {KEY_X}
        repeater = HeldKeyRepeater(repeat_period=0.1)

        self.assertEqual(repeater.poll_gripper(0.0, held_keys.__contains__), -1.0)
        self.assertEqual(repeater.poll_gripper(0.05, held_keys.__contains__), 0.0)
        self.assertEqual(repeater.poll_gripper(0.10, held_keys.__contains__), -1.0)


if __name__ == "__main__":
    unittest.main()
