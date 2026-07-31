"""Keyboard teleop test for SO-101 differential IK in MuJoCo."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import mujoco
from mujoco import viewer as mujoco_viewer
import numpy as np

SIM_ROOT = Path(__file__).resolve().parent.parent
if str(SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SIM_ROOT))

from mujoco_tasks.envs.scene import HOME_QPOS
from mujoco_tasks.viz.scene_viz import configure_scene_camera
from mujoco_tasks.envs.utils.collision import configure_collision_debug_viewer
from mujoco_tasks.envs.gym_env import make_environment
from mujoco_tasks.motion import IKConfig, SO101IKSolver, apply_delta_rotation
from mujoco_tasks.motion.sticky_grasp import StickyGraspAssist

# GLFW key codes used by MuJoCo passive viewer.
KEY_UP = 265
KEY_DOWN = 264
KEY_LEFT = 263
KEY_RIGHT = 262
KEY_PAGE_UP = 266
KEY_PAGE_DOWN = 267
KEY_ESCAPE = 256
KEY_R = ord("R")
KEY_Z = ord("Z")
KEY_X = ord("X")

# Windows virtual-key codes. MuJoCo's passive-viewer callback does not expose
# key-release events, so it cannot reliably implement hold-to-move by itself.
_VK_UP = 0x26
_VK_DOWN = 0x28
_VK_LEFT = 0x25
_VK_RIGHT = 0x27
_VK_PAGE_UP = 0x21
_VK_PAGE_DOWN = 0x22
_HELD_TRANSLATION_KEYS = {
    ord("W"): np.array([1.0, 0.0, 0.0]),
    _VK_UP: np.array([1.0, 0.0, 0.0]),
    ord("S"): np.array([-1.0, 0.0, 0.0]),
    _VK_DOWN: np.array([-1.0, 0.0, 0.0]),
    ord("D"): np.array([0.0, 1.0, 0.0]),
    _VK_RIGHT: np.array([0.0, 1.0, 0.0]),
    ord("A"): np.array([0.0, -1.0, 0.0]),
    _VK_LEFT: np.array([0.0, -1.0, 0.0]),
    ord("E"): np.array([0.0, 0.0, 1.0]),
    _VK_PAGE_UP: np.array([0.0, 0.0, 1.0]),
    ord("Q"): np.array([0.0, 0.0, -1.0]),
    _VK_PAGE_DOWN: np.array([0.0, 0.0, -1.0]),
}
_HELD_GRIPPER_KEYS = {
    ord("Z"): 1.0,
    ord("X"): -1.0,
}
KEY_REPEAT_PERIOD = 0.10
GRIPPER_STEP_RAD = 0.02
IK_PLAN_STEPS = 32
POS_STEP = 0.005
ROT_STEP = 0.08
SLEW_ALPHA = 0.15


@dataclass
class TeleopState:
    target_pos: np.ndarray
    target_quat: np.ndarray
    gripper_qpos: float
    home_pos: np.ndarray
    home_quat: np.ndarray
    pos_step: float = POS_STEP
    rot_step: float = ROT_STEP
    gripper_step_rad: float = GRIPPER_STEP_RAD
    verbose: bool = False
    status: str = ""
    quit_requested: bool = False


class HeldKeyRepeater:
    """Turn Windows key-down state into a first press plus timed repeats."""

    def __init__(self, repeat_period: float) -> None:
        self.repeat_period = repeat_period
        self._pressed_at: dict[int, float] = {}

    def _should_apply(self, keycode: int, now: float, is_down: Callable[[int], bool]) -> bool:
        if not is_down(keycode):
            self._pressed_at.pop(keycode, None)
            return False
        last_applied = self._pressed_at.get(keycode)
        if last_applied is not None and now - last_applied < self.repeat_period:
            return False
        self._pressed_at[keycode] = now
        return True

    def poll_translation(self, now: float, is_down: Callable[[int], bool]) -> np.ndarray:
        """Return the target-position direction to apply on this viewer frame."""

        direction = np.zeros(3, dtype=np.float64)
        for keycode, key_direction in _HELD_TRANSLATION_KEYS.items():
            if self._should_apply(keycode, now, is_down):
                direction += key_direction
        return direction

    def poll_gripper(self, now: float, is_down: Callable[[int], bool]) -> float:
        """Return the signed gripper-radian increment multiplier for this frame."""

        return sum(
            direction
            for keycode, direction in _HELD_GRIPPER_KEYS.items()
            if self._should_apply(keycode, now, is_down)
        )


def _windows_key_down(keycode: int) -> bool:
    """Return whether a Windows virtual key is currently held down."""

    if sys.platform != "win32":
        return False
    import ctypes

    return bool(ctypes.windll.user32.GetAsyncKeyState(keycode) & 0x8000)


def _update_status(state: TeleopState) -> None:
    if not state.verbose:
        return
    state.status = (
        f"target=({state.target_pos[0]:.3f}, {state.target_pos[1]:.3f}, {state.target_pos[2]:.3f}) "
        f"gripper={state.gripper_qpos:.3f}"
    )


def _apply_translation(state: TeleopState, direction: np.ndarray) -> None:
    state.target_pos += np.asarray(direction, dtype=np.float64) * state.pos_step
    _update_status(state)


def _rotate_target(state: TeleopState, axis: np.ndarray, sign: float) -> None:
    state.target_quat = apply_delta_rotation(
        state.target_quat,
        axis,
        sign * state.rot_step,
        frame="tool",
    )


def _on_key(
    keycode: int,
    state: TeleopState,
    *,
    allow_translation: bool = True,
    allow_gripper: bool = True,
) -> None:
    if keycode == KEY_ESCAPE:
        state.quit_requested = True
        print("退出")
        return

    if keycode == KEY_R:
        state.target_pos = state.home_pos.copy()
        state.target_quat = state.home_quat.copy()
        state.gripper_qpos = float(HOME_QPOS[-1])
        print("已复位到 home 位姿")
        return

    moved = False
    if allow_translation and keycode in {KEY_UP, ord("W")}:
        _apply_translation(state, np.array([1.0, 0.0, 0.0]))
        return
    elif allow_translation and keycode in {KEY_DOWN, ord("S")}:
        _apply_translation(state, np.array([-1.0, 0.0, 0.0]))
        return
    elif allow_translation and keycode in {KEY_RIGHT, ord("D")}:
        _apply_translation(state, np.array([0.0, 1.0, 0.0]))
        return
    elif allow_translation and keycode in {KEY_LEFT, ord("A")}:
        _apply_translation(state, np.array([0.0, -1.0, 0.0]))
        return
    elif allow_translation and keycode in {KEY_PAGE_UP, ord("E")}:
        _apply_translation(state, np.array([0.0, 0.0, 1.0]))
        return
    elif allow_translation and keycode in {KEY_PAGE_DOWN, ord("Q")}:
        _apply_translation(state, np.array([0.0, 0.0, -1.0]))
        return
    elif keycode == ord("T"):
        _rotate_target(state, np.array([1.0, 0.0, 0.0]), +1.0)
        moved = True
    elif keycode == ord("G"):
        _rotate_target(state, np.array([1.0, 0.0, 0.0]), -1.0)
        moved = True
    elif keycode == ord("Y"):
        _rotate_target(state, np.array([0.0, 1.0, 0.0]), +1.0)
        moved = True
    elif keycode == ord("H"):
        _rotate_target(state, np.array([0.0, 1.0, 0.0]), -1.0)
        moved = True
    elif keycode == ord("U"):
        _rotate_target(state, np.array([0.0, 0.0, 1.0]), +1.0)
        moved = True
    elif keycode == ord("J"):
        _rotate_target(state, np.array([0.0, 0.0, 1.0]), -1.0)
        moved = True
    elif allow_gripper and keycode == KEY_Z:
        state.gripper_qpos += state.gripper_step_rad
        moved = True
    elif allow_gripper and keycode == KEY_X:
        state.gripper_qpos -= state.gripper_step_rad
        moved = True

    if moved:
        _update_status(state)


def _print_help(*, position_only: bool) -> None:
    mode = "仅位置 IK" if position_only else "位置 + 旋转 IK（5 自由度，姿态误差可能较大）"
    print(
        f"""
SO-101 IK 键盘测试（{mode}）
------------------------------------------------
平移（Windows 上按住按键每 0.10 秒持续累积目标；+X 朝桌面方向）:
  W / Up     +X
  S / Down   -X
  D / Right  +Y
  A / Left   -Y
  E / PgUp   +Z
  Q / PgDn   -Z
"""
    )
    if not position_only:
        print(
            """旋转（末端坐标系）:
  T / G      绕末端 X
  Y / H      绕末端 Y
  U / J      绕末端 Z
"""
        )
    print(
        """夹爪（每 0.10 秒步进 0.02 rad）:
  Z          张开
  X          闭合

其他:
  R          复位到 home
  Esc        退出
"""
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["cube", "bottle"], default="cube")
    parser.add_argument(
        "--position-only",
        action="store_true",
        help="Disable rotation IK and keep the home orientation fixed",
    )
    parser.add_argument(
        "--physics-substeps",
        type=int,
        default=20,
        help="MuJoCo steps per viewer frame",
    )
    parser.add_argument(
        "--show-collision",
        action="store_true",
        help="Show semi-transparent collision geoms for the bottle/cube and gripper jaws",
    )
    parser.add_argument(
        "--sticky-grasp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Attach an object after both jaws contact it; open the gripper to release (default: enabled)",
    )
    parser.add_argument(
        "--show-grasp",
        action="store_true",
        help="Show the demo grasp visualization in the scene",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print IK target, error, and iteration telemetry while teleoperating",
    )
    parser.add_argument(
        "--sticky-grasp-penetration",
        type=float,
        default=0.001,
        help="Required penetration of each jaw into the object before attaching (m)",
    )
    args = parser.parse_args()
    if args.sticky_grasp_penetration < 0.0:
        parser.error("--sticky-grasp-penetration must be non-negative")

    _print_help(position_only=args.position_only)

    env = make_environment(
        args.task,
        render_mode=None,
        show_collision_debug=args.show_collision,
        show_grasp=args.show_grasp,
    )
    try:
        solver = SO101IKSolver(
            env.model,
            env.data,
            config=IKConfig(
                max_steps=1,
                position_only=args.position_only,
                ik_gain=1.0,
                null_gain=0.05,
                max_joint_step=0.12,
                rot_weight=0.35,
                rot_tolerance=0.5,
            ),
        )
        home_pos, home_quat = solver.eef_pose()
        teleop = TeleopState(
            target_pos=home_pos.copy(),
            target_quat=home_quat.copy(),
            gripper_qpos=float(HOME_QPOS[-1]),
            home_pos=home_pos.copy(),
            home_quat=home_quat.copy(),
            gripper_step_rad=GRIPPER_STEP_RAD,
            verbose=args.verbose,
        )

        held_key_repeater = HeldKeyRepeater(KEY_REPEAT_PERIOD) if sys.platform == "win32" else None

        def key_callback(keycode: int) -> None:
            _on_key(
                keycode,
                teleop,
                allow_translation=held_key_repeater is None,
                allow_gripper=held_key_repeater is None,
            )

        sticky_grasp = (
            StickyGraspAssist(
                env.model,
                args.task,
                min_penetration=args.sticky_grasp_penetration,
            )
            if args.sticky_grasp
            else None
        )
        with mujoco_viewer.launch_passive(env.model, env.data, key_callback=key_callback) as mj_viewer:
            configure_scene_camera(mj_viewer.cam)
            configure_collision_debug_viewer(mj_viewer, enabled=args.show_collision)
            gripper_range = env.model.joint("so101_gripper").range
            gripper_ctrl_id = env.model.actuator("act_so101_gripper").id
            last_status_print = 0.0
            while mj_viewer.is_running() and not teleop.quit_requested:
                if held_key_repeater is not None:
                    now = time.monotonic()
                    direction = held_key_repeater.poll_translation(now, _windows_key_down)
                    if np.any(direction):
                        _apply_translation(teleop, direction)
                    gripper_direction = held_key_repeater.poll_gripper(now, _windows_key_down)
                    if gripper_direction:
                        teleop.gripper_qpos = float(
                            np.clip(
                                teleop.gripper_qpos + gripper_direction * teleop.gripper_step_rad,
                                gripper_range[0],
                                gripper_range[1],
                            )
                        )
                        _update_status(teleop)
                if sticky_grasp is not None and sticky_grasp.attached:
                    teleop.gripper_qpos = sticky_grasp.clamp_gripper_target(teleop.gripper_qpos)
                target_quat = teleop.home_quat if args.position_only else teleop.target_quat
                pos_error, rot_error = solver.pose_error(teleop.target_pos, target_quat)
                pos_err_norm = float(np.linalg.norm(pos_error))
                rot_err_norm = float(np.linalg.norm(rot_error))
                ik_iterations = min(
                    96,
                    IK_PLAN_STEPS + int(pos_err_norm * 200) + int(rot_err_norm * 80),
                )

                joint_target = solver.plan_joint_target(
                    teleop.target_pos,
                    target_quat,
                    iterations=ik_iterations,
                    gripper_qpos=teleop.gripper_qpos,
                )
                solver.apply_ctrl_target(
                    joint_target,
                    gripper_qpos=teleop.gripper_qpos,
                    slew_alpha=SLEW_ALPHA,
                )
                for _ in range(args.physics_substeps):
                    if sticky_grasp is not None:
                        event = sticky_grasp.follow_or_release(env.data, teleop.gripper_qpos)
                        if event is not None:
                            print(event.format())
                    mujoco.mj_step(env.model, env.data)
                    if sticky_grasp is not None:
                        event = sticky_grasp.observe_contacts(env.data, teleop.gripper_qpos)
                        if event is not None:
                            teleop.gripper_qpos = sticky_grasp.grasp_width_rad
                            env.data.ctrl[gripper_ctrl_id] = sticky_grasp.grasp_width_rad
                            print(event.format())

                now = time.time()
                if (
                    args.verbose
                    and teleop.status
                    and now - last_status_print > 0.2
                ):
                    eef_pos, eef_quat = solver.eef_pose()
                    _, live_rot_error = solver.pose_error(teleop.target_pos, target_quat)
                    live_rot_err_norm = float(np.linalg.norm(live_rot_error))
                    print(
                        f"{teleop.status} | pos_err={pos_err_norm:.4f} rot_err={live_rot_err_norm:.4f} "
                        f"eef=({eef_pos[0]:.3f}, {eef_pos[1]:.3f}, {eef_pos[2]:.3f}) "
                        f"ik_iter={ik_iterations}"
                    )
                    teleop.status = ""
                    last_status_print = now

                mj_viewer.sync()
                time.sleep(0.02)
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
