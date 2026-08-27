"""Run the Lecture 07 pick-place state machine inside MuJoCo.

This entry wires the document's state machine
(``robot_pick_place.state_machine.PickPlaceStateMachine``) to the real SO-101
MuJoCo scene through :class:`mujoco_tasks.fsm_backend.MuJoCoFSMBackend`.
The five action poses are generated from the scene layout and can be drawn in
the viewer with ``--show-poses``.

Examples::

    python simulation/mujoco_pick_place.py --task cube
    python simulation/mujoco_pick_place.py --task bottle --show-poses
    python simulation/mujoco_pick_place.py --task cube --viewer null
    python simulation/mujoco_pick_place.py --task cube --inspect-poses
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mujoco_tasks.envs.scene import HOME_QPOS, JOINT_NAMES, build_model
from mujoco_tasks.envs.utils.collision import configure_collision_debug_viewer
from mujoco_tasks.fsm_backend import MuJoCoFSMBackend
from mujoco_tasks.pose_targets import make_five_targets, make_task_config
from mujoco_tasks.viz.pose_viz import (
    POSE_COLORS,
    pose_legend,
    set_pose_marker_opacity,
    show_poses_in_viewer,
    state_to_pose_name,
)
from mujoco_tasks.viz.scene_viz import configure_scene_camera, make_scene_camera

from robot_pick_place.state_machine import PickPlaceStateMachine
from robot_pick_place.task_logger import TaskLogger


def patch_state_machine_pose_generation() -> None:
    """Point the state machine's pose generator at the scene-aware version.

    ``PickPlaceStateMachine.run_attempt`` calls the module-level
    ``generate_targets`` imported from ``robot_pick_place.pose_generator``.
    For MuJoCo the generic generator would reuse the grasp orientation for the
    placement pose, which is unreachable for the bottle's mirrored box side.
    ``pose_targets.make_five_targets`` keeps the same five-pose contract but
    uses the scene-calibrated (and, for the bottle, mirrored) orientations.
    """

    import robot_pick_place.state_machine as state_machine_module
    from mujoco_tasks.pose_targets import make_five_targets

    state_machine_module.generate_targets = make_five_targets


class ConsoleStateMachine(PickPlaceStateMachine):
    """State machine that prints every state transition to the console."""

    def __init__(self, backend, task, logger, *, on_state=None) -> None:
        super().__init__(backend, task, logger)
        self._on_state = on_state

    def _enter(self, state, **extra) -> None:
        if self._on_state is not None:
            self._on_state(state)
        detail = " ".join(f"{key}={value}" for key, value in extra.items())
        print(f"[state] -> {state.name}{(' ' + detail) if detail else ''}")
        super()._enter(state, **extra)

    def _fail(self, code, reason):
        print(f"[failure] {code.name}: {reason}")
        return super()._fail(code, reason)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["cube", "bottle"], default="cube")
    parser.add_argument(
        "--viewer",
        choices=["human", "null"],
        default="human",
        help="human opens the MuJoCo passive viewer; null runs headless",
    )
    parser.add_argument(
        "--show-poses",
        action="store_true",
        help="Draw the five action poses (pre_grasp/grasp/lift/place/retreat) in the scene",
    )
    parser.add_argument(
        "--inspect-poses",
        action="store_true",
        help="Only show the five action poses with the robot held at home (no motion)",
    )
    parser.add_argument(
        "--show-collision",
        action="store_true",
        help="Show semi-transparent collision geoms and contact overlays",
    )
    parser.add_argument(
        "--show-grasp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show the GraspNet visualization on the gripper end effector (default: on)",
    )
    parser.add_argument(
        "--grasp-roll",
        type=float,
        default=None,
        help="Extra gripper roll about its pointing axis for the cube grasp (deg; default 90)",
    )
    parser.add_argument(
        "--sticky-grasp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Attach an object after both jaws contact it; open to release (default: enabled)",
    )
    parser.add_argument(
        "--output",
        default="runs",
        help="Directory for the JSONL task logs",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print IK / attach telemetry while the state machine runs",
    )
    parser.add_argument(
        "--move-timeout",
        type=float,
        default=30.0,
        help="Single move / gripper operation timeout in seconds (default: 30). "
        "Raise this if bottle runs abort with 'motion timeout' on a slow machine "
        "or with live viewer rendering.",
    )
    parser.add_argument(
        "--video",
        default=None,
        help="Record the run to an MP4 file (e.g. runs/bottle_horizontal.mp4)",
    )
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument(
        "--video-stride",
        type=int,
        default=5,
        help="Record one frame every N state updates (larger = shorter video)",
    )
    parser.add_argument("--video-width", type=int, default=1280)
    parser.add_argument("--video-height", type=int, default=720)
    return parser


class FrameRecorder:
    """Offscreen MuJoCo renderer that appends one frame per ``sync()`` call."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        path: str,
        *,
        fps: int,
        stride: int,
        width: int,
        height: int,
    ) -> None:
        import imageio

        self._data = data
        self._renderer = mujoco.Renderer(model, height=height, width=width)
        self._camera = make_scene_camera()
        self._writer = imageio.get_writer(
            path,
            fps=fps,
            codec="libx264",
            pixelformat="yuv420p",
            macro_block_size=None,
        )
        self._stride = max(1, stride)
        self._since_last = self._stride  # render the very first sync
        self.frames = 0
        self.calls = 0

    def sync(self) -> None:
        self.calls += 1
        self._since_last += 1
        if self._since_last < self._stride:
            return
        self._since_last = 0
        self._renderer.update_scene(self._data, camera=self._camera)
        frame = self._renderer.render()
        self._writer.append_data(frame)
        self.frames += 1

    def close(self) -> None:
        self._writer.close()
        self._renderer.close()


class SyncHub:
    """Call ``sync()`` on every attached viewer/recorder."""

    def __init__(self, *targets) -> None:
        self.targets = [target for target in targets if target is not None]

    def sync(self) -> None:
        for target in self.targets:
            target.sync()


def _print_targets(task: str, model, data, *, grasp_roll: float | None = None) -> None:
    task_config = make_task_config(task, model, data, grasp_roll_deg=grasp_roll)
    targets = make_five_targets(task_config)
    print(f"\n任务 {task_config.name} 的五个动作位姿 (EEF, 基座坐标系):")
    for name, pose in asdict(targets).items():
        rgba = POSE_COLORS[name]
        print(
            f"  {name:<10s} pos=({pose['x']:.3f}, {pose['y']:.3f}, {pose['z']:.3f}) "
            f"quat=({pose['qw']:.3f}, {pose['qx']:.3f}, {pose['qy']:.3f}, {pose['qz']:.3f}) "
            f"color=({rgba[0]:.2f}, {rgba[1]:.2f}, {rgba[2]:.2f})"
        )
    print(pose_legend())


def main() -> int:
    args = build_parser().parse_args()

    model = build_model(
        args.task,
        show_poses=args.show_poses or args.inspect_poses,
        show_grasp=args.show_grasp,
    )
    data = mujoco.MjData(model)
    for idx, name in enumerate(JOINT_NAMES):
        data.qpos[model.joint(name).qposadr[0]] = HOME_QPOS[idx]
    mujoco.mj_forward(model, data)

    _print_targets(args.task, model, data, grasp_roll=args.grasp_roll)

    if args.inspect_poses:
        return show_poses_in_viewer(model, args.task)

    viewer = None
    if args.viewer == "human":
        from mujoco import viewer as mujoco_viewer

        viewer = mujoco_viewer.launch_passive(model, data)
        configure_scene_camera(viewer.cam)
        configure_collision_debug_viewer(viewer, enabled=args.show_collision)

    recorder = None
    if args.video:
        recorder = FrameRecorder(
            model,
            data,
            args.video,
            fps=args.video_fps,
            stride=args.video_stride,
            width=args.video_width,
            height=args.video_height,
        )
        recorder.sync()

    sync_target = SyncHub(viewer, recorder)
    task_config = make_task_config(
        args.task,
        model,
        data,
        grasp_roll_deg=args.grasp_roll,
    )
    patch_state_machine_pose_generation()
    logger = TaskLogger(args.output, task_config.name)
    backend = MuJoCoFSMBackend(
        model,
        data,
        args.task,
        viewer=sync_target,
        verbose=args.verbose,
        move_timeout=args.move_timeout,
    )

    backend.connect()
    try:
        on_state = None
        if args.show_poses:
            on_state = lambda state: set_pose_marker_opacity(
                model, state_to_pose_name(state)
            )
            set_pose_marker_opacity(model, "pre_grasp")
        machine = ConsoleStateMachine(backend, task_config, logger, on_state=on_state)
        success = machine.run()
    finally:
        print("[state] -> HOME")
        backend.return_home()
        backend.disconnect()

    print(f"\ntask={task_config.name}, success={success}")
    print(f"log_dir={logger.task_dir}")
    if recorder is not None:
        print(f"video={args.video} frames={recorder.frames} sync_calls={recorder.calls}")

    if viewer is not None:
        # Hold the final pose briefly so the result is visible, then close.
        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline and viewer.is_running():
            sync_target.sync()
            time.sleep(0.02)
        viewer.close()
    elif recorder is not None:
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            sync_target.sync()
            time.sleep(0.02)

    if recorder is not None:
        recorder.close()

    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
