"""MuJoCo visualization of the five key action poses (pre_grasp ... retreat).

Each pose is drawn as a coloured sphere at the EEF origin plus an RGB XYZ axis
frame showing the wrist orientation.  The markers are static visual-only geoms,
so they can be attached to the scene before compilation and remain visible in
the passive viewer while the state machine runs.
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

from mujoco_tasks.pose_targets import make_five_targets, make_task_config
from mujoco_tasks.motion.grasp_to_gripper import eef_pose_to_grasp
from mujoco_tasks.viz.grasp_viz import _gripper_part_boxes

POSE_COLORS = {
    "pre_grasp": (0.20, 0.55, 1.00, 0.95),
    "grasp": (0.10, 0.85, 0.25, 0.95),
    "lift": (1.00, 0.70, 0.10, 0.95),
    "place": (1.00, 0.25, 0.25, 0.95),
    "retreat": (0.75, 0.35, 0.95, 0.95),
}

POSE_AXIS_LENGTH = 0.05
GRASP_DEPTH = 0.02
GRASP_JAW_HEIGHT = 0.004
GRASP_AXIS_LENGTH = 0.04

ACTIVE_ALPHA = 1.0
INACTIVE_ALPHA = 0.10


def state_to_pose_name(state) -> str | None:
    """Map a state-machine state to the five-pose marker it executes."""

    from robot_pick_place.models import State

    mapping = {
        State.INIT: "pre_grasp",
        State.MOVE_PRE_GRASP: "pre_grasp",
        State.APPROACH: "grasp",
        State.CLOSE_GRIPPER: "grasp",
        State.VERIFY_GRASP: "grasp",
        State.LIFT: "lift",
        State.VERIFY_LIFT: "lift",
        State.MOVE_PLACE: "place",
        State.OPEN_GRIPPER: "place",
        State.VERIFY_PLACE: "place",
        State.RETREAT: "retreat",
        State.DONE: "retreat",
    }
    return mapping.get(state)


def set_pose_marker_opacity(
    model: mujoco.MjModel,
    active_pose: str | None,
) -> None:
    """Highlight the active GraspNet pose marker; fade the rest out."""

    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if not name or not name.startswith("graspnet_"):
            continue
        pose_name = name[len("graspnet_") :].split("_", 1)[0]
        alpha = ACTIVE_ALPHA if pose_name == active_pose else INACTIVE_ALPHA
        rgba = model.geom_rgba[geom_id]
        model.geom_rgba[geom_id] = (rgba[0], rgba[1], rgba[2], alpha)


def _add_graspnet_gripper(
    body: mujoco.MjsBody,
    *,
    prefix: str,
    width: float,
    rgba: tuple[float, float, float, float],
) -> None:
    """Draw a GraspNet-style parallel-jaw gripper in the body's local grasp frame."""

    parts = ("left", "right", "bottom", "tail")
    for part_name, (local_center, half_size) in zip(
        parts,
        _gripper_part_boxes(width, GRASP_DEPTH, GRASP_JAW_HEIGHT),
    ):
        geom = body.add_geom(
            name=f"{prefix}_{part_name}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=local_center.tolist(),
            size=half_size.tolist(),
            rgba=list(rgba),
        )
        geom.contype = 0
        geom.conaffinity = 0


def _add_grasp_axes(
    body: mujoco.MjsBody,
    *,
    prefix: str,
    length: float = GRASP_AXIS_LENGTH,
) -> None:
    """Draw the grasp-frame XYZ axes (X/Y/Z = red/green/blue)."""

    axes = (
        ("x", [length, 0.0, 0.0], [1.0, 0.15, 0.15, 0.9]),
        ("y", [0.0, length, 0.0], [0.15, 1.0, 0.15, 0.9]),
        ("z", [0.0, 0.0, length], [0.15, 0.35, 1.0, 0.9]),
    )
    for axis_name, endpoint, rgba in axes:
        geom = body.add_geom(
            name=f"{prefix}_axis_{axis_name}",
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            fromto=[0.0, 0.0, 0.0, *endpoint],
            size=[0.0015, 0.0, 0.0],
            rgba=list(rgba),
        )
        geom.contype = 0
        geom.conaffinity = 0


def _eef_quat_wxyz(pose: dict[str, float]) -> list[float]:
    return [pose["qw"], pose["qx"], pose["qy"], pose["qz"]]


def add_pose_markers(
    scene_spec: mujoco.MjSpec,
    task: str,
    *,
    open_width: float | None = None,
) -> dict[str, object]:
    """Attach the five action-pose markers to a scene spec before compilation."""

    task_config = make_task_config(task, open_width=open_width)
    targets = make_five_targets(task_config)
    gripper_width = task_config.gripper_open_width

    for pose_name, pose in asdict(targets).items():
        rgba = POSE_COLORS[pose_name]
        # GraspNet-style parallel-jaw gripper at the grasp frame derived from
        # this EEF pose (inverse of the repo's grasp_to_eef_pose mapping).
        grasp = eef_pose_to_grasp(
            np.array([pose["x"], pose["y"], pose["z"]], dtype=np.float64),
            np.asarray(_eef_quat_wxyz(pose), dtype=np.float64),
            width=gripper_width,
            depth=GRASP_DEPTH,
            height=GRASP_JAW_HEIGHT,
        )
        quat_grasp = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quat_grasp, grasp.rotation_matrix.reshape(-1))
        grasp_body = scene_spec.worldbody.add_body(
            name=f"graspnet_{pose_name}",
            pos=grasp.translation.tolist(),
            quat=quat_grasp.tolist(),
        )
        _add_graspnet_gripper(grasp_body, prefix=f"graspnet_{pose_name}", width=gripper_width, rgba=rgba)
        _add_grasp_axes(grasp_body, prefix=f"graspnet_{pose_name}")
    return asdict(targets)


def build_pose_viz_model(task: str, *, show_poses: bool = True) -> mujoco.MjModel:
    """Build the task scene with (optionally) the five pose markers attached."""

    from mujoco_tasks.envs.scene import build_model

    return build_model(task, show_poses=show_poses)


def pose_legend() -> str:
    lines = [
        "五个动作位姿（GraspNet 风格平行夹爪 + grasp 坐标系，颜色区分）:",
        "  夹爪盒子为 GraspNet 6D 抓取表示（X=夹持宽度方向，Z=接近轴）",
        "  坐标系: X=红, Y=绿, Z=蓝",
    ]
    for name, rgba in POSE_COLORS.items():
        rgb = " ".join(f"{channel:.2f}" for channel in rgba[:3])
        lines.append(f"  {name:<10s} rgb=({rgb})")
    return "\n".join(lines)


def show_poses_in_viewer(model: mujoco.MjModel, task: str) -> int:
    """Open the passive viewer and hold the robot at home next to the markers."""

    from mujoco import viewer as mujoco_viewer

    from mujoco_tasks.envs.scene import HOME_QPOS, JOINT_NAMES
    from mujoco_tasks.viz.scene_viz import configure_scene_camera

    data = mujoco.MjData(model)
    joint_ids = [model.joint(name).qposadr[0] for name in JOINT_NAMES]
    data.qpos[joint_ids] = HOME_QPOS
    data.ctrl[:] = HOME_QPOS
    mujoco.mj_forward(model, data)

    stop = threading.Event()

    def wait_for_enter() -> None:
        input("窗口已打开，按 Enter 关闭仿真... ")
        stop.set()

    threading.Thread(target=wait_for_enter, daemon=True).start()
    print(pose_legend())

    with mujoco_viewer.launch_passive(model, data) as viewer:
        configure_scene_camera(viewer.cam)
        while not stop.is_set() and viewer.is_running():
            data.qpos[joint_ids] = HOME_QPOS
            data.ctrl[:] = HOME_QPOS
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(0.02)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["cube", "bottle"], default="cube")
    args = parser.parse_args()

    model = build_pose_viz_model(args.task, show_poses=True)
    return show_poses_in_viewer(model, args.task)


if __name__ == "__main__":
    raise SystemExit(main())
