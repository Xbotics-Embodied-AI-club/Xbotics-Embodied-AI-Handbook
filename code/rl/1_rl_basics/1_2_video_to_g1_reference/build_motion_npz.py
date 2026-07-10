from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch
from mjlab.entity import Entity
from mjlab.scene import Scene
from mjlab.scripts.csv_to_npz import MotionLoader
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg


REQUIRED_KEYS = [
    "fps",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
]


def validate_motion_npz(path: str | Path) -> dict[str, float | int]:
    data = np.load(path)
    missing = [key for key in REQUIRED_KEYS if key not in data.files]
    if missing:
        raise ValueError(f"motion npz missing required keys: {missing}")

    joint_pos = data["joint_pos"]
    joint_vel = data["joint_vel"]
    body_pos_w = data["body_pos_w"]
    body_quat_w = data["body_quat_w"]
    body_lin_vel_w = data["body_lin_vel_w"]
    body_ang_vel_w = data["body_ang_vel_w"]

    if joint_pos.ndim != 2 or joint_pos.shape[1] != 29:
        raise ValueError(f"joint_pos must have shape (T, 29), got {joint_pos.shape}")
    if joint_vel.shape != joint_pos.shape:
        raise ValueError(f"joint_vel must have shape {joint_pos.shape}, got {joint_vel.shape}")
    if body_pos_w.ndim != 3 or body_pos_w.shape[-1] != 3:
        raise ValueError(f"body_pos_w must have shape (T, B, 3), got {body_pos_w.shape}")
    if body_quat_w.shape[:2] != body_pos_w.shape[:2] or body_quat_w.shape[-1] != 4:
        raise ValueError(f"body_quat_w must have shape (T, B, 4), got {body_quat_w.shape}")
    for key, value in [("body_lin_vel_w", body_lin_vel_w), ("body_ang_vel_w", body_ang_vel_w)]:
        if value.shape != body_pos_w.shape:
            raise ValueError(f"{key} must have shape {body_pos_w.shape}, got {value.shape}")

    return {
        "fps": float(np.asarray(data["fps"]).reshape(-1)[0]),
        "frames": int(joint_pos.shape[0]),
        "joints": int(joint_pos.shape[1]),
        "bodies": int(body_pos_w.shape[1]),
    }


def copy_motion_npz(source: str | Path, output: str | Path) -> dict[str, float | int]:
    summary = validate_motion_npz(source)
    data = np.load(source)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        fps=np.asarray(data["fps"], dtype=np.float32),
        joint_pos=np.asarray(data["joint_pos"], dtype=np.float32),
        joint_vel=np.asarray(data["joint_vel"], dtype=np.float32),
        body_pos_w=np.asarray(data["body_pos_w"], dtype=np.float32),
        body_quat_w=np.asarray(data["body_quat_w"], dtype=np.float32),
        body_lin_vel_w=np.asarray(data["body_lin_vel_w"], dtype=np.float32),
        body_ang_vel_w=np.asarray(data["body_ang_vel_w"], dtype=np.float32),
    )
    return summary


def _replay_g1_csv_with_mjlab(
    input_file: Path,
    output: Path,
    *,
    input_fps: int,
    output_fps: int,
    device: str,
    render: bool,
) -> None:
    if render:
        raise ValueError("render=True is not used in this course pipeline; write the motion npz only.")
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    sim_cfg = SimulationCfg()
    sim_cfg.mujoco.timestep = 1.0 / output_fps
    scene = Scene(unitree_g1_flat_tracking_env_cfg().scene, device=device)
    model = scene.compile()
    sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
    scene.initialize(sim.mj_model, sim.model, sim.data)

    motion = MotionLoader(
        motion_file=str(input_file),
        input_fps=input_fps,
        output_fps=output_fps,
        device=sim.device,
    )
    robot: Entity = scene["robot"]
    robot_joint_indexes = robot.find_joints(
        [
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ],
        preserve_order=True,
    )[0]

    log: dict[str, list[np.ndarray] | np.ndarray] = {
        "fps": np.array([output_fps], dtype=np.float32),
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }

    scene.reset()
    done = False
    while not done:
        (
            (
                motion_base_pos,
                motion_base_rot,
                motion_base_lin_vel,
                motion_base_ang_vel,
                motion_dof_pos,
                motion_dof_vel,
            ),
            done,
        ) = motion.get_next_state()

        root_states = robot.data.default_root_state.clone()
        root_states[:, 0:3] = motion_base_pos
        root_states[:, :2] += scene.env_origins[:, :2]
        root_states[:, 3:7] = motion_base_rot
        root_states[:, 7:10] = motion_base_lin_vel
        root_states[:, 10:] = motion_base_ang_vel
        robot.write_root_state_to_sim(root_states)

        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, robot_joint_indexes] = motion_dof_pos
        joint_vel[:, robot_joint_indexes] = motion_dof_vel
        robot.write_joint_state_to_sim(joint_pos, joint_vel)

        sim.forward()
        scene.update(sim.mj_model.opt.timestep)

        log["joint_pos"].append(robot.data.joint_pos[0, :].cpu().numpy().copy())
        log["joint_vel"].append(robot.data.joint_vel[0, :].cpu().numpy().copy())
        log["body_pos_w"].append(robot.data.body_link_pos_w[0, :].cpu().numpy().copy())
        log["body_quat_w"].append(robot.data.body_link_quat_w[0, :].cpu().numpy().copy())
        log["body_lin_vel_w"].append(robot.data.body_link_lin_vel_w[0, :].cpu().numpy().copy())
        log["body_ang_vel_w"].append(robot.data.body_link_ang_vel_w[0, :].cpu().numpy().copy())

    output.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"fps": log["fps"]}
    for key in REQUIRED_KEYS[1:]:
        arrays[key] = np.stack(log[key], axis=0).astype(np.float32)
    np.savez(output, **arrays)


def build_motion_npz_from_csv(
    source: str | Path,
    output: str | Path,
    *,
    input_fps: int = 30,
    output_fps: int = 50,
    device: str = "cuda:0",
    render: bool = False,
    builder: Callable[..., None] | None = None,
) -> dict[str, float | int]:
    source = Path(source)
    if not source.is_file():
        raise FileNotFoundError(f"GMR CSV not found: {source}")
    output = Path(output)
    if builder is None:
        builder = _replay_g1_csv_with_mjlab
    builder(source, output, input_fps=input_fps, output_fps=output_fps, device=device, render=render)
    return validate_motion_npz(output)


def main() -> None:
    group_root = Path(__file__).resolve().parents[1]

    # 主要修改这一段：输入源、输出 npz、帧率与设备。
    # input_format="npz" 直接校验并写出规范课程数据；="csv" 用 mjlab 回放 GMR 导出的 G1 CSV 生成 npz。
    source = group_root / "data/g1_reference_motions/marshal-arts.npz"
    output = group_root / "data/g1_reference_motions/marshal-arts.npz"
    input_format = "npz"
    input_fps = 30
    output_fps = 50
    device = "cuda:0"
    render = False

    if input_format == "csv":
        summary = build_motion_npz_from_csv(
            source,
            output,
            input_fps=input_fps,
            output_fps=output_fps,
            device=device,
            render=render,
        )
    else:
        summary = copy_motion_npz(source, output)
    print(
        f"wrote {output} "
        f"frames={summary['frames']} joints={summary['joints']} bodies={summary['bodies']} fps={summary['fps']}"
    )


if __name__ == "__main__":
    main()
