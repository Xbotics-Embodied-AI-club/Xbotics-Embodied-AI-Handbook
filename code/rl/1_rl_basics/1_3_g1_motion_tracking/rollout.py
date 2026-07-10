from __future__ import annotations

import json
import os
from pathlib import Path

import mediapy as media
import numpy as np
import torch

from .env import BeyondMimicEnv
from .motion import MotionClip
from .train_v3_ppo import ActorCritic


def result_root() -> Path:
    return Path(__file__).resolve().parents[1] / "result" / "1_3_g1_motion_tracking"


def default_rollout_output(checkpoint: str | Path, motion_file: str | Path) -> Path:
    checkpoint = Path(checkpoint)
    motion_file = Path(motion_file)
    return result_root() / f"{motion_file.stem}-{checkpoint.stem}.json"


def default_rollout_video_output(checkpoint: str | Path, motion_file: str | Path) -> Path:
    checkpoint = Path(checkpoint)
    motion_file = Path(motion_file)
    return result_root() / f"{motion_file.stem}-{checkpoint.stem}.mp4"


def load_policy(checkpoint: str | Path, device: str | torch.device) -> tuple[ActorCritic, int]:
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = ActorCritic(obs_dim=160, critic_obs_dim=286, action_dim=29)
    model.load_state_dict(checkpoint_data["actor_critic"])
    model.eval()
    model.to(device)
    return model, int(checkpoint_data.get("iteration", -1))


def _recorded_frame(frame) -> np.ndarray:
    frame = frame[0] if isinstance(frame, np.ndarray) and frame.ndim == 4 else frame
    frame = np.asarray(frame)
    if frame.dtype != np.uint8:
        frame = (np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8)
    return frame


def run_rollout(
    motion_file: Path,
    checkpoint: Path,
    output: Path | None = None,
    video_output: Path | None = None,
    num_envs: int = 16,
    num_steps: int = 200,
    device: str = "cuda:0",
    seed: int = 1,
    episode_length_s: float = 10.0,
    show_reference_ghost: bool = False,
) -> dict[str, int | float | str]:
    torch.manual_seed(seed)
    motion = MotionClip.load(motion_file, device="cpu")
    output = output or default_rollout_output(checkpoint, motion_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    if video_output is not None:
        video_output.parent.mkdir(parents=True, exist_ok=True)

    if video_output is None:
        env = BeyondMimicEnv(
            motion_file=motion_file,
            num_envs=num_envs,
            device=device,
            episode_length_s=episode_length_s,
            seed=seed,
            show_reference_ghost=show_reference_ghost,
        )
    else:
        env = BeyondMimicEnv(
            motion_file=motion_file,
            num_envs=num_envs,
            device=device,
            episode_length_s=episode_length_s,
            seed=seed,
            show_reference_ghost=show_reference_ghost,
            render_mode="rgb_array",
        )

    rewards = []
    action_abs_means = []
    action_abs_maxes = []
    frames = []
    done_count = 0
    try:
        model, checkpoint_iteration = load_policy(checkpoint, env.device)
        obs, _critic_obs = env.reset()
        for _ in range(num_steps):
            with torch.no_grad():
                actions = model.act_inference(obs)
            obs, _critic_obs, reward, done, _info = env.step(actions)
            rewards.append(float(reward.mean().detach().cpu()))
            action_abs_means.append(float(actions.detach().abs().mean().cpu()))
            action_abs_maxes.append(float(actions.detach().abs().max().cpu()))
            done_count += int(done.detach().sum().cpu())
            if video_output is not None:
                frame = env.render()
                if frame is not None:
                    frames.append(_recorded_frame(frame))
    finally:
        env.close()

    summary = {
        "motion_file": str(Path(motion_file)),
        "checkpoint": str(Path(checkpoint)),
        "checkpoint_iteration": checkpoint_iteration,
        "output": str(output),
        "video_output": str(video_output) if video_output is not None else "",
        "video_frames": int(len(frames)),
        "num_envs": int(num_envs),
        "steps": int(num_steps),
        "motion_frames": int(motion.num_frames),
        "motion_fps": float(motion.fps),
        "mean_reward": float(sum(rewards) / len(rewards)) if rewards else 0.0,
        "action_abs_mean": float(sum(action_abs_means) / len(action_abs_means)) if action_abs_means else 0.0,
        "action_abs_max": float(max(action_abs_maxes)) if action_abs_maxes else 0.0,
        "done_count": int(done_count),
        "done_fraction": float(done_count / (num_envs * num_steps)) if num_envs and num_steps else 0.0,
    }
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    # 视频是展示产物，放在最后写；summary 先落盘，方便训练/rollout 结果先可读。
    if video_output is not None and frames:
        fps = float(getattr(env, "metadata", {}).get("render_fps", motion.fps))
        media.write_video(str(video_output), frames, fps=fps)

    return summary


def main() -> None:
    group_root = Path(__file__).resolve().parents[1]
    datasets_root = Path(os.environ["DATASETS_ROOT"])

    # 主要修改这一段：motion、checkpoint 和录制长度。
    motion_file = group_root / "data/g1_reference_motions/marshal-arts.npz"
    checkpoint = (
        datasets_root
        / "models/trained/xbotics_rl_beyondmimic/beyondmimic-marshal-arts-lightning-10000/model_10000.pt"
    )
    output = group_root / "result/1_3_g1_motion_tracking/marshal-arts-model_10000.json"
    video_output = group_root / "result/1_3_g1_motion_tracking/marshal-arts-model_10000.mp4"
    num_envs = 16
    num_steps = 400
    device = "cuda:0"
    seed = 1
    episode_length_s = 10.0
    show_reference_ghost = False

    summary = run_rollout(
        motion_file=motion_file,
        checkpoint=checkpoint,
        output=output,
        video_output=video_output,
        num_envs=num_envs,
        num_steps=num_steps,
        device=device,
        seed=seed,
        episode_length_s=episode_length_s,
        show_reference_ghost=show_reference_ghost,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
