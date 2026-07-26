"""载入某个版本训练出的策略，在仿真里跑一段并录成视频，三版同口径便于对照。"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import mediapy as media
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env import G1WalkEnv  # noqa: E402
from model import ActorCritic  # noqa: E402


def result_root() -> Path:
    return Path(__file__).resolve().parents[1] / "result" / "1_1_g1_walk_rl"


def load_policy(checkpoint, device):
    data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    settings = data.get("training_settings", {})
    model = ActorCritic(
        obs_dim=settings["obs_dim"],
        critic_obs_dim=settings["critic_obs_dim"],
        action_dim=settings["action_dim"],
    )
    model.load_state_dict(data["actor_critic"])
    model.eval()
    model.to(device)
    return model, int(data.get("iteration", -1))


def _recorded_frame(frame) -> np.ndarray:
    frame = frame[0] if isinstance(frame, np.ndarray) and frame.ndim == 4 else frame
    frame = np.asarray(frame)
    if frame.dtype != np.uint8:
        frame = (np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8)
    return frame


def run_rollout(checkpoint, run_name, num_envs=16, num_steps=400, device="cuda:0", seed=1):
    torch.manual_seed(seed)
    out_dir = result_root()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_output = out_dir / f"{run_name}.json"
    video_output = out_dir / f"{run_name}.mp4"

    env = G1WalkEnv(num_envs=num_envs, device=device, seed=seed, render_mode="rgb_array")
    rewards, action_abs_means, frames = [], [], []
    done_count = 0
    try:
        model, iteration = load_policy(checkpoint, env.device)
        obs, _critic_obs = env.reset()
        for _ in range(num_steps):
            with torch.no_grad():
                actions = model.act_inference(obs)
            obs, _critic_obs, reward, done, _info = env.step(actions)
            rewards.append(float(reward.mean().detach().cpu()))
            action_abs_means.append(float(actions.detach().abs().mean().cpu()))
            done_count += int(done.detach().sum().cpu())
            frame = env.render()
            if frame is not None:
                frames.append(_recorded_frame(frame))
    finally:
        env.close()

    summary = {
        "run_name": run_name,
        "checkpoint": str(Path(checkpoint)),
        "checkpoint_iteration": iteration,
        "video_output": str(video_output),
        "video_frames": int(len(frames)),
        "num_envs": int(num_envs),
        "steps": int(num_steps),
        "mean_reward": float(sum(rewards) / len(rewards)) if rewards else 0.0,
        "action_abs_mean": float(sum(action_abs_means) / len(action_abs_means)) if action_abs_means else 0.0,
        "done_count": int(done_count),
        "done_fraction": float(done_count / (num_envs * num_steps)) if num_envs and num_steps else 0.0,
    }
    json_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    if frames:
        fps = float(getattr(env, "metadata", {}).get("render_fps", 50.0))
        media.write_video(str(video_output), frames, fps=fps)
    return summary


def main():
    datasets_root = Path(os.environ["DATASETS_ROOT"])
    trained = datasets_root / "models" / "trained" / "xbotics_rl_g1_walk"

    # 对三个版本各跑一遍对照（用各自最终 checkpoint）。
    runs = {
        "g1-walk-reinforce": "g1-walk-reinforce/model_3000.pt",
        "g1-walk-a2c": "g1-walk-a2c/model_3000.pt",
        "g1-walk-ppo": "g1-walk-ppo/model_3000.pt",
    }
    for run_name, rel in runs.items():
        checkpoint = trained / rel
        if not checkpoint.exists():
            print(f"skip {run_name}: {checkpoint} not found")
            continue
        summary = run_rollout(checkpoint=checkpoint, run_name=run_name, device="cuda:0")
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
