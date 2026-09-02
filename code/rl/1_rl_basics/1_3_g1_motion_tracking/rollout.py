"""按统一口径评测三个版本训练出的策略，并录成对照视频。

与行走线的同名脚本逐行对应，只是任务换成了动作跟踪：同一个环境、同样 16 个并行
环境跑 400 步、一律取确定性动作。讲义第 8 节表格里的每一个"评测平均奖励"和"摔倒率"
都由本文件产出，摘要写成 `result/1_3_g1_motion_tracking/<run>.json`，视频写成同名 `.mp4`。

讲义对应：第14讲 1.4 节（评测口径）、第 8 节（动作跟随的三版对照）。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import mediapy as media
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env import BeyondMimicEnv  # noqa: E402
from model import ActorCritic  # noqa: E402
from motion import MotionClip  # noqa: E402


def result_root() -> Path:
    """给出本模块结果目录的路径。

    Returns:
        `result/1_3_g1_motion_tracking/` 的绝对路径。
    """
    return Path(__file__).resolve().parents[1] / "result" / "1_3_g1_motion_tracking"


def load_policy(checkpoint, device):
    """从 checkpoint 里恢复一个可推理的策略。

    网络的三个维度不写死在这里，而是从 checkpoint 存的 `training_settings` 里读——
    环境一改维度就跟着变，硬编码迟早对不上。

    Args:
        checkpoint: checkpoint 文件路径。
        device: 模型放到哪个设备上。

    Returns:
        (模型, 该 checkpoint 对应的训练迭代数)。
    """
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


def run_rollout(motion_file, checkpoint, run_name, num_envs=16, num_steps=400, device="cuda:0", seed=1,
                show_reference_ghost=False):
    """跑一段确定性 rollout，落盘评测摘要与视频。

    动作取 `act_inference`（分布均值）而不是采样，评测成绩才不掺探索噪声。

    Args:
        motion_file: 要跟随的参考动作。
        checkpoint: 要评测的权重文件。
        run_name: 结果文件名前缀，同时写进摘要里便于溯源。
        num_envs: 并行环境数，三个版本必须一致。
        num_steps: 每个环境跑多少步。
        device: 仿真与推理所在设备。
        seed: 随机种子，固定它三个版本才面对同一批初始状态。
        show_reference_ghost: 是否把参考姿态的虚影一起画进视频。默认关掉，画面里
            只留机器人本身；要看「跟得准不准」就把它打开。

    Returns:
        评测摘要字典，字段与落盘的 `.json` 一致。
    """
    torch.manual_seed(seed)
    out_dir = result_root()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_output = out_dir / f"{run_name}.json"
    video_output = out_dir / f"{run_name}.mp4"
    motion = MotionClip.load(motion_file, device="cpu")

    env = BeyondMimicEnv(motion_file, num_envs=num_envs, device=device, seed=seed,
                         show_reference_ghost=show_reference_ghost, render_mode="rgb_array")
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
        "motion_file": str(Path(motion_file)),
        "motion_frames": int(motion.num_frames),
        "motion_fps": float(motion.fps),
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
        fps = float(getattr(env, "metadata", {}).get("render_fps", motion.fps))
        media.write_video(str(video_output), frames, fps=fps)
    return summary


def main():
    """依次评测三个版本各自的最终 checkpoint。"""
    group_root = Path(__file__).resolve().parents[1]
    datasets_root = Path(os.environ["DATASETS_ROOT"])
    trained = datasets_root / "models" / "trained" / "xbotics_rl_beyondmimic"
    motion_file = group_root / "data/g1_reference_motions/marshal-arts.npz"

    # 对三个版本各跑一遍对照（用各自最终 checkpoint）。
    runs = {
        "track-v1-reinforce": "beyondmimic-reinforce/model_3000.pt",
        "track-v2-a2c": "beyondmimic-a2c/model_3000.pt",
        "track-v3-ppo": "beyondmimic-ppo/model_3000.pt",
    }
    for run_name, rel in runs.items():
        checkpoint = trained / rel
        if not checkpoint.exists():
            print(f"skip {run_name}: {checkpoint} not found")
            continue
        summary = run_rollout(motion_file=motion_file, checkpoint=checkpoint, run_name=run_name, device="cuda:0")
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
