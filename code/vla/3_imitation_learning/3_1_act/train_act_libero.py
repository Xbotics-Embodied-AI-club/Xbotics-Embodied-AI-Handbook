"""ACT 在 LIBERO 短程单任务上的训练示例。"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from lerobot.configs.default import DatasetConfig, EvalConfig, WandBConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.scripts.lerobot_train import train

os.environ.setdefault("MUJOCO_GL", "egl")

# ─────────────────────── 数据集与任务 ───────────────────────
# 用官方总数据集，但只筛一个短程 task 来训练
DATASET_ID     = "lerobot/libero"
TARGET_TASK    = "put the bowl on the plate"
TARGET_SUITE   = "libero_goal"
TARGET_TASK_ID = 8
RUN_NAME       = datetime.now().strftime("act_libero_goal_plate_%Y%m%d_%H%M%S")
OUTPUT_DIR     = Path("vla/3_imitation_learning/3_1_act/outputs") / RUN_NAME

# ─────────────────────── 训练参数 ───────────────────────────
TRAIN_STEPS   = 100000
BATCH_SIZE    = 256
EVAL_EPISODES = 10
LOG_FREQ      = 5
WANDB_PROJECT = "act-libero"
EPOCHS_PER_EVAL = 2


def get_steps_per_epoch(num_episodes: int) -> int:
    return max(1, (num_episodes + BATCH_SIZE - 1) // BATCH_SIZE)


def get_task_episodes(metadata: LeRobotDatasetMetadata, task_name: str) -> list[int]:
    """从数据集 metadata 中筛选出属于指定 task 的 episode。"""
    if "tasks" in metadata.episodes.features:
        selected = []
        for i in range(len(metadata.episodes)):
            row = metadata.episodes[i]
            tasks = row["tasks"]
            if len(tasks) == 1 and tasks[0] == task_name:
                selected.append(int(row["episode_index"]))
        return selected

    target_task_index = int(metadata.tasks.loc[task_name, "task_index"])
    metadata.pull_from_repo(allow_patterns="data/")

    selected = []
    for parquet_path in sorted((metadata.root / "data").glob("chunk-*/*.parquet")):
        df = pd.read_parquet(parquet_path, columns=["episode_index", "task_index"])
        matched = df.loc[df["task_index"] == target_task_index, "episode_index"].unique().tolist()
        selected.extend(int(ep) for ep in matched)

    return sorted(set(selected))


def main() -> None:
    metadata = LeRobotDatasetMetadata(DATASET_ID)
    episodes = get_task_episodes(metadata, TARGET_TASK)
    if not episodes:
        raise RuntimeError(f"未找到 task: {TARGET_TASK}")

    print(f"[数据] 数据集: {DATASET_ID}")
    print(f"[数据] 目标 task: {TARGET_TASK}")
    print(f"[数据] 筛选到 {len(episodes)} 条轨迹, episode_ids: {episodes[:5]}...")

    steps_per_epoch = get_steps_per_epoch(len(episodes))
    eval_freq = steps_per_epoch * EPOCHS_PER_EVAL
    save_freq = eval_freq

    policy_cfg = ACTConfig(
        device="cuda",
        push_to_hub=False,
    )

    dataset_cfg = DatasetConfig(
        repo_id=DATASET_ID,
        episodes=episodes,
        use_imagenet_stats=False,
        video_backend="pyav",
    )

    env_cfg = LiberoEnvConfig(
        task=TARGET_SUITE,
        task_ids=[TARGET_TASK_ID],
        obs_type="pixels_agent_pos",
        observation_height=256,
        observation_width=256,
    )

    cfg = TrainPipelineConfig(
        dataset=dataset_cfg,
        env=env_cfg,
        policy=policy_cfg,
        output_dir=OUTPUT_DIR,
        batch_size=BATCH_SIZE,
        num_workers=2,
        steps=TRAIN_STEPS,
        eval_freq=eval_freq,
        save_freq=save_freq,
        log_freq=LOG_FREQ,
        save_checkpoint=True,
        wandb=WandBConfig(enable=True, project=WANDB_PROJECT),
        eval=EvalConfig(
            n_episodes=EVAL_EPISODES,
            batch_size=1,
            use_async_envs=False,
        ),
    )

    cfg.validate()

    print(f"\n[训练] 总步数: {TRAIN_STEPS}")
    print(f"[训练] batch_size: {BATCH_SIZE}")
    print(f"[训练] steps_per_epoch: {steps_per_epoch}")
    print(f"[训练] 每 {EPOCHS_PER_EVAL} 个 epoch 评估/保存一次")
    print(f"[训练] eval_freq: {eval_freq}")
    print(f"[训练] save_freq: {save_freq}")
    print(f"[训练] log_freq: {LOG_FREQ}")
    print(f"[训练] 输出目录: {OUTPUT_DIR}")
    print(f"[训练] suite/task_id: {TARGET_SUITE}/{TARGET_TASK_ID}")
    print(f"[训练] policy.device: {policy_cfg.device}")
    print(f"[训练] wandb.project: {cfg.wandb.project}")
    print(f"[训练] eval videos: {OUTPUT_DIR / 'eval'}")
    print(
        "[训练] 默认 ACT: "
        f"dim_model={policy_cfg.dim_model}, chunk_size={policy_cfg.chunk_size}, "
        f"n_action_steps={policy_cfg.n_action_steps}, n_encoder_layers={policy_cfg.n_encoder_layers}"
    )
    print()

    train(cfg)


if __name__ == "__main__":
    main()
