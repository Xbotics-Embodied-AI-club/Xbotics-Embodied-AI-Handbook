"""在 LIBERO 的一条短程任务上训练 ACT。

对应第10讲 3.1 节与 3.3.1 节：从 `lerobot/libero` 里筛出「把碗放到盘子上」这一条
任务的轨迹，用默认 `ACTConfig` 训练，并在训练过程中定期回 LIBERO 仿真器里做闭环评估。

在 `code/` 目录下运行：`uv run python vla/3_imitation_learning/3_1_act/train_act_libero.py`
"""
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

# LIBERO 底层是 MuJoCo，评估时要离屏渲染出图像喂给策略；EGL 后端不需要桌面环境。
os.environ["MUJOCO_GL"] = "egl"

# 显存放得下就尽量大。ACT 的监督信号是动作块级的，一条样本就带 100 步动作。
BATCH_SIZE = 256


def get_steps_per_epoch(num_episodes: int) -> int:
    """按筛出来的轨迹条数估算一个 epoch 有多少训练步。

    这里用 episode 数而不是帧数做近似，够用来定评估间隔——只要间隔稳定，
    曲线就能对齐着看，不需要精确对应「看完一遍数据集」。

    Args:
        num_episodes: 筛选出的轨迹条数。

    Returns:
        一个 epoch 的训练步数，至少为 1。
    """
    return max(1, (num_episodes + BATCH_SIZE - 1) // BATCH_SIZE)


def get_task_episodes(metadata: LeRobotDatasetMetadata, task_name: str) -> list[int]:
    """从数据集 metadata 中筛出属于指定 task 的 episode 编号。

    `lerobot/libero` 是把十个任务套件混在一起的总数据集，直接拿来训会把十几个任务
    的示教混进同一个策略。这里只留目标任务那一份。

    走两条路是因为数据集版本不同：新版 metadata 里每条 episode 自带 `tasks` 字段，
    直接读就行；旧版只有一张 task 表，得把 parquet 拉下来按 `task_index` 反查。

    Args:
        metadata: 已加载的数据集元信息。
        task_name: 任务的自然语言描述，要和数据集里记录的字符串完全一致。

    Returns:
        升序的 episode 编号列表；没有匹配时返回空列表。
    """
    if "tasks" in metadata.episodes.features:
        selected = []
        for i in range(len(metadata.episodes)):
            row = metadata.episodes[i]
            tasks = row["tasks"]
            # 只要单任务 episode：`len(tasks) == 1` 排掉那些被标了多个任务的长程轨迹。
            if len(tasks) == 1 and tasks[0] == task_name:
                selected.append(int(row["episode_index"]))
        return selected

    target_task_index = int(metadata.tasks.loc[task_name, "task_index"])
    metadata.pull_from_repo(allow_patterns="data/")

    selected = []
    for parquet_path in sorted((metadata.root / "data").glob("chunk-*/*.parquet")):
        # 只读这两列，避免把整份图像数据也载进内存。
        df = pd.read_parquet(parquet_path, columns=["episode_index", "task_index"])
        matched = df.loc[df["task_index"] == target_task_index, "episode_index"].unique().tolist()
        selected.extend(int(ep) for ep in matched)

    return sorted(set(selected))


# 一次评估跑多少个 episode。成功率是 0/1 统计，跑得太少方差会大到看不出趋势。
EVAL_EPISODES = 10

# 每训练多少个 epoch 评估并保存一次。评估要在仿真器里跑完整 rollout，比训练一步贵得多，
# 所以不能每步都评。
EPOCHS_PER_EVAL = 2


def main() -> None:
    """筛数据、组装训练配置、启动训练。"""
    dataset_id = "lerobot/libero"
    target_task = "put the bowl on the plate"

    metadata = LeRobotDatasetMetadata(dataset_id)
    episodes = get_task_episodes(metadata, target_task)
    if not episodes:
        raise RuntimeError(f"未找到 task: {target_task}")

    print(f"[数据] 数据集: {dataset_id}")
    print(f"[数据] 目标 task: {target_task}")
    print(f"[数据] 筛选到 {len(episodes)} 条轨迹, episode_ids: {episodes[:5]}...")

    steps_per_epoch = get_steps_per_epoch(len(episodes))
    eval_freq = steps_per_epoch * EPOCHS_PER_EVAL

    # 超参全用 ACTConfig 的默认值：chunk_size=100、latent_dim=32、kl_weight=10.0。
    # 这组值来自原始 ACT 那批任务，换任务时至少要重调 chunk_size（它按秒理解才有意义）。
    policy_cfg = ACTConfig(
        device="cuda",
        push_to_hub=False,
    )

    dataset_cfg = DatasetConfig(
        repo_id=dataset_id,
        episodes=episodes,
        # 观测归一化用这份数据集自己的统计量，而不是 ImageNet 的均值方差——
        # 仿真图像的分布和自然照片差得远。
        use_imagenet_stats=False,
        video_backend="pyav",
    )

    # 评估环境要和数据集对齐：同一个套件、同一条任务、同样的图像尺寸，
    # 否则策略在评估时看到的画面和训练时不是一回事。
    env_cfg = LiberoEnvConfig(
        task="libero_goal",
        task_ids=[8],
        obs_type="pixels_agent_pos",
        observation_height=256,
        observation_width=256,
    )

    run_name = datetime.now().strftime("act_libero_goal_plate_%Y%m%d_%H%M%S")
    output_dir = Path("vla/3_imitation_learning/3_1_act/outputs") / run_name

    cfg = TrainPipelineConfig(
        dataset=dataset_cfg,
        env=env_cfg,
        policy=policy_cfg,
        output_dir=output_dir,
        batch_size=BATCH_SIZE,
        num_workers=2,
        steps=100000,
        eval_freq=eval_freq,
        # 每次评估都存一份 checkpoint，这样曲线上任何一个点都能回去复现。
        save_freq=eval_freq,
        log_freq=5,
        save_checkpoint=True,
        wandb=WandBConfig(enable=True, project="act-libero"),
        eval=EvalConfig(
            n_episodes=EVAL_EPISODES,
            batch_size=1,
            use_async_envs=False,
        ),
    )

    cfg.validate()

    print(f"\n[训练] 总步数: {cfg.steps}")
    print(f"[训练] batch_size: {BATCH_SIZE}")
    print(f"[训练] steps_per_epoch: {steps_per_epoch}")
    print(f"[训练] 每 {EPOCHS_PER_EVAL} 个 epoch 评估/保存一次")
    print(f"[训练] eval_freq: {eval_freq}")
    print(f"[训练] 输出目录: {output_dir}")
    print(f"[训练] policy.device: {policy_cfg.device}")
    print(f"[训练] eval videos: {output_dir / 'eval'}")
    print(
        "[训练] 默认 ACT: "
        f"dim_model={policy_cfg.dim_model}, chunk_size={policy_cfg.chunk_size}, "
        f"n_action_steps={policy_cfg.n_action_steps}, n_encoder_layers={policy_cfg.n_encoder_layers}"
    )
    print()

    train(cfg)


if __name__ == "__main__":
    main()
