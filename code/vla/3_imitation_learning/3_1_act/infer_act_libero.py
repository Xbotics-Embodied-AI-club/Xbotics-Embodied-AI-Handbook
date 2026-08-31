"""把训练好的 ACT 放回 LIBERO 仿真器里跑一次闭环，录像并记录推理时延。

对应第10讲 3.1 节：训练脚本 `train_act_libero.py` 存下 checkpoint 之后，用这个脚本
加载它、在仿真器里完整跑一个 episode，看策略到底会不会做这件事。

在 `code/` 目录下运行：`uv run python vla/3_imitation_learning/3_1_act/infer_act_libero.py`
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import add_envs_task, preprocess_observation

try:
    from lerobot.policies import get_policy_class, make_pre_post_processors
except ImportError:  # 兼容旧版 LeRobot。
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

from lerobot.utils.io_utils import write_video
import lerobot.policies  # noqa: F401  确保 policy registry 完成注册。


# LIBERO 底层是 MuJoCo，要离屏渲染出图像喂给策略；EGL 后端不需要桌面环境。
os.environ["MUJOCO_GL"] = "egl"

# 一个 episode 最多跑多少步。LIBERO 的成功判定只在回合结束那一步给出，
# 步数给太小会把还没做完的尝试当成失败。
MAX_STEPS = 300


def set_episode_index(env, episode_index: int | None) -> None:
    """固定 LIBERO 的初始状态，让同一次实验可以复现。

    LIBERO 每条任务有一组预设初始状态，默认按顺序轮换。评估一个改动到底有没有用时，
    初始状态不固定就分不清是改动生效了还是这一局刚好简单。

    Args:
        env: 已创建的向量化 LIBERO 环境。
        episode_index: 要固定到第几个预设初始状态；`None` 表示保持默认轮换。
    """
    if episode_index is None:
        return
    for inner_env in env.envs:
        inner_env.episode_index = episode_index
        inner_env.init_state_id = episode_index


def success_from_info(info: dict[str, Any]) -> bool:
    """从环境返回的 info 里取出这一回合成功与否。

    成功信号只在回合结束那一步出现在 `final_info` 里，中途各步都没有；
    所以取不到时按「尚未成功」处理，而不是报错。

    Args:
        info: `env.step()` 返回的 info 字典。

    Returns:
        这一步是否给出了成功信号。
    """
    final_info = info.get("final_info")
    if not isinstance(final_info, dict) or "is_success" not in final_info:
        return False
    successes = final_info["is_success"]
    if torch.is_tensor(successes):
        successes = successes.detach().cpu().numpy()
    return bool(np.asarray(successes).reshape(-1)[0])


def main() -> None:
    """加载 checkpoint、跑一个闭环 episode、存视频与统计结果。"""
    # 固定随机种子，让每次跑拿到同一段 rollout，便于对照。
    seed = 7
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 指向 train_act_libero.py 的输出目录。`last` 是 LeRobot 存的最新 checkpoint 软链，
    # 想看训练早期的表现就把它换成具体的步数目录。
    policy_path = Path(
        "vla/3_imitation_learning/3_1_act/outputs/act_libero_goal_plate/checkpoints/last/pretrained_model"
    )

    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    policy_cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

    # strict=False 允许权重里缺少一些运行期才建的 buffer；这些 buffer 不参与前向，
    # 跳过它们不影响推理结果。
    policy = get_policy_class(policy_cfg.type).from_pretrained(
        policy_path,
        config=policy_cfg,
        strict=False,
    )

    # 前后处理器必须和权重一起加载：preprocessor 用训练时那份统计量做归一化，
    # postprocessor 做反归一化。统计量对不上，动作尺度就整体错了。
    preprocessor, postprocessor = make_pre_post_processors(policy_cfg, pretrained_path=policy_path)

    # 环境参数要和训练时一致，否则策略看到的画面和它训练时不是一回事。
    task_suite = "libero_goal"
    task_id = 8
    env_cfg = LiberoEnvConfig(
        task=task_suite,
        task_ids=[task_id],
        obs_type="pixels_agent_pos",
        observation_height=256,
        observation_width=256,
        episode_length=MAX_STEPS,
    )
    env = make_env(env_cfg, n_envs=1)[task_suite][task_id]

    # env processor 负责把 LIBERO 原始观测里嵌套的 robot_state 压平成 policy 要的
    # observation.state，再把 policy 输出的动作转回 LIBERO 接受的格式。
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg, policy_cfg)

    frames: list[np.ndarray] = []
    inference_times_ms: list[float] = []
    success = False
    episode_index = None

    try:
        policy.reset()
        set_episode_index(env, episode_index)
        observation, _ = env.reset(seed=[seed + (episode_index or 0)])

        for step in range(MAX_STEPS):
            # 观测走两层处理：先 LIBERO -> LeRobot 的通用格式，再 -> 这个 policy 的输入格式。
            observation_batch = preprocess_observation(observation)
            observation_batch = add_envs_task(env, observation_batch)
            observation_batch = env_preprocessor(observation_batch)
            observation_batch = preprocessor(observation_batch)

            start = time.perf_counter()
            with torch.inference_mode():
                # select_action 每次只返回一步：动作队列没空时它直接弹出上次预测的结果，
                # 所以这里量到的时延会是"偶尔很长、大部分很短"。
                action = policy.select_action(observation_batch)
            inference_times_ms.append((time.perf_counter() - start) * 1000)

            # 动作走回程的两层处理，和观测那两层对称。
            action = postprocessor(action)
            action = env_postprocessor({"action": action})["action"]
            action = action.cpu().numpy() if torch.is_tensor(action) else np.asarray(action)

            observation, _, terminated, truncated, info = env.step(action)
            frames.append(env.envs[0].render())

            success = success or success_from_info(info)
            if bool(terminated[0]) or bool(truncated[0]):
                break

        video_path = Path("vla/3_imitation_learning/3_1_act/output/act_libero_rollout.mp4")
        if frames:
            video_path.parent.mkdir(parents=True, exist_ok=True)
            write_video(str(video_path), frames, fps=20)

        # 时延同时记均值和 p95：均值被大量"直接弹队列"的步数拉低，p95 才接近
        # 真正重新跑一次模型的开销，而后者决定了控制频率的上限。
        result = {
            "policy_path": str(policy_path),
            "suite": task_suite,
            "task_id": task_id,
            "episode_index": episode_index,
            "success": success,
            "steps": step + 1,
            "avg_policy_inference_ms": float(np.mean(inference_times_ms)) if inference_times_ms else None,
            "p95_policy_inference_ms": (
                float(np.percentile(inference_times_ms, 95)) if inference_times_ms else None
            ),
            "video": str(video_path) if frames else None,
        }
        result_path = Path("vla/3_imitation_learning/3_1_act/output/act_libero_rollout.json")
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
