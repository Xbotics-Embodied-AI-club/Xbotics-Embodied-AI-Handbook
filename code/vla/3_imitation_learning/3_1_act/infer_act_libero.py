"""在 LIBERO 仿真里直接运行本地 ACT policy。"""
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


os.environ.setdefault("MUJOCO_GL", "egl")


# 课堂演示配置：只需要改这里，不需要命令行参数。
POLICY_PATH = Path("vla/3_imitation_learning/3_1_act/outputs/act_libero_goal_plate_20260409_141037/checkpoints/000954/pretrained_model")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TASK_SUITE = "libero_goal"
TASK_ID = 8
EPISODE_INDEX: int | None = None
MAX_STEPS = 300
OBS_HEIGHT = 256
OBS_WIDTH = 256
FPS = 20
SEED = 7
STRICT_LOAD = False
SAVE_VIDEO = True
VIDEO_PATH = Path("vla/3_imitation_learning/3_1_act/output/act_libero_rollout.mp4")
RESULT_PATH = Path("vla/3_imitation_learning/3_1_act/output/act_libero_rollout.json")


def set_episode_index(env, episode_index: int | None) -> None:
    """固定 LIBERO 初始状态，方便复现实验；None 表示使用默认顺序。"""
    if episode_index is None:
        return
    for inner_env in env.envs:
        inner_env.episode_index = episode_index
        inner_env.init_state_id = episode_index


def success_from_info(info: dict[str, Any]) -> bool:
    """LIBERO 成功信息在 final_info 里，转成普通 bool 方便打印/保存。"""
    final_info = info.get("final_info")
    if not isinstance(final_info, dict) or "is_success" not in final_info:
        return False
    successes = final_info["is_success"]
    if torch.is_tensor(successes):
        successes = successes.detach().cpu().numpy()
    return bool(np.asarray(successes).reshape(-1)[0])


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # 加载 LeRobot 训练出的 ACT policy，以及 policy 自带的输入/输出 processor。
    policy_cfg = PreTrainedConfig.from_pretrained(POLICY_PATH)
    policy_cfg.device = DEVICE

    policy = get_policy_class(policy_cfg.type).from_pretrained(
        POLICY_PATH,
        config=policy_cfg,
        strict=STRICT_LOAD,
    )
    preprocessor, postprocessor = make_pre_post_processors(policy_cfg, pretrained_path=POLICY_PATH)

    # 创建 LIBERO 环境；env processor 负责把 LIBERO observation 转成 policy 需要的 key。
    env_cfg = LiberoEnvConfig(
        task=TASK_SUITE,
        task_ids=[TASK_ID],
        obs_type="pixels_agent_pos",
        observation_height=OBS_HEIGHT,
        observation_width=OBS_WIDTH,
        episode_length=MAX_STEPS,
    )
    env = make_env(env_cfg, n_envs=1)[TASK_SUITE][TASK_ID]
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg, policy_cfg)

    # 记录视频和推理时间，方便演示时快速看结果。
    frames: list[np.ndarray] = []
    inference_times_ms: list[float] = []
    success = False

    try:
        policy.reset()
        set_episode_index(env, EPISODE_INDEX)
        observation, _ = env.reset(seed=[SEED + (EPISODE_INDEX or 0)])

        for step in range(MAX_STEPS):
            # observation: LIBERO -> LeRobot env processor -> policy processor。
            observation_batch = preprocess_observation(observation)
            observation_batch = add_envs_task(env, observation_batch)
            observation_batch = env_preprocessor(observation_batch)
            observation_batch = preprocessor(observation_batch)

            start = time.perf_counter()
            with torch.inference_mode():
                action = policy.select_action(observation_batch)
            inference_times_ms.append((time.perf_counter() - start) * 1000)

            # action: policy tensor -> policy postprocessor -> LIBERO action。
            action = postprocessor(action)
            action = env_postprocessor({"action": action})["action"]
            action = action.cpu().numpy() if torch.is_tensor(action) else np.asarray(action)

            observation, _, terminated, truncated, info = env.step(action)

            if SAVE_VIDEO:
                frames.append(env.envs[0].render())

            success = success or success_from_info(info)
            if bool(terminated[0]) or bool(truncated[0]):
                break

        if SAVE_VIDEO and frames:
            VIDEO_PATH.parent.mkdir(parents=True, exist_ok=True)
            write_video(str(VIDEO_PATH), frames, fps=FPS)

        # 保存一个轻量结果文件，方便课堂上直接看统计。
        result = {
            "policy_path": str(POLICY_PATH),
            "suite": TASK_SUITE,
            "task_id": TASK_ID,
            "episode_index": EPISODE_INDEX,
            "success": success,
            "steps": step + 1,
            "avg_policy_inference_ms": float(np.mean(inference_times_ms)) if inference_times_ms else None,
            "p95_policy_inference_ms": float(np.percentile(inference_times_ms, 95)) if inference_times_ms else None,
            "video": str(VIDEO_PATH) if SAVE_VIDEO and frames else None,
        }
        RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESULT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
