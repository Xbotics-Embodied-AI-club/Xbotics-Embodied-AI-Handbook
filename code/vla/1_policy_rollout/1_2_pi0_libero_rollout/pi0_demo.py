from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import add_envs_task, preprocess_observation
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.utils.io_utils import write_video
import lerobot.policies  # noqa: F401

# MuJoCo 的离屏渲染需要 EGL。
os.environ.setdefault("MUJOCO_GL", "egl")

# 关闭 torch compile / inductor，避免首次运行时出现大量 autotune 开销，
# 让课堂 demo 更稳定、更可复现。
os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

# 下面这组参数是已经验证过能跑出 success=True 的固定配置。
POLICY_PATH = "lerobot/pi0_libero_finetuned_v044"
TASK_SUITE = "libero_goal"
TASK_ID = 5
EPISODE_INDEX = 2
MAX_STEPS = 180
FPS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 7
OUT_PATH = Path("vla/1_policy_rollout/1_2_pi0_libero_rollout/output/pi0_libero_success.mp4")


def set_episode_index(env, episode_index: int) -> None:
    # LeRobot 的 LIBERO 向量环境外面包了一层 SyncVectorEnv。
    # 真正控制初始状态的是里面每个子环境的 episode_index / init_state_id。
    # 这里只跑 1 个环境，所以直接把第 0 个子环境切到我们选好的成功初始状态。
    for inner_env in env.envs:
        inner_env.episode_index = episode_index
        inner_env.init_state_id = episode_index


def main() -> None:
    # 固定随机种子，保证每次讲课演示时拿到相同的 rollout。
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 1. 读取 Hugging Face 上保存的 pi0 配置。
    # 2. 把 device 改成当前机器可用的 cuda / cpu。
    # 3. 加载策略本体和它对应的 preprocess / postprocess 流水线。
    # strict=False 是为了兼容当前 lerobot 版本和权重里少量 buffer 命名差异。
    policy_cfg = PreTrainedConfig.from_pretrained(POLICY_PATH)
    policy_cfg.device = DEVICE
    policy = get_policy_class(policy_cfg.type).from_pretrained(POLICY_PATH, config=policy_cfg, strict=False)
    preprocessor, postprocessor = make_pre_post_processors(policy_cfg, pretrained_path=POLICY_PATH)

    # 构建 LIBERO 环境。这里只保留和成功案例匹配的最小参数。
    env_cfg = LiberoEnvConfig(
        task=TASK_SUITE,
        task_ids=[TASK_ID],
        obs_type="pixels_agent_pos",
        observation_height=256,
        observation_width=256,
        episode_length=MAX_STEPS,
    )
    env = make_env(env_cfg, n_envs=1)[TASK_SUITE][TASK_ID]
    print(f"task: {env.envs[0].task} | instruction: {env.envs[0].task_description}")
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg, policy_cfg)

    frames = []
    success = False

    try:
        policy.reset()

        # 切换到底层 LIBERO 子环境里已经验证可成功的 init state。
        set_episode_index(env, EPISODE_INDEX)
        observation, _ = env.reset(seed=[SEED + EPISODE_INDEX])

        for _ in range(MAX_STEPS):
            # 环境原始 observation 先转成 LeRobot 约定的扁平 key 格式，
            # 再补上 task 文本，随后送进 env processor 和 policy processor。
            observation_batch = preprocess_observation(observation)
            observation_batch = add_envs_task(env, observation_batch)
            observation_batch = env_preprocessor(observation_batch)
            observation_batch = preprocessor(observation_batch)

            # pi0 每次根据当前观测输出一个动作。
            with torch.inference_mode():
                action = policy.select_action(observation_batch)

            # postprocessor 负责把 policy 输出还原回环境动作空间。
            action = postprocessor(action)
            action = env_postprocessor({"action": action})["action"].cpu().numpy()

            # 执行动作，并把渲染帧缓存下来，最后统一写 mp4。
            observation, _, terminated, truncated, info = env.step(action)
            frames.append(env.envs[0].render())

            # LIBERO 的 success 信号放在 final_info 里。
            if "final_info" in info and isinstance(info["final_info"], dict):
                success = bool(info["final_info"]["is_success"][0])

            if bool(terminated[0]) or bool(truncated[0]):
                break

        # 这两个检查保证 demo 不是“看起来运行了”，而是真的有结果、而且真的成功。
        if not frames:
            raise RuntimeError("no frames")
        if not success:
            raise RuntimeError("no success")

        # 把整段 rollout 直接导出成 mp4。
        write_video(str(OUT_PATH), frames, fps=FPS)
        print(OUT_PATH)
    finally:
        # 关闭环境，避免 MuJoCo / EGL 资源泄漏。
        env.close()


if __name__ == "__main__":
    main()
