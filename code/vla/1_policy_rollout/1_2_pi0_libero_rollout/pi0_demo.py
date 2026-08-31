"""用 $\\pi_0$ 在 LIBERO 里跑通第一个策略闭环：观测进、动作出、录成视频。

第8讲 4.3 节的配套代码。把 $\\pi_0$ 当成一个黑盒策略，在 libero_goal 的一个任务上
闭环跑到底，成功后把整段 rollout 导出成 mp4。
"""
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

# MuJoCo 的离屏渲染需要 EGL，这样没有桌面环境也能渲出画面。
os.environ["MUJOCO_GL"] = "egl"

# 关掉 torch 的编译与自动调优。开着的话首次运行要先花几分钟做 autotune，
# 而且每次挑中的 kernel 不一定相同，演示时结果不好复现。
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"


def set_episode_index(env, episode_index: int) -> None:
    """把向量环境里的子环境切到指定的 LIBERO 预设初始状态。

    LeRobot 的 LIBERO 环境外面包了一层 SyncVectorEnv，真正决定初始状态的是里面
    每个子环境的 episode_index / init_state_id，从外层的向量环境改不到。

    Args:
        env: make_env 返回的向量环境。
        episode_index: LIBERO 预设初始状态的序号。
    """
    for inner_env in env.envs:
        inner_env.episode_index = episode_index
        inner_env.init_state_id = episode_index


def main() -> None:
    """在一个 LIBERO 任务上闭环跑一遍 $\\pi_0$，成功则把 rollout 存成 mp4。

    Raises:
        RuntimeError: 一帧画面都没渲染出来，或这一次 rollout 没有完成任务。
    """
    # 固定随机种子，让每次演示拿到同一段 rollout。
    seed = 7
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 这份权重是 π0 在 LIBERO 上微调过的版本。π0 的基座没见过 LIBERO 仿真数据，
    # 换成基座权重零样本跑，成功率会掉到接近 0。
    policy_path = "lerobot/pi0_libero_finetuned_v044"
    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    policy_cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

    # strict=False：权重里有少量 buffer 的命名和当前 lerobot 版本对不上，
    # 这些 buffer 不参与前向计算，跳过它们不影响推理结果。
    policy = get_policy_class(policy_cfg.type).from_pretrained(
        policy_path, config=policy_cfg, strict=False
    )

    # 归一化统计量跟着权重一起存，这两个 processor 就是 3.4 节说的
    # "正向归一化"与"反归一化"，训练和推理共用同一套统计量。
    preprocessor, postprocessor = make_pre_post_processors(policy_cfg, pretrained_path=policy_path)

    task_suite = "libero_goal"
    task_id = 5
    max_steps = 180
    env_cfg = LiberoEnvConfig(
        task=task_suite,
        task_ids=[task_id],
        obs_type="pixels_agent_pos",
        observation_height=256,
        observation_width=256,
        episode_length=max_steps,
    )
    env = make_env(env_cfg, n_envs=1)[task_suite][task_id]
    print(f"task: {env.envs[0].task} | instruction: {env.envs[0].task_description}")

    # 环境侧也有一对 processor：把 LIBERO 的观测/动作格式与策略的接口对齐。
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg, policy_cfg)

    frames = []
    success = False

    try:
        policy.reset()

        # 这个初始状态已经验证过能成功，换一个初始状态成功率就不一定了——
        # 3.2 节说的分布漂移，从起始位姿就开始起作用。
        episode_index = 2
        set_episode_index(env, episode_index)
        observation, _ = env.reset(seed=[seed + episode_index])

        for _ in range(max_steps):
            # 环境原始 observation 先转成 LeRobot 约定的扁平 key 格式，
            # 再补上任务的语言指令，随后依次过环境侧和策略侧的预处理。
            observation_batch = preprocess_observation(observation)
            observation_batch = add_envs_task(env, observation_batch)
            observation_batch = env_preprocessor(observation_batch)
            observation_batch = preprocessor(observation_batch)

            # 整个闭环里，策略只做这一件事：看一眼观测，给出一个动作。
            with torch.inference_mode():
                action = policy.select_action(observation_batch)

            # 反归一化 + 还原成环境的动作口径，正是 3 节那条"来回路"的回程。
            action = postprocessor(action)
            action = env_postprocessor({"action": action})["action"].cpu().numpy()

            # 动作作用回环境，环境给出新的观测——闭环的下一圈从这里开始。
            observation, _, terminated, truncated, info = env.step(action)
            frames.append(env.envs[0].render())

            # LIBERO 把成功信号放在 final_info 里，只在回合结束那一步出现。
            if "final_info" in info and isinstance(info["final_info"], dict):
                success = bool(info["final_info"]["is_success"][0])

            if bool(terminated[0]) or bool(truncated[0]):
                break

        # 跑完不等于跑对。这两个检查把"跑完了但什么都没发生"和
        # "跑完了但任务没成"这两种情况直接变成报错，不让它们混成一次成功。
        if not frames:
            raise RuntimeError("no frames")
        if not success:
            raise RuntimeError("no success")

        # 路径相对 code/ 目录写，和 README 里的运行方式（cd code 之后再跑）一致，
        # 脚本和 notebook 用同一个工作目录，产物就不会散到两处去。
        fps = 10
        out_path = Path("vla/1_policy_rollout/1_2_pi0_libero_rollout/output/pi0_libero_success.mp4")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_video(str(out_path), frames, fps=fps)
        print(out_path)
    finally:
        # MuJoCo / EGL 的上下文不会自己释放，异常退出时也必须关掉。
        env.close()


if __name__ == "__main__":
    main()
