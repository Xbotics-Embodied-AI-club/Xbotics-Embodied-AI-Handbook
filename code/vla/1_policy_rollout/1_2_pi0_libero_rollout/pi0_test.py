"""白噪声对照实验：$\\pi_0$ 在 LIBERO 上到底有没有在用视觉。

第8讲 4.4 节的配套代码。同一批任务跑两遍，除了送进网络的图像（真实相机图 / 均匀白噪声）
之外其余输入完全一致，比较两次的成功率。同时也是一个批量评测脚本的样板：多任务、多初始
状态、逐 episode 存视频与动作序列，最后汇总成一张成功率表。
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import add_envs_task, preprocess_observation
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.utils.io_utils import write_video
import lerobot.policies  # noqa: F401
from libero.libero import benchmark

# MuJoCo 的离屏渲染需要 EGL，这样没有桌面环境也能渲出画面。
os.environ["MUJOCO_GL"] = "egl"

# 关掉 torch 的编译与自动调优。批量评测要反复建环境、反复推理，
# 开着 autotune 不但慢，还会让两组对照跑在不同的 kernel 上。
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"


def replace_images_with_white_noise(observation_batch: dict[str, Any]) -> dict[str, Any]:
    """把观测里的图像换成同形状的均匀白噪声，其余字段一律不动。

    只替换 `observation.images.` 开头的键，本体状态、语言指令都保持原样——
    对照实验一次只能改一个变量，否则成功率掉下来也说不清是被哪一项改掉的。

    Args:
        observation_batch: 已经过 preprocess_observation 摊平的观测字典。

    Returns:
        原地修改后的同一个观测字典。
    """
    for key, value in observation_batch.items():
        if key.startswith("observation.images.") and torch.is_tensor(value):
            observation_batch[key] = torch.rand_like(value)
    return observation_batch


def apply_image_mode(observation_batch: dict[str, Any], image_mode: str) -> dict[str, Any]:
    """按实验组别决定这一帧的图像是原图还是白噪声。

    Args:
        observation_batch: 摊平后的观测字典。
        image_mode: `"real"` 用真实相机图，`"white_noise"` 换成白噪声。

    Returns:
        处理后的观测字典。

    Raises:
        ValueError: image_mode 不是上述两种之一。
    """
    if image_mode == "white_noise":
        return replace_images_with_white_noise(observation_batch)
    if image_mode == "real":
        return observation_batch
    raise ValueError(f"Unsupported image_mode: {image_mode}")


def _successes_from_info(info: dict[str, Any], n_envs: int) -> np.ndarray:
    """从环境返回的 info 里取出这一步的成功标志。

    LIBERO 只在回合结束那一步把 is_success 放进 final_info，其余步没有这个字段，
    所以取不到时一律当作"还没成功"，而不是报错。

    Args:
        info: env.step 返回的 info 字典。
        n_envs: 并行的子环境个数。

    Returns:
        长度为 n_envs 的布尔数组。
    """
    if "final_info" not in info:
        return np.zeros(n_envs, dtype=bool)

    final_info = info["final_info"]
    if not isinstance(final_info, dict) or "is_success" not in final_info:
        return np.zeros(n_envs, dtype=bool)

    successes = final_info["is_success"]
    if torch.is_tensor(successes):
        successes = successes.detach().cpu().numpy()
    else:
        successes = np.asarray(successes)
    return successes.astype(bool).reshape(-1)[:n_envs]


def get_suite_task_ids(suite: str) -> list[int]:
    """列出一个 LIBERO 套件里全部任务的 id。

    Args:
        suite: 套件名，如 `"libero_goal"`、`"libero_10"`。

    Returns:
        从 0 开始的任务 id 列表。

    Raises:
        ValueError: 套件名不在 LIBERO 的注册表里。
    """
    benchmarks = benchmark.get_benchmark_dict()
    if suite not in benchmarks:
        raise ValueError(f"Unknown LIBERO suite: {suite}")
    task_suite = benchmarks[suite]()
    return list(range(len(task_suite.tasks)))


def render_active_envs(env, frames: list[list[np.ndarray]], done: np.ndarray) -> None:
    """只给还没结束的子环境抓帧。

    已经结束的回合再渲染下去，视频尾巴上会多出一段静止画面，看起来像卡住了。

    Args:
        env: 向量环境。
        frames: 每个子环境一个帧列表，原地追加。
        done: 每个子环境是否已结束。
    """
    for episode_index, inner_env in enumerate(env.envs):
        if not done[episode_index]:
            frames[episode_index].append(inner_env.render())


def get_episode_metadata(env) -> list[dict[str, str]]:
    """记下每个子环境的任务名与语言指令，写进结果文件备查。

    Args:
        env: 向量环境。

    Returns:
        与子环境一一对应的元数据列表。
    """
    metadata = []
    for inner_env in env.envs:
        metadata.append(
            {
                "task_name": str(getattr(inner_env, "task", "")),
                "task_description": str(getattr(inner_env, "task_description", "")),
            }
        )
    return metadata


def record_active_actions(
    actions: list[list[list[float]]],
    action: np.ndarray,
    done: np.ndarray,
) -> None:
    """把这一步的动作存下来，供事后核对数值量级。

    动作序列是排查 3.3、3.4 两类故障的第一手证据：相对口径下它应当是一串贴近 0
    的小数，出现大数就说明口径或归一化统计量有问题。

    Args:
        actions: 每个子环境一个动作列表，原地追加。
        action: 这一步下发的动作，形状为 (n_envs, action_dim)。
        done: 每个子环境是否已结束。
    """
    for episode_index, episode_action in enumerate(action):
        if not done[episode_index]:
            actions[episode_index].append(episode_action.astype(float).tolist())


def write_episode_artifacts(
    *,
    frames: list[list[np.ndarray]],
    actions: list[list[list[float]]],
    successes: np.ndarray,
    metadata: list[dict[str, str]],
    suite: str,
    task_id: int,
    image_mode: str,
    video_dir: Path,
    fps: int,
) -> list[dict[str, Any]]:
    """把每个回合的视频与动作序列落盘。

    成功率只是一个数字，回放和动作序列才说得清失败发生在哪一步，
    所以两组对照都完整存下来，而不是只留汇总。

    Args:
        frames: 每个回合的渲染帧。
        actions: 每个回合下发过的动作。
        successes: 每个回合是否成功。
        metadata: 每个回合的任务名与指令。
        suite: 套件名。
        task_id: 任务 id。
        image_mode: 这一组用的是原图还是白噪声。
        video_dir: 视频与 json 的输出根目录。
        fps: 导出视频的帧率。

    Returns:
        每个回合一条的产物索引，供汇总文件引用。
    """
    artifacts = []
    for episode_index, episode_frames in enumerate(frames):
        episode_dir = video_dir / suite / f"task_{task_id:02d}"
        video_path = episode_dir / f"episode_{episode_index:03d}.mp4"
        json_path = episode_dir / f"episode_{episode_index:03d}.json"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        write_video(str(video_path), episode_frames, fps=fps)

        episode_payload = {
            "suite": suite,
            "task_id": task_id,
            "episode_index": episode_index,
            "task_name": metadata[episode_index]["task_name"],
            "task_description": metadata[episode_index]["task_description"],
            "image_mode": image_mode,
            "success": bool(successes[episode_index]),
            "steps": len(actions[episode_index]),
            "action_dim": len(actions[episode_index][0]) if actions[episode_index] else 0,
            "actions": actions[episode_index],
            "video": str(video_path),
        }
        json_path.write_text(json.dumps(episode_payload, indent=2, ensure_ascii=False), encoding="utf-8")

        artifacts.append(
            {
                "episode_index": episode_index,
                "video": str(video_path),
                "json": str(json_path),
                "task_name": episode_payload["task_name"],
                "task_description": episode_payload["task_description"],
                "success": episode_payload["success"],
                "steps": episode_payload["steps"],
            }
        )
    return artifacts


# 长程套件一个回合要走的步数比 libero_90 多得多，步数给少了会把成功判成失败。
MAX_STEPS = {
    "libero_10": 520,
    "libero_90": 400,
}


def evaluate_task(
    *,
    policy,
    policy_cfg,
    preprocessor,
    postprocessor,
    suite: str,
    task_id: int,
    episodes: int,
    seed: int,
    image_mode: str,
    video_dir: Path,
    fps: int,
) -> dict[str, Any]:
    """在一个任务的多个初始状态上闭环跑 rollout，返回这个任务的成功率。

    每个初始状态是一个并行子环境，跑到全部结束或达到步数上限为止。

    Args:
        policy: 已加载的策略。
        policy_cfg: 策略配置，环境处理器要按它对齐接口。
        preprocessor: 策略侧预处理（含正向归一化）。
        postprocessor: 策略侧后处理（含反归一化）。
        suite: LIBERO 套件名。
        task_id: 任务 id。
        episodes: 评几个初始状态。
        seed: 随机种子，逐个子环境错开。
        image_mode: `"real"` 或 `"white_noise"`。
        video_dir: 视频输出根目录。
        fps: 导出视频的帧率。

    Returns:
        含 successes / success_rate / steps / artifacts 的结果字典。
    """
    env_cfg = LiberoEnvConfig(
        task=suite,
        task_ids=[task_id],
        obs_type="pixels_agent_pos",
        observation_height=256,
        observation_width=256,
        episode_length=MAX_STEPS.get(suite),
    )
    env = make_env(env_cfg, n_envs=episodes)[suite][task_id]
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg, policy_cfg)

    max_steps = env.call("_max_episode_steps")[0]
    successes = np.zeros(episodes, dtype=bool)
    done = np.zeros(episodes, dtype=bool)
    frames: list[list[np.ndarray]] = [[] for _ in range(episodes)]
    actions: list[list[list[float]]] = [[] for _ in range(episodes)]

    try:
        policy.reset()
        observation, _ = env.reset(seed=[seed + i for i in range(episodes)])
        metadata = get_episode_metadata(env)
        render_active_envs(env, frames, done)

        for step in range(max_steps):
            observation_batch = preprocess_observation(observation)
            observation_batch = apply_image_mode(observation_batch, image_mode)
            observation_batch = add_envs_task(env, observation_batch)
            observation_batch = env_preprocessor(observation_batch)
            observation_batch = preprocessor(observation_batch)

            with torch.inference_mode():
                action = policy.select_action(observation_batch)

            action = postprocessor(action)
            action = env_postprocessor({"action": action})["action"].cpu().numpy()
            record_active_actions(actions, action, done)

            observation, _, terminated, truncated, info = env.step(action)
            # 成功信号只出现一次，用 |= 记住它，不能只看最后一步。
            successes |= _successes_from_info(info, episodes)
            done |= terminated | truncated
            render_active_envs(env, frames, done)

            if np.all(done):
                break

        artifacts = write_episode_artifacts(
            frames=frames,
            actions=actions,
            successes=successes,
            metadata=metadata,
            suite=suite,
            task_id=task_id,
            image_mode=image_mode,
            video_dir=video_dir,
            fps=fps,
        )

        return {
            "suite": suite,
            "task_id": task_id,
            "episodes": episodes,
            "successes": int(successes.sum()),
            "success_rate": float(successes.mean()),
            "steps": int(step + 1),
            "artifacts": artifacts,
        }
    finally:
        # MuJoCo / EGL 的上下文不会自己释放，批量评测里漏关一次就会越跑越吃显存。
        env.close()


def main() -> None:
    """跑完选定套件的全部任务，把成功率汇总成一个 json。

    两组对照怎么做：把 image_mode 设成 "real" 跑一遍、设成 "white_noise" 再跑一遍，
    两次的 summary 摆在一起看。白噪声那一组掉得越多，说明策略越依赖视觉。
    """
    policy_path = "lerobot/pi0_libero_finetuned_v044"
    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    policy_cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = get_policy_class(policy_cfg.type).from_pretrained(
        policy_path, config=policy_cfg, strict=False
    )
    preprocessor, postprocessor = make_pre_post_processors(policy_cfg, pretrained_path=policy_path)

    # 两组对照唯一不同的就是这一行。
    image_mode = "white_noise"

    suites = ["libero_90", "libero_10"]
    task_ids = None       # None 表示整个套件；也可以写成 [0, 3, 7] 只评几个任务
    episodes = 1          # 每个任务评几个初始状态
    seed = 7
    fps = 10

    torch.manual_seed(seed)
    np.random.seed(seed)

    # 路径相对 code/ 目录写，和 README 里的运行方式（cd code 之后再跑）一致。
    output_dir = Path("vla/1_policy_rollout/1_2_pi0_libero_rollout/output")
    out_path = output_dir / "pi0_white_noise_results.json"
    video_dir = output_dir / "pi0_white_noise_videos"

    all_results: list[dict[str, Any]] = []

    for suite in suites:
        suite_task_ids = task_ids
        if suite_task_ids is None:
            suite_task_ids = get_suite_task_ids(suite)

        print(f"\n=== {suite}: {len(suite_task_ids)} tasks, {episodes} episodes/task ===")
        for task_id in suite_task_ids:
            result = evaluate_task(
                policy=policy,
                policy_cfg=policy_cfg,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                suite=suite,
                task_id=task_id,
                episodes=episodes,
                seed=seed,
                image_mode=image_mode,
                video_dir=video_dir,
                fps=fps,
            )
            all_results.append(result)
            print(
                f"{suite} task {task_id:02d}: "
                f"{result['successes']}/{result['episodes']} "
                f"success_rate={result['success_rate']:.3f}"
            )

    summary = {}
    for suite in suites:
        suite_results = [r for r in all_results if r["suite"] == suite]
        total_successes = sum(r["successes"] for r in suite_results)
        total_episodes = sum(r["episodes"] for r in suite_results)
        summary[suite] = {
            "successes": total_successes,
            "episodes": total_episodes,
            "success_rate": total_successes / total_episodes if total_episodes else 0.0,
        }

    payload = {
        "policy_path": policy_path,
        "image_input": "uniform_white_noise_[0,1]" if image_mode == "white_noise" else "real_camera_images",
        "image_mode": image_mode,
        "seed": seed,
        "episodes_per_task": episodes,
        "summary": summary,
        "results": all_results,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== Summary ===")
    for suite, item in summary.items():
        print(f"{suite}: {item['successes']}/{item['episodes']} success_rate={item['success_rate']:.3f}")
    print(out_path)


if __name__ == "__main__":
    main()
