"""VLA-0 在 LIBERO 上闭环推理一次，并把第一个动作块的原始数字串打印出来。

第11讲 6.5 节的配套代码。"动作就是一串数字"这句话在这里可以直接看见：模型的全部输出就是
一行空格分隔的整数，解码不过是按空格切开、再查一张 bin 中心表。
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import xgrammar as xgr

from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import add_envs_task, preprocess_observation
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.vla0_smol.modeling_vla0_smol import VLA0SmolPolicy
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.io_utils import write_video

# MuJoCo 的离屏渲染需要 EGL。
os.environ["MUJOCO_GL"] = "egl"

# 关闭 torch compile / inductor：首次运行的 autotune 开销很大，而这里只跑一局，
# 编译省下的时间远不够抵掉它，还会让两次运行的结果对不上。
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

# 下面这组参数是已经验证过能跑出 success=True 的固定配置。
# VLA-0：0.5B SmolVLM2 底座，动作就是一串数字 token（每个整数是一个 bin 编号）。
# checkpoint 用讲 15 GRPO 后训练挑出的峰值权重（LIBERO task0 成功率 60-67%）。
CKPT_PATH = Path(os.environ["DATASETS_ROOT"]) / "models" / "trained" / "xbotics_rl_grpo_vla0" / "grpo_runs" / "iter002"
TASK_SUITE = "libero_object"
TASK_ID = 0
EPISODE_INDEX = 0
MAX_STEPS = 280
FPS = 10
SEED = 7
OUT_PATH = Path("vla/4_vla_inference/4_3_vla0_infer/output/vla0_libero_success.mp4")


def set_episode_index(env, episode_index: int) -> None:
    """把 LIBERO 环境切到指定编号的初始状态。

    LeRobot 的 LIBERO 向量环境外面包了一层 SyncVectorEnv，真正决定物体初始摆放的是里面
    每个子环境的 episode_index / init_state_id，改外层那一层没有用。

    Args:
        env: make_env 返回的向量环境，这里只跑 1 个子环境。
        episode_index: 初始状态编号，演示固定用已经验证过能跑成功的那一局。
    """
    for inner_env in env.envs:
        inner_env.episode_index = episode_index
        inner_env.init_state_id = episode_index


def generate_digit_string(m, batch) -> tuple[str, torch.Tensor]:
    """让 VLA-0 生成一个动作块，把模型吐出的原始数字串和解码后的连续动作一起返回。

    这是 rl/2_grpo_posttraining/2_2 里 sample_chunk + decode_actions 的单环境贪婪版：
    1. 观测图 + 状态 + 指令拼成 VLM 输入；
    2. xgrammar 约束解码，保证模型只可能输出数字串；
    3. 数字串按空格切开：horizon×action_dim 个整数，每个整数是一个 bin 编号，
       查 bin 中心表映射回 [-1, 1] 的连续动作。

    Args:
        m: VLA0SmolPolicy 内部的模型本体，带 processor、compiled_grammar 与 bin 配置。
        batch: 已经过预处理管线的一帧观测，含图像、OBS_STATE 与 task 文本。

    Returns:
        (数字串原文, 连续动作块) 二元组。动作块形状 (1, action_horizon, action_dim)；
        若配置里 relative_actions 为真，块已加回当前状态、变成绝对目标。
    """
    images = m.prepare_images(batch)
    padded, _ = m.create_input_tokens(states=batch[OBS_STATE], images=images,
                                      lang_text=batch.get("task", ""), actions=None)
    input_len = padded["input_ids"].shape[1]
    proc = xgr.contrib.hf.LogitsProcessor(m.compiled_grammar)
    out = m.vlm.generate(
        input_ids=padded["input_ids"], attention_mask=padded["attention_mask"],
        pixel_values=padded["pixel_values"], pixel_attention_mask=padded["pixel_attention_mask"],
        use_cache=True, max_new_tokens=m.config.max_decoding_steps,
        num_beams=1, do_sample=False,  # 贪婪解码：推理要确定性
        eos_token_id=m.eos_token_id, pad_token_id=m.pad_token_id,
        logits_processor=[proc], return_dict_in_generate=True,
    )
    gen = out.sequences[:, input_len:]

    # 把生成 token 还原成文本，就是「动作即数字串」的原始样子。
    valid = (gen != m.eos_token_id) & (gen != m.pad_token_id)
    toks = torch.where(valid, gen, torch.tensor(m.pad_token_id, device=gen.device))
    text = m.processor.batch_decode(toks, skip_special_tokens=True)[0].strip()

    # 数字串 → bin 编号 → bin 中心值（[-1,1] 连续动作）。
    n_expected = m.action_horizon * m.action_dim
    n_bins = m.config.n_state_bins
    digits = text.split()
    assert len(digits) == n_expected, f"数字串长度 {len(digits)} != {n_expected}"
    disc = torch.tensor([int(x) for x in digits], device=gen.device).reshape(1, -1, m.action_dim)
    eps = 1e-6
    bins = torch.linspace(-1.0 - eps, 1.0 + eps, n_bins + 1, device=gen.device)
    centers = 0.5 * (bins[:-1] + bins[1:])
    chunk = centers[disc.clamp(0, n_bins - 1)]
    if m.config.relative_actions:
        chunk = chunk + batch[OBS_STATE].unsqueeze(1)
    return text, chunk


def main() -> None:
    """加载策略、建好 LIBERO 环境、闭环跑完一局，把整段 rollout 写成 mp4。

    跑不出成功就直接抛异常，不会静悄悄留下一段失败录像——一次演示要么给出成功的结果，
    要么明确报错，不能只是"看起来运行过了"。
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 从本地训练产物目录加载（讲 15 GRPO 实验的产物，非 HF Hub 模型）。
    policy = VLA0SmolPolicy.from_pretrained(str(CKPT_PATH)).to("cuda")
    policy.eval()
    m = policy.model
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=str(CKPT_PATH),
        preprocessor_overrides={"device_processor": {"device": "cuda"},
                                "rename_observations_processor": {"rename_map": {}}})

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
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg, policy_cfg=policy.config)

    frames = []
    success = False
    first_chunk_printed = False

    try:
        policy.reset()
        set_episode_index(env, EPISODE_INDEX)
        observation, _ = env.reset(seed=[SEED + EPISODE_INDEX])

        queue: list[torch.Tensor] = []
        for _ in range(MAX_STEPS):
            if not queue:
                # 动作块用完了：走一遍预处理管线，让 VLA-0 生成下一串数字。
                observation_batch = preprocess_observation(observation)
                observation_batch = add_envs_task(env, observation_batch)
                observation_batch = env_preprocessor(observation_batch)
                observation_batch = preprocessor(observation_batch)
                with torch.inference_mode():
                    text, chunk = generate_digit_string(m, observation_batch)
                if not first_chunk_printed:
                    # 展示一次「动作即数字串」：这就是 VLA-0 的全部输出。
                    print(f"数字串（{m.action_horizon}步×{m.action_dim}维）: {text}")
                    first_chunk_printed = True
                queue = list(chunk.transpose(0, 1))

            action = queue.pop(0)
            action = postprocessor(action)
            action = env_postprocessor({ACTION: action})[ACTION].cpu().numpy()

            observation, _, terminated, truncated, info = env.step(action)
            frames.append(env.envs[0].render())

            if "final_info" in info and isinstance(info["final_info"], dict):
                success = bool(info["final_info"]["is_success"][0])

            if bool(terminated[0]) or bool(truncated[0]):
                break

        # 这两个检查保证 demo 不是“看起来运行了”，而是真的有结果、而且真的成功。
        if not frames:
            raise RuntimeError("no frames")
        if not success:
            raise RuntimeError("no success")

        write_video(str(OUT_PATH), frames, fps=FPS)
        print(OUT_PATH)
    finally:
        # 关闭环境，避免 MuJoCo / EGL 资源泄漏。
        env.close()


if __name__ == "__main__":
    main()
