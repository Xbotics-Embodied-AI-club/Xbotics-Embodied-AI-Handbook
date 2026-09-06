"""录一段**跑满全过程**的策略 rollout —— 包含放手入箱之后的回家段。

`lerobot-eval` 录的像在成功那一刻就断了：环境把 `success` 并进 `terminated`
（`lerobot_env.py:339`），而这三个任务的成功判据是"物体进箱"。于是录像永远停在
"端着物体停在料箱上方"，示教的第五段「回家」（`expert.py:24`）从没被拍到过。

这里不改环境：`auto_reset=False` 时环境不会自己复位（那条路径本来就是给
`lerobot-record` 用的），本脚本的循环忽略 `terminated` 继续走，就拿到整条轨迹。

★ 每步的夹爪动作一并存进 `<场景>.tsv`。连拍图要按**夹爪开合**定位关键时刻，
  硬编帧号换一集就指向别处 —— 本项目在"取错时刻"上栽过四次。

用法：python record_full_rollout.py <checkpoint 目录> <输出目录> [步数] [场景 id ...]
      不给场景就录全部三个。
"""

import sys
from pathlib import Path

import numpy as np
import torch

TASKS = [
    "SO101PickPlaceCube40-v1",
    "SO101PickPlaceCube20-v1",
    "SO101PickPlaceCylinder40-v1",
]
# 相机键名映射（数据集相机名 → 策略输入名）。**只有 SmolVLA 需要**：它的预训练权重
# 认的是 camera1/camera2。ACT 从零训、pi0 用自己的命名，都直接用数据集里的 top/wrist，
# 传了反而会把键改错。所以按策略类型决定给不给。
RENAME_SMOLVLA = {
    "observation.images.top": "observation.images.camera1",
    "observation.images.wrist": "observation.images.camera2",
}
SEED = int(__import__("os").environ.get("ROLL_SEED", "0"))


def record(policy, ckpt: Path, task: str, steps: int, out_dir: Path) -> None:
    """跑一集并存 mp4 与逐步夹爪动作。

    Args:
        policy: 已加载到 GPU 的策略。
        ckpt: checkpoint 目录，pre/post 管线要从里面读归一化统计量。
        task: 环境 id。
        steps: 固定走多少步（忽略 terminated）。
        out_dir: 输出目录。
    """
    import imageio
    from lerobot.envs.utils import preprocess_observation
    from lerobot.policies.factory import make_pre_post_processors
    from so101_sim.lerobot_env import So101SimEnv

    env = So101SimEnv(
        task=task,
        task_description=None,
        control_mode="pd_joint_pos",
        observation_width=640,
        observation_height=480,
        episode_length=steps + 10,
        auto_reset=False,          # ★ 成功后不复位，循环自己决定走多久
        render_mode="rgb_array",
    )
    # 走与 `lerobot-eval` 逐字相同的 pre/post 管线（lerobot_eval.py:176）。手写一份
    # 会漏掉分词、归一化、设备搬运里的任何一步，而漏了**不报错**，只表现为成功率低。
    rename = RENAME_SMOLVLA if policy.config.type == "smolvla" else {}
    pre, post = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(ckpt),
        preprocessor_overrides={"device_processor": {"device": str(policy.config.device)},
                                "rename_observations_processor": {"rename_map": rename}},
    )
    # ★ 环境和**策略**都要定住。`env.reset(seed=…)` 只固定物体摆放；而 flow-matching
    #   采样动作用的是全局 torch 随机数，它的状态取决于此前抽过多少次 ——
    #   于是同一个种子在不同的场景顺序下给出不同轨迹。实测：cube20 单独跑时成功，
    #   排在 cube40 之后跑就失败，同一份权重同一个种子。不定住它，这个脚本的输出不可复现。
    torch.manual_seed(SEED)
    obs, _ = env.reset(seed=SEED)
    policy.reset()

    frames, grips, first_success = [], [], None
    for t in range(steps):
        batched = {k: (v[None] if isinstance(v, np.ndarray) else
                       {kk: vv[None] for kk, vv in v.items()}) for k, v in obs.items()}
        feed = preprocess_observation(batched)
        feed["task"] = [env.task_description]
        with torch.inference_mode():
            action = post(policy.select_action(pre(feed)))
        act = action.squeeze(0).cpu().numpy()
        obs, _, terminated, _, _ = env.step(act)
        frames.append(env.render())
        grips.append(float(act[-1]))
        if terminated and first_success is None:
            first_success = t
    env.close()

    out_dir.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_dir / f"{task}.mp4", frames, fps=30, macro_block_size=1)
    (out_dir / f"{task}.tsv").write_text(
        "\n".join(f"{i}\t{g:.4f}" for i, g in enumerate(grips)) + "\n")
    print(f"  {task}: {len(frames)} 帧，种子 {SEED}，首次判成功在第 {first_success} 步")


def main(argv) -> int:
    if len(argv) < 2:
        sys.exit(__doc__)
    ckpt, out_dir = Path(argv[0]), Path(argv[1])
    steps = int(argv[2]) if len(argv) >= 3 else 300
    tasks = argv[3:] or TASKS

    # 按 checkpoint 自报的 type 取策略类，别写死某一个 —— 这个脚本要同时服务
    # SmolVLA / ACT / pi0 三条线，写死就得复制三份，改一处忘两处只是时间问题。
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class

    cfg = PreTrainedConfig.from_pretrained(ckpt)
    cls = get_policy_class(cfg.type)
    if (ckpt / "adapter_model.safetensors").is_file():
        # LoRA 产物落的是适配器，不是整套权重：要先读 PeftConfig 找到基座，再把旁路挂上去。
        # 直接 from_pretrained(适配器目录) 会因为没有 model.safetensors 而失败。
        from peft import PeftConfig, PeftModel
        pc = PeftConfig.from_pretrained(str(ckpt))
        base = cls.from_pretrained(pc.base_model_name_or_path)
        policy = PeftModel.from_pretrained(base, str(ckpt)).to("cuda").eval()
        # PeftModel 把策略包了一层，而下面要用 policy.config / .reset / .select_action，
        # 它们都在被包的那个对象上 —— 取出来直接用，别隔着包装层调。
        policy = policy.merge_and_unload().to("cuda").eval()
        print(f"  策略类型 {cfg.type}（适配器已合并进基座）")
    else:
        policy = cls.from_pretrained(ckpt).to("cuda").eval()
        print(f"  策略类型 {cfg.type}")
    for task in tasks:
        record(policy, ckpt, task, steps, out_dir)
    print("RECORD_FULL_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
