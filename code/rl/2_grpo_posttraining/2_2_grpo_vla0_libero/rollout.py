"""before / after 确定性评测：GRPO 到底有没有把成功率提上去。

第15讲 8.6 节那张成功率对照图和那张表里的每个数字，都出自跑这个脚本；
同节末尾那条"必须看贪婪解码"的备注说的也是它。

训练日志里的 group_success 是带温度采样的探索成功率，数值偏低且波动大；
判断「学没学到」必须看贪婪解码的确定性评测。本脚本对基座和训后 checkpoint
在同一批初始状态上各跑一遍贪婪 rollout，打印逐状态与总体的成功率对照。
"""

import os
from pathlib import Path

import numpy as np
import torch

from lerobot.envs.factory import make_env_pre_post_processors
from lerobot.policies.factory import make_pre_post_processors

from env import build_group_env, rollout_group
from model import load_policy


def eval_policy(ckpt, task_id, eval_states, episodes_per_state):
    """贪婪解码逐初始状态评测。

    `temperature=None` 和 `max_steps_cap=0` 是这里与训练采样的两处关键区别：
    评测要确定性、也不能提前截断，否则测出来的不是策略的真实水平。

    Args:
        ckpt: 待评测的 checkpoint 目录。
        task_id: libero_object 套件内的任务序号。
        eval_states: 要评的初始状态编号列表。
        episodes_per_state: 每个初始状态跑几个回合。

    Returns:
        dict: {初始状态编号: 成功率}。逐状态看，才发现得了 8.4 节那种
        "这个状态在涨、别的状态在塌"的顾此失彼。
    """

    policy = load_policy(ckpt)
    policy.eval()
    env, cfg = build_group_env(task_id, episodes_per_state)
    pre, post = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=str(ckpt),
        preprocessor_overrides={"device_processor": {"device": "cuda"},
                                "rename_observations_processor": {"rename_map": {}}})
    env_pre, env_post = make_env_pre_post_processors(env_cfg=cfg, policy_cfg=policy.config)
    rates = {}
    for i, init_idx in enumerate(eval_states):
        with torch.no_grad():
            _, rewards = rollout_group(
                policy, env, pre, env_pre, post, env_post,
                seed=20000 + i, group_size=episodes_per_state, init_idx=init_idx,
                temperature=None, max_steps_cap=0, collect_records=False,
            )
        rates[init_idx] = float(np.mean(rewards))
    env.close()
    del policy
    torch.cuda.empty_cache()
    return rates


def main():
    """把弱基座和训后 checkpoint 摆在同一批初始状态上跑一遍，打印前后对照。"""

    # ---- 待对照的两个 checkpoint：SFT 弱基座 vs GRPO 训后（挑评测最高的迭代）----
    trained_root = Path(os.environ["DATASETS_ROOT"]) / "models" / "trained" / "xbotics_rl_grpo_vla0"
    base_ckpt = trained_root / "weak_base_2000"
    grpo_ckpt = trained_root / "grpo_runs" / "iter002"

    task_id = 0
    eval_states = [0, 1, 3, 6]   # 与训练用的前沿状态池一致，前后可比
    episodes_per_state = 3

    base_rates = eval_policy(base_ckpt, task_id, eval_states, episodes_per_state)
    grpo_rates = eval_policy(grpo_ckpt, task_id, eval_states, episodes_per_state)

    print(f"{'初始状态':>8} {'基座':>8} {'GRPO':>8}")
    for s in eval_states:
        print(f"{s:>8} {base_rates[s]:>8.2f} {grpo_rates[s]:>8.2f}")
    print(f"{'总体':>8} {np.mean(list(base_rates.values())):>8.2f} "
          f"{np.mean(list(grpo_rates.values())):>8.2f}")


if __name__ == "__main__":
    main()
