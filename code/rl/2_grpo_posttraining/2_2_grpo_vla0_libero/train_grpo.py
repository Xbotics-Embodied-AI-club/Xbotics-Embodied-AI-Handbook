"""GRPO 后训练 VLA-0：任务成败当 0/1 奖励，组相对优势 + KL-to-ref。

和 2_1 让 VLM 学数数是同一套 GRPO 外壳，只是「文本 token」换成了「动作 token」：
在 LIBERO 里同一初始状态采一组 rollout，组内按成败算相对优势（不需要 critic），
再用 KL-to-ref 把策略锚在 SFT 基座附近——没有这个信赖域，稀疏 0/1 奖励的
纯 REINFORCE 会把策略从基座的胜任区拖走（成功率不升反降）。
"""

import os
import shutil
from pathlib import Path

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

from lerobot.envs.factory import make_env_pre_post_processors
from lerobot.policies.factory import make_pre_post_processors

from env import build_group_env, rollout_group
from model import load_policy, sequence_logprob_tok


class GRPORolloutDataset(IterableDataset):
    """在线采样数据集：每个样本 = 一次迭代的全部组 rollout。

    每次迭代把前沿状态池扫一遍（每个初始状态采一组），组内算相对优势后
    展平成 (records, advantages) 交给训练模块。策略更新后下一次迭代自动
    用新策略采样——on-policy 循环由 dataloader 的迭代天然驱动。
    """

    def __init__(self, policy, envs, pre, env_pre, post, env_post,
                 state_pool, group_size, temperature, max_steps_cap, seed0):
        self.policy = policy
        self.envs = envs
        self.pre, self.env_pre, self.post, self.env_post = pre, env_pre, post, env_post
        self.state_pool = state_pool
        self.group_size = group_size
        self.temperature = temperature
        self.max_steps_cap = max_steps_cap
        self.seed0 = seed0

    def __iter__(self):
        it = 0
        while True:
            records, advantages, rewards_log = [], [], []
            mixed_groups, n_groups = 0, 0
            for task_id, (env, _) in self.envs.items():
                for s, init_idx in enumerate(self.state_pool):
                    grp_records, grp_rewards = rollout_group(
                        self.policy, env, self.pre, self.env_pre, self.post, self.env_post,
                        seed=self.seed0 + it * 1000 + task_id * 100 + s,
                        group_size=self.group_size, init_idx=init_idx,
                        temperature=self.temperature, max_steps_cap=self.max_steps_cap,
                    )
                    rewards_log += grp_rewards
                    n_groups += 1
                    R = np.array(grp_rewards)
                    if 0.0 < R.mean() < 1.0:
                        mixed_groups += 1   # 组内有成有败，才有非零优势信号
                    adv = (R - R.mean()) / (R.std() + 1e-6)
                    for recs, a in zip(grp_records, adv):
                        for rec in recs:
                            records.append(rec)
                            advantages.append(float(a))
            yield {
                "records": records,
                "advantages": advantages,
                "group_success": float(np.mean(rewards_log)),
                "mixed_groups": mixed_groups,
                "n_groups": n_groups,
            }
            it += 1


class GRPOData(L.LightningDataModule):
    """把在线采样数据集包成 dataloader（batch 已由数据集组好，batch_size=None 原样透传）。"""

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=None)


class VLA0LightningGRPO(L.LightningModule):
    """GRPO 更新：loss = -优势×logprob + β·KL(π‖π_ref)。

    每次迭代的样本数不固定（只有优势非零的生成才有梯度），所以用手动优化：
    逐条记录复算 logprob、累积梯度，最后统一裁剪并 step。
    """

    def __init__(self, policy, ref_model, lr, beta_kl, save_every, save_root, base_ckpt):
        super().__init__()
        self.automatic_optimization = False
        self.policy = policy
        self.ref_model = ref_model
        self.lr = lr
        self.beta_kl = beta_kl
        self.save_every = save_every
        self.save_root = Path(save_root)
        self.base_ckpt = Path(base_ckpt)

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()
        active = [(r, a) for r, a in zip(batch["records"], batch["advantages"]) if abs(a) >= 1e-8]
        gnorm = 0.0
        if active:
            self.policy.train()
            for rec, adv in active:
                lp_tok, valid = sequence_logprob_tok(self.policy.model, rec, requires_grad=True)
                denom = valid.sum().clamp(min=1)
                lp = (lp_tok * valid).sum() / denom
                loss = -(adv * lp)
                # KL-to-ref（DeepSeek GRPO 的 k3 估计器，逐 token ≥ 0）：把策略锚在冻结基座附近。
                lp_ref, _ = sequence_logprob_tok(self.ref_model, rec, requires_grad=False)
                delta = lp_ref - lp_tok
                kl_tok = torch.exp(delta) - delta - 1.0
                loss = loss + self.beta_kl * (kl_tok * valid).sum() / denom
                self.manual_backward(loss / len(active))
            gnorm = float(torch.nn.utils.clip_grad_norm_(self.policy.model.parameters(), 1.0))
            opt.step()
            self.policy.eval()
        print(f"[iter {batch_idx}] group_success={batch['group_success']:.3f} "
              f"mixed_groups={batch['mixed_groups']}/{batch['n_groups']} "
              f"n_active_gen={len(active)} gnorm={gnorm:.3f}", flush=True)
        # 密集存档：成功率峰值来得早、之后会退化，训完后按确定性评测挑峰值 checkpoint。
        if (batch_idx + 1) % self.save_every == 0:
            outdir = self.save_root / f"iter{batch_idx + 1:03d}"
            self.policy.save_pretrained(outdir)
            for f in self.base_ckpt.glob("policy_*"):
                shutil.copy(f, outdir)
            print(f"  saved {outdir}", flush=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.policy.model.parameters(), lr=self.lr)


def main():
    torch.manual_seed(0)

    # ---- 路径：弱基座与输出都放共享训练产物根下 ----
    trained_root = Path(os.environ["DATASETS_ROOT"]) / "models" / "trained" / "xbotics_rl_grpo_vla0"
    base_ckpt = trained_root / "weak_base_2000"   # 少步 SFT 的弱基座（复现方式见 README）
    save_root = trained_root / "grpo_runs"

    # ---- 超参：出涨点的配方，含义见各行注释 ----
    task_ids = [0]              # libero_object 的第 0 个任务
    state_pool = [0, 1, 3, 6]   # 前沿初始状态：基座成功率非 0 非 1，组内才有混合成败
    group_size = 8              # 每个初始状态采 8 条 rollout 组成一组
    temperature = 1.0           # 训练采样温度：1.0 即策略自然分布；调高反而压掉信号
    max_steps_cap = 220         # 截断超长 episode，省掉失败轨迹的尾巴
    lr = 1e-5
    beta_kl = 2.0               # KL-to-ref 强度：消融显示没有它策略会退化
    n_iters = 10                # 迭代数不必多：峰值常在前几次更新，靠评测挑 checkpoint
    save_every = 2

    policy = load_policy(base_ckpt)
    policy.eval()
    ref_model = load_policy(base_ckpt).model   # 冻结基座：KL 的参考分布
    for p in ref_model.parameters():
        p.requires_grad_(False)
    ref_model.eval()

    envs = {tid: build_group_env(tid, group_size) for tid in task_ids}
    any_cfg = next(iter(envs.values()))[1]
    pre, post = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=str(base_ckpt),
        preprocessor_overrides={"device_processor": {"device": "cuda"},
                                "rename_observations_processor": {"rename_map": {}}})
    env_pre, env_post = make_env_pre_post_processors(env_cfg=any_cfg, policy_cfg=policy.config)

    dataset = GRPORolloutDataset(policy, envs, pre, env_pre, post, env_post,
                                 state_pool, group_size, temperature, max_steps_cap, seed0=10000)
    data = GRPOData(dataset)
    model = VLA0LightningGRPO(policy, ref_model, lr, beta_kl, save_every, save_root, base_ckpt)

    trainer = L.Trainer(accelerator="gpu", devices=1, max_steps=n_iters,
                        enable_checkpointing=False, logger=False, enable_progress_bar=False)
    trainer.fit(model, data)


if __name__ == "__main__":
    main()
