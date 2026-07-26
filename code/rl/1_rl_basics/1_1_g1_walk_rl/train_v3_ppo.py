"""第三版：完整 PPO。

相比 v2 多了三件让训练更稳的关键机制：
  1. 重要性采样比值的裁剪（clip），允许同一段数据反复用多轮而不跑偏；
  2. 一段 rollout 切成多个 minibatch、训多个 epoch，样本效率更高；
  3. 按 KL 散度自适应调整学习率，更新步长自动收放。
这就是 BeyondMimic 用的那套核心，直接套到 G1 行走任务上。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterator

import lightning as L
import torch
import wandb
from torch.utils.data import DataLoader, IterableDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env import G1WalkEnv  # noqa: E402
from model import ActorCritic, compute_gae  # noqa: E402


def default_checkpoint_root(run_name: str) -> Path:
    datasets_root = Path(os.environ["DATASETS_ROOT"])
    return datasets_root / "models" / "trained" / "xbotics_rl_g1_walk" / run_name


def adapt_learning_rate_to_kl(optimizer, kl, desired_kl, min_lr=1.0e-5, max_lr=1.0e-2):
    current_lr = optimizer.param_groups[0]["lr"]
    kl_value = float(kl.detach().cpu())
    if kl_value > 2.0 * desired_kl:
        current_lr = max(min_lr, current_lr / 1.5)
    elif 0.0 < kl_value < 0.5 * desired_kl:
        current_lr = min(max_lr, current_lr * 1.5)
    optimizer.param_groups[0]["lr"] = current_lr
    return current_lr


def save_checkpoint(path, model, optimizer, iteration, training_settings):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "iteration": iteration,
            "actor_critic": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "training_settings": training_settings,
        },
        path,
    )


class G1WalkRolloutDataset(IterableDataset):
    """在线数据集：每轮先采一段轨迹，再切成若干 minibatch 逐个交给训练循环。"""

    def __init__(self, env, model, num_steps_per_env, gamma, lam, num_learning_epochs, num_mini_batches):
        super().__init__()
        self.env = env
        self.model = model
        self.num_steps_per_env = num_steps_per_env
        self.gamma = gamma
        self.lam = lam
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        rollout = self.sample_rollout()
        yield from self.iter_mini_batches(rollout)

    def sample_rollout(self):
        env = self.env
        num_envs = env.num_envs
        obs, critic_obs = env.get_observations()
        device = env.device
        obs_steps = torch.zeros(self.num_steps_per_env, num_envs, obs.shape[1], device=device)
        critic_obs_steps = torch.zeros(self.num_steps_per_env, num_envs, critic_obs.shape[1], device=device)
        actions_steps = torch.zeros(self.num_steps_per_env, num_envs, self.model.action_dim, device=device)
        log_prob_steps = torch.zeros(self.num_steps_per_env, num_envs, device=device)
        value_steps = torch.zeros(self.num_steps_per_env, num_envs, 1, device=device)
        reward_steps = torch.zeros(self.num_steps_per_env, num_envs, device=device)
        done_steps = torch.zeros(self.num_steps_per_env, num_envs, device=device)
        action_mean_steps = torch.zeros(self.num_steps_per_env, num_envs, self.model.action_dim, device=device)
        action_std_steps = torch.zeros(self.num_steps_per_env, num_envs, self.model.action_dim, device=device)
        reward_sum = 0.0

        for step in range(self.num_steps_per_env):
            with torch.no_grad():
                actions, log_probs, values, action_means, action_stds = self.model.act(obs, critic_obs)
            next_obs, next_critic_obs, rewards, dones, info = env.step(actions)
            if "time_outs" in info:
                rewards = rewards + self.gamma * values.squeeze(-1) * info["time_outs"].float()

            obs_steps[step].copy_(obs)
            critic_obs_steps[step].copy_(critic_obs)
            actions_steps[step].copy_(actions)
            log_prob_steps[step].copy_(log_probs)
            value_steps[step].copy_(values)
            reward_steps[step].copy_(rewards)
            done_steps[step].copy_(dones.float())
            action_mean_steps[step].copy_(action_means)
            action_std_steps[step].copy_(action_stds)

            self.model.update_actor_normalizer(next_obs)
            self.model.update_critic_normalizer(next_critic_obs)
            obs, critic_obs = next_obs, next_critic_obs
            reward_sum += float(rewards.mean().detach().cpu())

        with torch.no_grad():
            next_value = self.model.value(critic_obs)
        advantages, returns = compute_gae(reward_steps, done_steps, value_steps, next_value, self.gamma, self.lam)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1.0e-8)

        return {
            "obs": obs_steps,
            "critic_obs": critic_obs_steps,
            "actions": actions_steps,
            "log_probs": log_prob_steps,
            "values": value_steps,
            "returns": returns,
            "advantages": advantages,
            "action_means": action_mean_steps,
            "action_stds": action_std_steps,
            "reward_mean": torch.tensor(reward_sum / self.num_steps_per_env, device=device),
        }

    def iter_mini_batches(self, rollout):
        batch_size = rollout["actions"].shape[0] * rollout["actions"].shape[1]
        mini_batch_size = batch_size // self.num_mini_batches
        usable_size = mini_batch_size * self.num_mini_batches
        reward_mean = rollout["reward_mean"]
        flat = {
            "obs": rollout["obs"].reshape(batch_size, -1),
            "critic_obs": rollout["critic_obs"].reshape(batch_size, -1),
            "actions": rollout["actions"].reshape(batch_size, -1),
            "log_probs": rollout["log_probs"].reshape(batch_size),
            "values": rollout["values"].reshape(batch_size, 1),
            "returns": rollout["returns"].reshape(batch_size, 1),
            "advantages": rollout["advantages"].reshape(batch_size),
            "action_means": rollout["action_means"].reshape(batch_size, -1),
            "action_stds": rollout["action_stds"].reshape(batch_size, -1),
        }
        # 同一段 rollout 重复多遍：这就是 PPO 的多轮 minibatch 更新。
        for _ in range(self.num_learning_epochs):
            indices = torch.randperm(usable_size, device=flat["actions"].device)
            for start in range(0, usable_size, mini_batch_size):
                selected = indices[start : start + mini_batch_size]
                mini_batch = {key: value[selected] for key, value in flat.items()}
                mini_batch["reward_mean"] = reward_mean
                yield mini_batch


class G1WalkData(L.LightningDataModule):
    """持有持久环境，每轮把用最新策略采到的新数据集交给 Trainer。"""

    def __init__(self, env, model, num_steps_per_env, gamma, lam, num_learning_epochs, num_mini_batches):
        super().__init__()
        self.env = env
        self.model = model
        self.num_steps_per_env = num_steps_per_env
        self.gamma = gamma
        self.lam = lam
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches

    def train_dataloader(self):
        dataset = G1WalkRolloutDataset(
            env=self.env,
            model=self.model,
            num_steps_per_env=self.num_steps_per_env,
            gamma=self.gamma,
            lam=self.lam,
            num_learning_epochs=self.num_learning_epochs,
            num_mini_batches=self.num_mini_batches,
        )
        return DataLoader(dataset, batch_size=None)


class G1WalkLightningPPO(L.LightningModule):
    """只负责一个 minibatch 的 PPO loss、记录与保存；更新循环交给 Lightning。"""

    def __init__(self, model, run_name, max_iterations, save_interval, checkpoint_dir,
                 training_settings, wandb_project, wandb_mode):
        super().__init__()
        self.model = model
        self.run_name = run_name
        self.max_iterations = max_iterations
        self.save_interval = save_interval
        self.checkpoint_dir = checkpoint_dir
        self.training_settings = training_settings
        self.wandb_project = wandb_project
        self.wandb_mode = wandb_mode
        self.latest_checkpoint = self.checkpoint_dir / "model_0.pt"
        self.wandb_run = None
        self.optimizer = None
        self.latest_kl = torch.zeros(())
        self.epoch_records: list[dict[str, float]] = []

        self.value_loss_coef = 1.0
        self.use_clipped_value_loss = True
        self.clip_param = 0.2
        self.entropy_coef = 0.01
        self.learning_rate = 1.0e-3
        self.desired_kl = 0.01

    def setup(self, stage):
        if stage != "fit":
            return
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.wandb_run = wandb.init(
            project=self.wandb_project,
            name=self.run_name,
            mode=self.wandb_mode,
            dir=self.checkpoint_dir.as_posix(),
            config={**self.training_settings, "algo": "v3_ppo"},
        )

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return self.optimizer

    def on_train_epoch_start(self):
        self.epoch_records = []

    def training_step(self, mini_batch, batch_idx):
        loss, policy_loss, value_loss, entropy_loss, kl = self.loss_for_batch(mini_batch)
        self.latest_kl = kl.detach()
        record = {
            "reward": float(mini_batch["reward_mean"].detach().cpu()),
            "loss": float(loss.detach().cpu()),
            "value_loss": float(value_loss.detach().cpu()),
            "policy_loss": float(policy_loss.detach().cpu()),
            "entropy": float(entropy_loss.detach().cpu()),
            "kl": float(kl.detach().cpu()),
        }
        self.epoch_records.append(record)
        self.log_dict(record, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def on_before_optimizer_step(self, optimizer):
        adapt_learning_rate_to_kl(self.optimizer, self.latest_kl, self.desired_kl)

    def on_train_epoch_end(self):
        iteration = self.current_epoch + 1
        metrics = {key: sum(r[key] for r in self.epoch_records) / len(self.epoch_records) for key in self.epoch_records[0]}
        metrics["lr"] = self.optimizer.param_groups[0]["lr"]
        wandb.log(metrics, step=iteration)
        if iteration % self.save_interval == 0 or iteration == self.max_iterations:
            self.latest_checkpoint = self.checkpoint_dir / f"model_{iteration}.pt"
            save_checkpoint(self.latest_checkpoint, self.model, self.optimizer, iteration, self.training_settings)

    def loss_for_batch(self, batch):
        log_probs, entropy, values, action_means = self.model.evaluate(
            batch["obs"], batch["critic_obs"], batch["actions"]
        )
        with torch.no_grad():
            action_stds = self.model.std.clamp_min(1.0e-6).expand_as(action_means)
            old_stds = batch["action_stds"].clamp_min(1.0e-6)
            kl = torch.sum(
                torch.log(action_stds / old_stds + 1.0e-5)
                + (torch.square(old_stds) + torch.square(batch["action_means"] - action_means))
                / (2.0 * torch.square(action_stds))
                - 0.5,
                dim=-1,
            ).mean()

        ratio = torch.exp(log_probs - batch["log_probs"])
        unclipped = ratio * batch["advantages"]
        clipped = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * batch["advantages"]
        policy_loss = -torch.min(unclipped, clipped).mean()
        value_loss = self.value_loss(values, batch["values"], batch["returns"])
        entropy_loss = entropy.mean()
        loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_loss
        return loss, policy_loss, value_loss, entropy_loss, kl

    def value_loss(self, values, old_values, returns):
        if not self.use_clipped_value_loss:
            return torch.square(values - returns).mean()
        value_clipped = old_values + (values - old_values).clamp(-self.clip_param, self.clip_param)
        return torch.max(torch.square(values - returns), torch.square(value_clipped - returns)).mean()

    def teardown(self, stage):
        if self.wandb_run is not None:
            self.wandb_run.finish()


def run_training(run_name, num_envs, max_iterations, num_steps_per_env, save_interval, device,
                 seed=1, checkpoint_dir=None, wandb_project="rl_class", wandb_mode="online"):
    checkpoint_dir = checkpoint_dir or default_checkpoint_root(run_name)
    gamma, lam = 0.99, 0.95
    num_learning_epochs, num_mini_batches = 5, 4

    torch.manual_seed(seed)
    env = G1WalkEnv(num_envs=num_envs, device=device, seed=seed)
    env.reset()
    policy = ActorCritic(obs_dim=env.obs_dim, critic_obs_dim=env.critic_obs_dim, action_dim=env.action_dim)
    policy.to(env.device)

    training_settings = {
        "run_name": run_name, "num_envs": num_envs, "max_iterations": max_iterations,
        "num_steps_per_env": num_steps_per_env, "save_interval": save_interval, "device": device,
        "seed": seed, "checkpoint_dir": str(checkpoint_dir), "wandb_project": wandb_project,
        "wandb_mode": wandb_mode, "gamma": gamma, "lam": lam,
        "num_learning_epochs": num_learning_epochs, "num_mini_batches": num_mini_batches,
        "obs_dim": env.obs_dim, "critic_obs_dim": env.critic_obs_dim, "action_dim": env.action_dim,
    }

    data = G1WalkData(env, policy, num_steps_per_env, gamma, lam, num_learning_epochs, num_mini_batches)
    model = G1WalkLightningPPO(policy, run_name, max_iterations, save_interval, checkpoint_dir,
                               training_settings, wandb_project, wandb_mode)
    trainer = L.Trainer(
        accelerator="gpu" if device != "cpu" and torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=max_iterations,
        reload_dataloaders_every_n_epochs=1,
        gradient_clip_val=1.0,
        enable_checkpointing=False,
        logger=False,
        enable_model_summary=False,
        enable_progress_bar=True,
        log_every_n_steps=1,
    )
    trainer.fit(model, data)
    return model.latest_checkpoint


def main():
    run_training(
        run_name="g1-walk-ppo",
        num_envs=4096,
        max_iterations=3000,
        num_steps_per_env=24,
        save_interval=200,
        device="cuda:0",
        wandb_project="rl_class",
        wandb_mode="online",
    )


if __name__ == "__main__":
    main()
