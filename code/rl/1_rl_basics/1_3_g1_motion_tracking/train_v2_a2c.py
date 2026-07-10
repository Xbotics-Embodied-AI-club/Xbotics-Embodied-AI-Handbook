"""动作跟随任务上的第二版：A2C（带基线的策略梯度）。

和 `train.py`（完整 PPO，本专题“原版”）同任务同网络，只是算法更朴素：
  - 比 REINFORCE 多了 critic 基线 + GAE 优势，方差更小；
  - 但仍然单 epoch、单 minibatch、不裁剪、固定学习率，不做数据复用。
用来对照：在动作跟随这种较难的任务上，它比 REINFORCE 好一些，
但不如完整 PPO 稳、上限低，凸显 clip / 多轮 minibatch / KL 自适应的价值。
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
from env import BeyondMimicEnv  # noqa: E402
from model import ActorCritic, compute_gae  # noqa: E402


def default_checkpoint_root(run_name: str) -> Path:
    datasets_root = Path(os.environ["DATASETS_ROOT"])
    return datasets_root / "models" / "trained" / "xbotics_rl_beyondmimic" / run_name


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


class TrackingRolloutDataset(IterableDataset):
    """采一段轨迹，用 critic 基线 + GAE 算优势，整段只产出一个 batch（用一遍）。"""

    def __init__(self, env, model, num_steps_per_env, gamma, lam):
        super().__init__()
        self.env = env
        self.model = model
        self.num_steps_per_env = num_steps_per_env
        self.gamma = gamma
        self.lam = lam

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        yield self.sample_rollout()

    def sample_rollout(self):
        env = self.env
        num_envs = env.num_envs
        obs, critic_obs = env.get_observations()
        device = env.device
        obs_steps = torch.zeros(self.num_steps_per_env, num_envs, obs.shape[1], device=device)
        critic_obs_steps = torch.zeros(self.num_steps_per_env, num_envs, critic_obs.shape[1], device=device)
        actions_steps = torch.zeros(self.num_steps_per_env, num_envs, self.model.action_dim, device=device)
        value_steps = torch.zeros(self.num_steps_per_env, num_envs, 1, device=device)
        reward_steps = torch.zeros(self.num_steps_per_env, num_envs, device=device)
        done_steps = torch.zeros(self.num_steps_per_env, num_envs, device=device)
        reward_sum = 0.0

        for step in range(self.num_steps_per_env):
            with torch.no_grad():
                actions, _log_probs, values, _means, _stds = self.model.act(obs, critic_obs)
            next_obs, next_critic_obs, rewards, dones, info = env.step(actions)
            if "time_outs" in info:
                rewards = rewards + self.gamma * values.squeeze(-1) * info["time_outs"].float()

            obs_steps[step].copy_(obs)
            critic_obs_steps[step].copy_(critic_obs)
            actions_steps[step].copy_(actions)
            value_steps[step].copy_(values)
            reward_steps[step].copy_(rewards)
            done_steps[step].copy_(dones.float())

            self.model.update_actor_normalizer(next_obs)
            self.model.update_critic_normalizer(next_critic_obs)
            obs, critic_obs = next_obs, next_critic_obs
            reward_sum += float(rewards.mean().detach().cpu())

        with torch.no_grad():
            next_value = self.model.value(critic_obs)
        advantages, returns = compute_gae(reward_steps, done_steps, value_steps, next_value, self.gamma, self.lam)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1.0e-8)

        batch_size = self.num_steps_per_env * num_envs
        return {
            "obs": obs_steps.reshape(batch_size, -1),
            "critic_obs": critic_obs_steps.reshape(batch_size, -1),
            "actions": actions_steps.reshape(batch_size, -1),
            "returns": returns.reshape(batch_size, 1),
            "advantages": advantages.reshape(batch_size),
            "reward_mean": torch.tensor(reward_sum / self.num_steps_per_env, device=device),
        }


class TrackingData(L.LightningDataModule):
    def __init__(self, env, model, num_steps_per_env, gamma, lam):
        super().__init__()
        self.env = env
        self.model = model
        self.num_steps_per_env = num_steps_per_env
        self.gamma = gamma
        self.lam = lam

    def train_dataloader(self):
        dataset = TrackingRolloutDataset(self.env, self.model, self.num_steps_per_env, self.gamma, self.lam)
        return DataLoader(dataset, batch_size=None)


class TrackingLightningA2C(L.LightningModule):
    """A2C：策略梯度用 GAE 优势，critic 回归 returns；不裁剪、不复用数据、固定 lr。"""

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

        self.value_loss_coef = 1.0
        self.entropy_coef = 0.01
        self.learning_rate = 1.0e-3

    def setup(self, stage):
        if stage != "fit":
            return
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.wandb_run = wandb.init(
            project=self.wandb_project, name=self.run_name, mode=self.wandb_mode,
            dir=self.checkpoint_dir.as_posix(), config={**self.training_settings, "algo": "v2_a2c"},
        )

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return self.optimizer

    def training_step(self, batch, batch_idx):
        log_probs, entropy, values, _means = self.model.evaluate(
            batch["obs"], batch["critic_obs"], batch["actions"]
        )
        policy_loss = -(log_probs * batch["advantages"]).mean()
        value_loss = torch.square(values - batch["returns"]).mean()
        entropy_loss = entropy.mean()
        loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_loss

        iteration = self.current_epoch + 1
        metrics = {
            "reward": float(batch["reward_mean"].detach().cpu()),
            "loss": float(loss.detach().cpu()),
            "value_loss": float(value_loss.detach().cpu()),
            "policy_loss": float(policy_loss.detach().cpu()),
            "entropy": float(entropy_loss.detach().cpu()),
        }
        self.log_dict(metrics, prog_bar=True, on_step=True, on_epoch=False)
        wandb.log(metrics, step=iteration)
        if iteration % self.save_interval == 0 or iteration == self.max_iterations:
            self.latest_checkpoint = self.checkpoint_dir / f"model_{iteration}.pt"
            save_checkpoint(self.latest_checkpoint, self.model, self.optimizer, iteration, self.training_settings)
        return loss

    def teardown(self, stage):
        if self.wandb_run is not None:
            self.wandb_run.finish()


def run_training(motion_file, run_name, num_envs, max_iterations, num_steps_per_env, save_interval,
                 device, seed=1, checkpoint_dir=None, wandb_project="rl_class", wandb_mode="online"):
    checkpoint_dir = checkpoint_dir or default_checkpoint_root(run_name)
    gamma, lam = 0.99, 0.95

    torch.manual_seed(seed)
    env = BeyondMimicEnv(motion_file, num_envs=num_envs, device=device, seed=seed)
    obs, critic_obs = env.reset()
    policy = ActorCritic(obs_dim=obs.shape[1], critic_obs_dim=critic_obs.shape[1], action_dim=env.action_dim)
    policy.to(env.device)

    training_settings = {
        "motion_file": str(motion_file), "run_name": run_name, "num_envs": num_envs,
        "max_iterations": max_iterations, "num_steps_per_env": num_steps_per_env,
        "save_interval": save_interval, "device": device, "seed": seed,
        "checkpoint_dir": str(checkpoint_dir), "wandb_project": wandb_project, "wandb_mode": wandb_mode,
        "gamma": gamma, "lam": lam,
        "obs_dim": obs.shape[1], "critic_obs_dim": critic_obs.shape[1], "action_dim": env.action_dim,
    }

    data = TrackingData(env, policy, num_steps_per_env, gamma, lam)
    model = TrackingLightningA2C(policy, run_name, max_iterations, save_interval, checkpoint_dir,
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
    group_root = Path(__file__).resolve().parents[1]
    run_training(
        motion_file=group_root / "data/g1_reference_motions/marshal-arts.npz",
        run_name="beyondmimic-a2c",
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
