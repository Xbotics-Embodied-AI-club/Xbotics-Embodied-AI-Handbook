"""第一版：REINFORCE（最朴素的策略梯度）。

只有一个 actor，没有 critic：
  - 优势直接用“折扣回报（reward-to-go）减去整批均值”这个常数基线，不学价值函数；
  - 没有 GAE、没有重要性采样裁剪、没有数据复用，一段数据只用一遍。
方差很大，面对 G1 行走这种较难的任务通常学得慢、走不稳，
正好用来对比后面 v2/v3 加入 critic、GAE、clip 之后的提升。
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
from model import ActorCritic  # noqa: E402


def default_checkpoint_root(run_name: str) -> Path:
    datasets_root = Path(os.environ["DATASETS_ROOT"])
    return datasets_root / "models" / "trained" / "xbotics_rl_g1_walk" / run_name


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


def reward_to_go(reward_steps, done_steps, gamma):
    """从后往前累加折扣回报；遇到 done 截断（不跨 episode）。"""
    returns = torch.zeros_like(reward_steps)
    running = torch.zeros(reward_steps.shape[1], device=reward_steps.device)
    for step in reversed(range(reward_steps.shape[0])):
        running = reward_steps[step] + gamma * running * (1.0 - done_steps[step])
        returns[step] = running
    return returns


class G1WalkRolloutDataset(IterableDataset):
    """采一段轨迹，用 reward-to-go 减常数基线当优势；整段只产出一个 batch。"""

    def __init__(self, env, model, num_steps_per_env, gamma):
        super().__init__()
        self.env = env
        self.model = model
        self.num_steps_per_env = num_steps_per_env
        self.gamma = gamma

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        yield self.sample_rollout()

    def sample_rollout(self):
        env = self.env
        num_envs = env.num_envs
        obs, critic_obs = env.get_observations()
        device = env.device
        obs_steps = torch.zeros(self.num_steps_per_env, num_envs, obs.shape[1], device=device)
        actions_steps = torch.zeros(self.num_steps_per_env, num_envs, self.model.action_dim, device=device)
        reward_steps = torch.zeros(self.num_steps_per_env, num_envs, device=device)
        done_steps = torch.zeros(self.num_steps_per_env, num_envs, device=device)
        reward_sum = 0.0

        for step in range(self.num_steps_per_env):
            with torch.no_grad():
                actions, _log_probs, _values, _means, _stds = self.model.act(obs, critic_obs)
            next_obs, next_critic_obs, rewards, dones, _info = env.step(actions)

            obs_steps[step].copy_(obs)
            actions_steps[step].copy_(actions)
            reward_steps[step].copy_(rewards)
            done_steps[step].copy_(dones.float())

            self.model.update_actor_normalizer(next_obs)
            obs, critic_obs = next_obs, next_critic_obs
            reward_sum += float(rewards.mean().detach().cpu())

        returns = reward_to_go(reward_steps, done_steps, self.gamma)
        # 唯一的基线就是整批均值（常数），没有学习的 critic。
        advantages = returns - returns.mean()
        advantages = advantages / (advantages.std() + 1.0e-8)

        batch_size = self.num_steps_per_env * num_envs
        return {
            "obs": obs_steps.reshape(batch_size, -1),
            "actions": actions_steps.reshape(batch_size, -1),
            "advantages": advantages.reshape(batch_size),
            "reward_mean": torch.tensor(reward_sum / self.num_steps_per_env, device=device),
        }


class G1WalkData(L.LightningDataModule):
    def __init__(self, env, model, num_steps_per_env, gamma):
        super().__init__()
        self.env = env
        self.model = model
        self.num_steps_per_env = num_steps_per_env
        self.gamma = gamma

    def train_dataloader(self):
        dataset = G1WalkRolloutDataset(self.env, self.model, self.num_steps_per_env, self.gamma)
        return DataLoader(dataset, batch_size=None)


class G1WalkLightningReinforce(L.LightningModule):
    """REINFORCE：loss = -(logπ(a|s) · advantage)，只更新 actor。"""

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

        self.entropy_coef = 0.01
        self.learning_rate = 1.0e-3

    def setup(self, stage):
        if stage != "fit":
            return
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.wandb_run = wandb.init(
            project=self.wandb_project, name=self.run_name, mode=self.wandb_mode,
            dir=self.checkpoint_dir.as_posix(), config={**self.training_settings, "algo": "v1_reinforce"},
        )

    def configure_optimizers(self):
        # 只优化 actor 和动作标准差；critic 不参与训练。
        params = list(self.model.actor.parameters()) + [self.model.std]
        self.optimizer = torch.optim.Adam(params, lr=self.learning_rate)
        return self.optimizer

    def training_step(self, batch, batch_idx):
        distribution = self.model.action_distribution(batch["obs"])
        log_probs = distribution.log_prob(batch["actions"]).sum(dim=-1)
        entropy = distribution.entropy().sum(dim=-1)
        policy_loss = -(log_probs * batch["advantages"]).mean()
        entropy_loss = entropy.mean()
        loss = policy_loss - self.entropy_coef * entropy_loss

        iteration = self.current_epoch + 1
        metrics = {
            "reward": float(batch["reward_mean"].detach().cpu()),
            "loss": float(loss.detach().cpu()),
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


def run_training(run_name, num_envs, max_iterations, num_steps_per_env, save_interval, device,
                 seed=1, checkpoint_dir=None, wandb_project="rl_class", wandb_mode="online"):
    checkpoint_dir = checkpoint_dir or default_checkpoint_root(run_name)
    gamma = 0.99

    torch.manual_seed(seed)
    env = G1WalkEnv(num_envs=num_envs, device=device, seed=seed)
    env.reset()
    policy = ActorCritic(obs_dim=env.obs_dim, critic_obs_dim=env.critic_obs_dim, action_dim=env.action_dim)
    policy.to(env.device)

    training_settings = {
        "run_name": run_name, "num_envs": num_envs, "max_iterations": max_iterations,
        "num_steps_per_env": num_steps_per_env, "save_interval": save_interval, "device": device,
        "seed": seed, "checkpoint_dir": str(checkpoint_dir), "wandb_project": wandb_project,
        "wandb_mode": wandb_mode, "gamma": gamma,
        "obs_dim": env.obs_dim, "critic_obs_dim": env.critic_obs_dim, "action_dim": env.action_dim,
    }

    data = G1WalkData(env, policy, num_steps_per_env, gamma)
    model = G1WalkLightningReinforce(policy, run_name, max_iterations, save_interval, checkpoint_dir,
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
        run_name="g1-walk-reinforce",
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
