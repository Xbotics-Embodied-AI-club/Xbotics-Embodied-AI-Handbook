from __future__ import annotations

import torch
from torch import nn


def compute_gae(rewards, dones, values, next_value, gamma, lam):
    """广义优势估计（GAE）。

    rewards/dones 形状 (T, N)，values 形状 (T, N, 1)，next_value 形状 (N, 1)。
    从后往前递推：delta = r + gamma * V(s') * (1-done) - V(s)，
    advantage 累加 gamma*lam 折扣。returns = advantage + value。
    """
    rewards_3d = rewards.unsqueeze(-1) if rewards.ndim == 2 else rewards
    dones_3d = dones.unsqueeze(-1) if dones.ndim == 2 else dones
    values_3d = values.unsqueeze(-1) if values.ndim == 2 else values
    next_value_3d = next_value.unsqueeze(-1) if next_value.ndim == 1 else next_value
    advantages = torch.zeros_like(values_3d)
    last_advantage = torch.zeros_like(next_value_3d)
    for step in reversed(range(rewards_3d.shape[0])):
        next_values = next_value_3d if step == rewards_3d.shape[0] - 1 else values_3d[step + 1]
        not_done = 1.0 - dones_3d[step]
        delta = rewards_3d[step] + gamma * next_values * not_done - values_3d[step]
        last_advantage = delta + gamma * lam * not_done * last_advantage
        advantages[step] = last_advantage
    returns = advantages + values_3d
    return advantages.squeeze(-1), returns


class ActorCritic(nn.Module):
    """三个版本共用的模型：actor 输出高斯动作分布，critic 估状态价值。

    带在线 running normalization（observation 标准化），维度由外部按环境实际
    观测维度传入，不在模型里写死。
    """

    def __init__(
        self,
        obs_dim,
        critic_obs_dim,
        action_dim,
        actor_hidden_dims=(512, 256, 128),
        critic_hidden_dims=(512, 256, 128),
        init_noise_std=1.0,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.normalizer_epsilon = 1.0e-2
        self.register_buffer("actor_mean", torch.zeros(obs_dim))
        self.register_buffer("actor_var", torch.ones(obs_dim))
        self.register_buffer("actor_count", torch.tensor(0.0))
        self.register_buffer("critic_mean", torch.zeros(critic_obs_dim))
        self.register_buffer("critic_var", torch.ones(critic_obs_dim))
        self.register_buffer("critic_count", torch.tensor(0.0))
        self.actor = self.build_mlp(obs_dim, actor_hidden_dims, action_dim)
        self.critic = self.build_mlp(critic_obs_dim, critic_hidden_dims, 1)
        self.std = nn.Parameter(torch.full((action_dim,), init_noise_std))

    @staticmethod
    def build_mlp(input_dim, hidden_dims, output_dim):
        layers, last = [], input_dim
        for hidden in hidden_dims:
            layers += [nn.Linear(last, hidden), nn.ELU()]
            last = hidden
        layers.append(nn.Linear(last, output_dim))
        return nn.Sequential(*layers)

    @torch.no_grad()
    def update_actor_normalizer(self, obs):
        self.actor_mean, self.actor_var, self.actor_count = self._update_stats(
            obs, self.actor_mean, self.actor_var, self.actor_count
        )

    @torch.no_grad()
    def update_critic_normalizer(self, obs):
        self.critic_mean, self.critic_var, self.critic_count = self._update_stats(
            obs, self.critic_mean, self.critic_var, self.critic_count
        )

    def _update_stats(self, x, mean, var, count):
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = torch.tensor(float(x.shape[0]), device=x.device)
        new_count = count + batch_count
        rate = batch_count / new_count
        delta = batch_mean - mean
        new_mean = mean + rate * delta
        new_var = var + rate * (batch_var - var + delta * (batch_mean - new_mean))
        return new_mean, new_var, new_count

    def normalize_actor(self, obs):
        return (obs - self.actor_mean) / (torch.sqrt(self.actor_var) + self.normalizer_epsilon)

    def normalize_critic(self, obs):
        return (obs - self.critic_mean) / (torch.sqrt(self.critic_var) + self.normalizer_epsilon)

    def action_distribution(self, obs):
        mean = self.actor(self.normalize_actor(obs))
        std = self.std.clamp_min(1.0e-6).expand_as(mean)
        return torch.distributions.Normal(mean, std)

    def value(self, critic_obs):
        return self.critic(self.normalize_critic(critic_obs))

    def act(self, obs, critic_obs):
        dist = self.action_distribution(obs)
        actions = dist.sample()
        log_prob = dist.log_prob(actions).sum(dim=-1)
        return actions, log_prob, self.value(critic_obs), dist.mean, dist.stddev

    def evaluate(self, obs, critic_obs, actions):
        dist = self.action_distribution(obs)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, self.value(critic_obs), dist.mean

    def act_inference(self, obs):
        return self.actor(self.normalize_actor(obs))
