"""第四版：SAC（off-policy 的 Soft Actor-Critic）。

前三版 REINFORCE→A2C→PPO 都是 **on-policy**：每一轮都得用当前策略现采一段新数据，
算完梯度这段数据就扔掉，样本利用率天生受限。这一版换成 **off-policy**，同一个 G1
行走任务、同一份奖励、同样的 Lightning 四件套，只把算法从 PPO 换成 SAC，让"在线采样"
和"经验回放"两条路线正面对照。

和 v3 相比，差异集中在三件事——这就是本节的知识点：

  1. **经验回放（replay buffer）**：几千个并行环境的转移全部汇进一个大池子，训练时
     从池子里随机抽样。数据不再是"采一段、用一遍、扔掉"，而是"存下来、反复抽"。
     on-policy 与 off-policy 的全部区别，在数据这一层直接看得见（对比 v3 的
     `G1WalkRolloutDataset` 与这里的 `G1WalkReplayDataset`）。
  2. **双 Q + 软目标 + 随机策略**：actor 不再输出一个待裁剪的高斯，而是一个"挤压高斯"，
     重参数化采样天然带探索；两个 Q 网络取小治高估，目标网络每步软更新。
  3. **自动温度 α**：最大熵目标里的熵权重不是手调常数，而是被优化到让策略熵匹配目标熵，
     训练自己决定"还该多探索多少"。

一个和 SO-101 那条 off-policy 阶梯一模一样的坑要先躲开：G1 的观测有 99 维、量纲各异，
不做归一化，actor 里的 tanh 会被大数值压饱和、梯度冻住、根本学不动。所以这里同样内联一个
在线观测归一化 `RunningNorm`——回放池里存原始观测，喂进网络前才归一化。

对照口径：off-policy 靠"每条经验反复用"换来样本效率，代价是每步更新更贵。所以最终要画
两张曲线——同一 reward 阈值下，SAC 相对 v3 PPO 在**环境步数**上应显著更省，在**墙钟时间**
上则未必占便宜。这正是 FastSAC/FastTD3 那套配方想讲的道理。
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import lightning as L
import torch
import torch.nn.functional as F
import wandb
from torch import nn
from torch.utils.data import DataLoader, IterableDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env import G1WalkEnv  # noqa: E402


def default_checkpoint_root(run_name: str) -> Path:
    datasets_root = Path(os.environ["DATASETS_ROOT"])
    return datasets_root / "models" / "trained" / "xbotics_rl_g1_walk" / run_name


class RunningNorm(nn.Module):
    """在线观测归一化：跟踪每一维观测的 running mean/var，把量纲各异、幅度可到几十的
    原始观测归一化到零均值单位方差。不做这一步，actor 里的 tanh 会被大数值压进饱和区、
    梯度接近 0——这是 off-policy 连续控制最容易踩的坑。统计量只在采集时更新，池子里存的
    始终是原始观测。"""

    def __init__(self, dim):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(1.0e-4))

    @torch.no_grad()
    def update(self, x):
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean += delta * batch_count / total
        m2 = self.var * self.count + batch_var * batch_count + delta**2 * self.count * batch_count / total
        self.var = m2 / total
        self.count = total

    def normalize(self, x):
        return (x - self.mean) / (self.var.sqrt() + 1.0e-8)


class SquashedGaussianActor(nn.Module):
    """观测 → 随机动作：MLP 出均值和 log 标准差，重参数化采样后过 tanh 挤进有界区间。

    相比前三版的 `ActorCritic.actor`（输出一个无界高斯的均值，靠外部标准差参数探索），
    SAC 的策略本身就是随机的：每次 `get_action` 都显式采样一个动作，并顺带算出它的
    log 概率——Critic 的熵目标和 Actor 的损失都要用到。mjlab 的动作是关节目标的缩放量，
    原始动作空间无界，这里用一个固定对称边界把 tanh 的输出映射进合理幅度。
    """

    LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0

    def __init__(self, obs_dim, action_dim, action_limit, hidden_dim=256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        self.fc_logstd = nn.Linear(hidden_dim, action_dim)
        # 关节目标缩放量，几个标准差就够覆盖正常步态，用一个固定幅度约束 tanh 的值域。
        self.action_limit = action_limit

    def _mean_logstd(self, obs):
        x = self.trunk(obs)
        mean = self.fc_mean(x)
        log_std = torch.tanh(self.fc_logstd(x))
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1.0)
        return mean, log_std

    def get_action(self, obs):
        """返回：随机动作、其 log 概率（含 tanh 雅可比修正）、确定性均值动作（评测用）。"""
        mean, log_std = self._mean_logstd(obs)
        normal = torch.distributions.Normal(mean, log_std.exp())
        x = normal.rsample()  # 重参数化采样，梯度能穿回 mean / log_std
        y = torch.tanh(x)
        action = y * self.action_limit
        # tanh 把采样值挤压了一次，概率密度要按雅可比修正，否则 log_prob 是错的。
        log_prob = normal.log_prob(x) - torch.log(self.action_limit * (1.0 - y.pow(2)) + 1.0e-6)
        log_prob = log_prob.sum(dim=-1)
        det_action = torch.tanh(mean) * self.action_limit
        return action, log_prob, det_action


class QCritic(nn.Module):
    """(观测, 动作) → 标量 Q：拼接后过 MLP，只输出一个数值。两份独立的 Q 用来取小治高估。"""

    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs, action):
        return self.net(torch.cat([obs, action], dim=-1)).squeeze(-1)


class ReplayBuffer:
    """定容经验回放池，整块开在 GPU 上。几千个并行环境每步产出的转移全写进来，训练时随机
    抽 minibatch——这就是 off-policy「每条经验反复用」的载体。存的是原始观测。"""

    def __init__(self, capacity, obs_dim, action_dim, device):
        zeros = lambda *shape: torch.zeros(*shape, device=device)  # noqa: E731
        self.obs = zeros(capacity, obs_dim)
        self.next_obs = zeros(capacity, obs_dim)
        self.action = zeros(capacity, action_dim)
        self.reward = zeros(capacity)
        self.not_done = zeros(capacity)
        self.capacity, self.device = capacity, device
        self.pos, self.full = 0, False

    def __len__(self):
        return self.capacity if self.full else self.pos

    def add(self, obs, action, reward, next_obs, not_done):
        n = obs.shape[0]
        idx = (torch.arange(n, device=self.device) + self.pos) % self.capacity
        self.obs[idx] = obs
        self.next_obs[idx] = next_obs
        self.action[idx] = action
        self.reward[idx] = reward.float()
        self.not_done[idx] = not_done.float()
        self.pos = (self.pos + n) % self.capacity
        self.full = self.full or self.pos < n

    def sample(self, batch_size):
        i = torch.randint(0, len(self), (batch_size,), device=self.device)
        return self.obs[i], self.action[i], self.reward[i], self.next_obs[i], self.not_done[i]


class G1WalkReplayDataset(IterableDataset):
    """off-policy 的数据层：先用当前策略往环境里推几步、把转移写进回放池，再从池子里随机
    抽若干 minibatch 交给训练循环。对比 v3 的 `G1WalkRolloutDataset`——那边一段 rollout
    只用一遍就扔，这边"存下来反复抽"，这就是两条路线最直观的分水岭。"""

    def __init__(self, data_module):
        super().__init__()
        self.dm = data_module

    def __iter__(self):
        dm = self.dm
        # 预热：池子太空时先用均匀随机动作把它填起来，别急着用没学好的策略采样。
        while len(dm.buffer) < dm.learning_starts:
            dm.collect_step(use_policy=False)

        reward_sum, fall_sum = 0.0, 0.0
        for _ in range(dm.steps_per_iter):
            reward_mean, fall_mean = dm.collect_step(use_policy=True)
            reward_sum += reward_mean
            fall_sum += fall_mean
        dm.last_reward = reward_sum / dm.steps_per_iter
        dm.last_fall = fall_sum / dm.steps_per_iter

        # 高更新采样比（UTD）：采一小段、抽很多次，把每条经验榨到位——这是 off-policy 的看家本领。
        for _ in range(dm.updates_per_iter):
            yield dm.buffer.sample(dm.batch_size)


class G1WalkData(L.LightningDataModule):
    """持有持久环境与回放池；每轮先采样（喂池子、更新归一化统计），再把 minibatch 交给 Trainer。"""

    def __init__(self, env, model, buffer, steps_per_iter, updates_per_iter, batch_size, learning_starts):
        super().__init__()
        self.env = env
        self.model = model
        self.buffer = buffer
        self.steps_per_iter = steps_per_iter
        self.updates_per_iter = updates_per_iter
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.obs, _ = env.get_observations()
        self.last_reward = 0.0
        self.last_fall = 0.0

    def collect_step(self, use_policy):
        obs = self.obs
        self.model.obs_norm.update(obs)  # 用原始观测更新归一化统计，每步一次
        if use_policy:
            action = self.model.sample_action(obs)
        else:
            action = self.model.action_limit * (2.0 * torch.rand(
                self.env.num_envs, self.model.action_dim, device=self.env.device) - 1.0)
        next_obs, _next_critic_obs, reward, done, info = self.env.step(action)
        # 摔倒是真正的终止，不该 bootstrap；超时只是回合到点，仍要 bootstrap。
        terminated = torch.logical_and(done, torch.logical_not(info["time_outs"]))
        not_done = torch.logical_not(terminated)
        self.buffer.add(obs, action, reward, next_obs, not_done)
        self.obs = next_obs
        return float(reward.mean().detach().cpu()), float(terminated.float().mean().detach().cpu())

    def train_dataloader(self):
        return DataLoader(G1WalkReplayDataset(self), batch_size=None)


class G1WalkLightningSAC(L.LightningModule):
    """一个 minibatch 的 SAC 更新：评论家（熵折进目标、双 Q 取小）→ 温度 α → 演员（熵正则的
    策略梯度）→ 目标网络软更新。手动优化三个优化器，更新循环仍交给 Lightning。"""

    def __init__(self, obs_dim, action_dim, action_limit, run_name, max_iterations, save_interval,
                 checkpoint_dir, training_settings, wandb_project, wandb_mode):
        super().__init__()
        self.automatic_optimization = False
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_limit = action_limit
        self.run_name = run_name
        self.max_iterations = max_iterations
        self.save_interval = save_interval
        self.checkpoint_dir = checkpoint_dir
        self.training_settings = training_settings
        self.wandb_project = wandb_project
        self.wandb_mode = wandb_mode
        self.latest_checkpoint = self.checkpoint_dir / "model_0.pt"
        self.wandb_run = None
        self.start_time = None
        self.history = []

        self.gamma = 0.99
        self.tau = 0.005
        self.learning_rate = 3.0e-4
        # 梯度裁剪（和前三版 Trainer 里的 gradient_clip_val 同一手段）：off-policy 高更新采样比下，
        # Q 的高估会正反馈地把动作越顶越大直到发散，裁掉过大的梯度范数能把这个失稳按住。
        self.grad_clip = 1.0
        # G1 的单步奖励很小（约 0.05～0.1），而演员损失里熵项 α·logπ 的量级由动作维度撑起来。
        # 不放大奖励，Q 值在损失里压不过熵项，策略会一味维持探索、原地打转而学不会迈步。
        # 放大奖励等价于降低相对温度，让"走得准"的回报信号主导策略更新。
        self.reward_scale = 10.0
        # 目标熵按动作维度定：动作维度越高，允许越确定；α 会被自动调到让策略熵匹配它。
        self.target_entropy = -float(action_dim)
        self.log_alpha = nn.Parameter(torch.zeros(()))
        # 温度下限：G1 这样的高维连续控制里，自动温度会把 α 一路压到接近 0，熵正则彻底消失后
        # 策略会退化成把关节动作顶到极限的确定性策略、越走越僵直到发散。给 α 一个下限，保证
        # 策略始终保留最低限度的随机性，训练能稳在"会走"的状态而不是冲过头。
        self.min_alpha = 0.1

        self.obs_norm = RunningNorm(obs_dim)
        self.actor = SquashedGaussianActor(obs_dim, action_dim, action_limit)
        self.critic1 = QCritic(obs_dim, action_dim)
        self.critic2 = QCritic(obs_dim, action_dim)
        self.critic1_target = QCritic(obs_dim, action_dim)
        self.critic2_target = QCritic(obs_dim, action_dim)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @torch.no_grad()
    def sample_action(self, obs):
        """采集用的随机动作。随机策略本身就是探索，不必像确定性策略那样另外叠噪声。"""
        action, _, _ = self.actor.get_action(self.obs_norm.normalize(obs))
        return action

    @torch.no_grad()
    def eval_action(self, obs):
        """评测用的确定性动作（取分布均值）。"""
        _, _, det_action = self.actor.get_action(self.obs_norm.normalize(obs))
        return det_action

    def setup(self, stage):
        if stage != "fit":
            return
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.start_time = time.time()
        self.wandb_run = wandb.init(
            project=self.wandb_project, name=self.run_name, mode=self.wandb_mode,
            dir=self.checkpoint_dir.as_posix(), config={**self.training_settings, "algo": "v4_sac"},
        )

    def configure_optimizers(self):
        critic_params = list(self.critic1.parameters()) + list(self.critic2.parameters())
        return (
            torch.optim.Adam(self.actor.parameters(), lr=self.learning_rate),
            torch.optim.Adam(critic_params, lr=self.learning_rate),
            torch.optim.Adam([self.log_alpha], lr=self.learning_rate),
        )

    def training_step(self, batch, batch_idx):
        actor_opt, critic_opt, alpha_opt = self.optimizers()
        obs, action, reward, next_obs, not_done = batch
        obs = self.obs_norm.normalize(obs)
        next_obs = self.obs_norm.normalize(next_obs)

        # —— 评论家：双 Q 取小算目标，把熵折进贝尔曼目标；摔倒的转移不 bootstrap ——
        with torch.no_grad():
            next_action, next_logp, _ = self.actor.get_action(next_obs)
            target_q1 = self.critic1_target(next_obs, next_action)
            target_q2 = self.critic2_target(next_obs, next_action)
            target_q = self.reward_scale * reward + self.gamma * not_done * (
                torch.min(target_q1, target_q2) - self.alpha.detach() * next_logp)
        critic_loss = (F.mse_loss(self.critic1(obs, action), target_q)
                       + F.mse_loss(self.critic2(obs, action), target_q))
        critic_opt.zero_grad()
        self.manual_backward(critic_loss)
        torch.nn.utils.clip_grad_norm_(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), self.grad_clip)
        critic_opt.step()

        # —— 温度 α：把当前策略熵调到目标熵 -action_dim ——
        with torch.no_grad():
            _, logp, _ = self.actor.get_action(obs)
        alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
        alpha_opt.zero_grad()
        self.manual_backward(alpha_loss)
        alpha_opt.step()
        with torch.no_grad():
            self.log_alpha.clamp_(min=math.log(self.min_alpha))  # 不让温度掉到熵正则失效的地步

        # —— 演员：最大化 min(q1,q2) − α·熵，即最小化 α·logπ − min(q1,q2) ——
        new_action, new_logp, _ = self.actor.get_action(obs)
        q1_pi = self.critic1(obs, new_action)
        q2_pi = self.critic2(obs, new_action)
        actor_loss = (self.alpha.detach() * new_logp - torch.min(q1_pi, q2_pi)).mean()
        actor_opt.zero_grad()
        self.manual_backward(actor_loss)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        actor_opt.step()

        # 目标网络每步软更新，让目标值缓慢跟上。
        with torch.no_grad():
            for p, tp in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                tp.mul_(1.0 - self.tau).add_(self.tau * p)
            for p, tp in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                tp.mul_(1.0 - self.tau).add_(self.tau * p)

        self.log_dict(
            {"critic_loss": critic_loss.detach(), "actor_loss": actor_loss.detach(), "alpha": self.alpha.detach()},
            prog_bar=True, on_step=True, on_epoch=False,
        )

    def on_train_epoch_end(self):
        iteration = self.current_epoch + 1
        elapsed = time.time() - self.start_time
        data_module = self.trainer.datamodule
        # 环境步数按"每轮采集 steps_per_iter 步 × 并行环境数"记，和 PPO 的"每轮 rollout 步 × 环境数"同口径，
        # 两条曲线的横轴才可比。
        env_steps = iteration * data_module.steps_per_iter * data_module.env.num_envs
        reward = data_module.last_reward
        fall = data_module.last_fall
        # 同一张 wandb run 里同时记 env_steps 与墙钟，方便后面画两条对照曲线。
        metrics = {"reward": reward, "fall_rate": fall, "alpha": float(self.alpha.detach().cpu()),
                   "env_steps": env_steps, "elapsed_time": elapsed}
        wandb.log(metrics, step=iteration)
        self.history.append({"iteration": iteration, **metrics})
        if iteration % 20 == 0 or iteration == self.max_iterations:
            print(f"  iter {iteration}: reward={reward:.4f}  fall_rate={fall:.3f}  "
                  f"alpha={metrics['alpha']:.3f}  env_steps={env_steps}  elapsed={elapsed:.0f}s", flush=True)
        if iteration % self.save_interval == 0 or iteration == self.max_iterations:
            self.latest_checkpoint = self.checkpoint_dir / f"model_{iteration}.pt"
            self.save_checkpoint_file(self.latest_checkpoint, iteration)

    def save_checkpoint_file(self, path, iteration):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "iteration": iteration,
                "actor": self.actor.state_dict(),
                "obs_norm": self.obs_norm.state_dict(),
                "training_settings": self.training_settings,
                "history": self.history,
            },
            path,
        )

    def teardown(self, stage):
        # 把 reward / env_steps / 墙钟的完整曲线单独存一份，画对照图时不必依赖 wandb。
        (self.checkpoint_dir / "history.json").write_text(json.dumps(self.history, indent=2) + "\n")
        if self.wandb_run is not None:
            self.wandb_run.finish()


def run_training(run_name, num_envs, max_iterations, steps_per_iter, updates_per_iter, batch_size,
                 buffer_capacity, learning_starts, action_limit, device,
                 seed=1, checkpoint_dir=None, wandb_project="rl_class", wandb_mode="online"):
    checkpoint_dir = checkpoint_dir or default_checkpoint_root(run_name)

    torch.manual_seed(seed)
    env = G1WalkEnv(num_envs=num_envs, device=device, seed=seed)
    env.reset()
    action_limit_t = torch.full((env.action_dim,), float(action_limit), device=env.device)
    model = G1WalkLightningSAC(
        obs_dim=env.obs_dim, action_dim=env.action_dim, action_limit=action_limit_t,
        run_name=run_name, max_iterations=max_iterations, save_interval=max(1, max_iterations // 10),
        checkpoint_dir=checkpoint_dir, training_settings={}, wandb_project=wandb_project, wandb_mode=wandb_mode,
    )
    model.to(env.device)
    buffer = ReplayBuffer(buffer_capacity, env.obs_dim, env.action_dim, env.device)

    model.training_settings.update({
        "run_name": run_name, "num_envs": num_envs, "max_iterations": max_iterations,
        "steps_per_iter": steps_per_iter, "updates_per_iter": updates_per_iter, "batch_size": batch_size,
        "buffer_capacity": buffer_capacity, "learning_starts": learning_starts, "action_limit": float(action_limit),
        "device": device, "seed": seed, "gamma": model.gamma, "tau": model.tau,
        "learning_rate": model.learning_rate, "reward_scale": model.reward_scale,
        "obs_dim": env.obs_dim, "action_dim": env.action_dim,
    })

    data = G1WalkData(env, model, buffer, steps_per_iter, updates_per_iter, batch_size, learning_starts)
    trainer = L.Trainer(
        accelerator="gpu" if device != "cpu" and torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=max_iterations,
        reload_dataloaders_every_n_epochs=1,
        enable_checkpointing=False,
        logger=False,
        enable_model_summary=False,
        enable_progress_bar=True,
        log_every_n_steps=1,
    )
    trainer.fit(model, data)
    env.close()
    return model.latest_checkpoint


def main():
    # off-policy 配方：几千个并行环境采样进同一个回放池、大 batch、较高的更新采样比（UTD）。
    # 动作幅度用一个固定对称边界（关节目标缩放量，几个标准差足够覆盖正常步态）。
    run_training(
        run_name="g1-walk-sac",
        num_envs=1024,
        max_iterations=3000,
        steps_per_iter=8,
        updates_per_iter=8,
        batch_size=8192,
        buffer_capacity=1_500_000,
        learning_starts=8192,
        action_limit=1.0,
        device="cuda:0",
        wandb_project="rl_class",
        wandb_mode="online",
    )


if __name__ == "__main__":
    main()
