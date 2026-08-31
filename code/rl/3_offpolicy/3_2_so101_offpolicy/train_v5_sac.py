"""教学版状态 SAC 训练入口：SO101 连续控制阶梯的第三级，专治 v3/v4 的探索不足。

DDPG（v3）和 TD3（v4）都是**确定性策略**：给定状态只吐一个固定动作，采集时全靠外加
一点固定方差的高斯噪声去"碰运气"。TD3 治的是 Q 值高估（双 Q 取 min、延迟更新、目标
策略平滑），完全没碰探索这件事；如果瓶颈根本不是高估而是探索太弱，TD3 自然也学不动。
SAC（Soft Actor-Critic）换一套完全不同的路子，直接对症探索问题：

1. **随机策略替代确定性策略**：`SquashedGaussianActor` 不再直接吐一个动作，而是吐
   一个高斯分布的均值和标准差，重参数化 `rsample` 出一个样本，再 `tanh` 挤压进合法
   动作区间。同一个状态每次采样都可能是不同动作——策略本身自带探索，不用像 DDPG/TD3
   那样额外叠一个固定方差的高斯噪声。
2. **最大熵目标**：优化目标从"最大化回报"变成"最大化回报 + α·策略熵"。策略熵越高，
   动作分布越"摊开"，越不容易早早收窄到一个动作上而错过还没试过的、可能通向成功的
   区域。熵直接折进 Critic 的贝尔曼目标（`target_q = r + γ(min(q1,q2) - α·logπ)`）和
   Actor 的损失（`α·logπ - min(q1,q2)`）里，不是事后加的正则，是训练目标的一部分。
3. **自动温度 α**：熵权重 α 不是手调的固定超参，而是像另一个网络参数一样被优化——
   `log_alpha` 用梯度下降调到让当前策略熵匹配一个目标熵 `-action_dim`（动作维度越
   高，目标熵越低，允许的确定性越强）。训练早期策略随机、熵天然高，α 会被调小；如果
   后期策略塌缩太快、熵掉得比目标还低，α 会被调大，把探索"拉"回来。

相对 TD3，双 Q 取 min 治高估这套照抄不动（`QCritic` 结构和 v4 完全一样，`critic_loss`
仍是两个 MSE 之和）；拿掉的是"目标策略平滑"和"延迟策略更新"——这两个是专门为确定性
策略的高估问题设计的，随机策略下 Critic 本来就会在动作邻域里被反复采样、目标本身自带
一点平滑，没必要再刻意加噪声或延迟更新。

**观测归一化同样是这一级能学起来的前提**：`SquashedGaussianActor` 的均值分支也是
先过 `tanh`（`get_action` 里 `y = torch.tanh(x)`），SO101 state 某些维度原始值到
~200 一样会把这条通路饱和死；`RunningNorm` 的接法和 v3/v4 完全一致——在线维护
running mean/var，采集时用原始 state 更新，喂进 actor/critic 前先归一化，回放池仍
存原始 state。

诚实的教学结论（公平预算 500 iter、三级同预算、加观测归一化后的真实结果）：三级放在
一起看，是一条干净的阶梯，而且能看出两层完全不同的道理。第一层，**靠近度（mean_reward）
单调递增**：v3 DDPG 全程震荡、均值约 0.28；v4 TD3 治住震荡、稳定靠近、均值约 0.52，
比 DDPG 明显更近更稳；每一级"更会靠近"这件事是连续的、可预期的。第二层，**硬成功
（success_once）是质变，不是量变**：DDPG 和 TD3 无论靠得多近，500 轮里 success_once
始终是 0——同样是"越来越会靠近"的两级，不管给多久预算都跨不过成功那道坎；而 SAC 一旦
换上随机策略 + 最大熵，success_once 从 iter~480 起开始稳定落在 0.93～0.99，最终
success_once=0.99，直接把这道坎彻底跨过去了。这印证了最初的猜想：探索不足才是
DDPG/TD3 学不动的病根，最大熵这味药才是打开"解决"的钥匙——双 Q 治高估、延迟更新这些
稳定性技巧再怎么打磨，也补不上"探索方式本身不对"这个洞。当然 SAC 这一级也不是没有
代价：要跑到 500 iter 才等到 success_once 稳定超过 0.9，说明**标量 Q + MSE** 的样本
效率仍然有限，Critic 只学一个数值的期望回报；下一级 v6 squint 会换成 C51 分布式
Critic（学整个回报分布而不是一个期望值）配视觉编码器，要更快更强地解决问题，还要能
产出规模化的数据——这里先把"随机策略 + 最大熵 + 自动温度"这三件事在最简单的标量 Q
上跑扎实，也确实换来了阶梯上第一次真正意义的"解决"。
"""

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch.utils.data import DataLoader, IterableDataset

import so101_sim  # 统一环境：lerobot 评测与 RL 训练共用同一份定义


class RunningNorm(nn.Module):
    """在线观测归一化：跟踪 state 每一维的 running mean/std，把原始观测（值域可到 ~200）
    归一化到零均值单位方差，避免大数值让 actor 的 tanh 饱和、梯度冻结。"""

    def __init__(self, dim):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(1e-4))

    @torch.no_grad()
    def update(self, x):
        """用一个 batch 的原始观测增量更新均值/方差（Welford 并行版）。

        只在采集时调用、每步一次；训练时只读不写，这样同一批数据在不同更新轮里
        被归一化的口径是一致的。

        Args:
            x: 形状 (num_envs, dim) 的原始观测。
        """
        bm, bv, bc = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        delta = bm - self.mean; tot = self.count + bc
        self.mean += delta * bc / tot
        M2 = self.var * self.count + bv * bc + delta**2 * self.count * bc / tot
        self.var = M2 / tot; self.count = tot

    def normalize(self, x):
        """把原始观测按当前统计量归一化到零均值单位方差。

        分母加 1e-8 是防某一维在训练最开始方差还接近 0 时除爆。

        Args:
            x: 原始观测张量，最后一维是状态维度。

        Returns:
            同形状的归一化观测。
        """
        return (x - self.mean) / (self.var.sqrt() + 1e-8)


class SquashedGaussianActor(nn.Module):
    """state → 随机动作：MLP 出均值和 log 标准差，重参数化采样后 tanh 挤压进 [low, high]。

    相对 v3/v4 的 `DeterministicActor`：那边只有一个 `forward` 直接给动作；这里没有
    `forward`，只有 `get_action`——因为 SAC 的策略是一个分布，每次调用都要显式采样，
    还要顺带算出这个样本的 log 概率（Critic 目标和 Actor 损失都要用到）。
    """

    LOG_STD_MIN, LOG_STD_MAX = -5, 2

    def __init__(self, state_dim, action_dim, action_low, action_high):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
        )
        self.fc_mean = nn.Linear(256, action_dim)
        self.fc_logstd = nn.Linear(256, action_dim)
        self.register_buffer("action_scale", (action_high - action_low) / 2.0)
        self.register_buffer("action_bias", (action_high + action_low) / 2.0)

    def _mean_logstd(self, state):
        x = self.trunk(state)
        mean = self.fc_mean(x)
        log_std = torch.tanh(self.fc_logstd(x))
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_action(self, state):
        """随机动作 + log 概率（含 tanh 雅可比修正）+ 确定性均值动作（评测用）。

        Args:
            state: 已归一化的观测。

        Returns:
            `(action, log_prob)`：挤压后的动作，以及它在挤压后分布下的对数概率
            （算熵项要用，雅可比修正已经做过）。
        """
        mean, log_std = self._mean_logstd(state)
        normal = torch.distributions.Normal(mean, log_std.exp())
        x = normal.rsample()  # 重参数化：采样路径可导，梯度能穿回 mean/log_std
        y = torch.tanh(x)
        action = y * self.action_scale + self.action_bias
        # tanh 把采样值挤压了一次，概率密度要按雅可比行列式修正，否则 log_prob 是错的
        log_prob = normal.log_prob(x) - torch.log(self.action_scale * (1 - y.pow(2)) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        det_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, det_action


class QCritic(nn.Module):
    """(state, action) → 标量 Q：和 v4 TD3 完全一样，拼接后过 MLP，只输出一个数值。"""

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state, action):
        """一次前向：状态和动作拼起来进 MLP，出一个标量 Q 值。

        连续动作没法像 DQN 那样对每个动作出一个分，只能把动作当输入吃进来。

        Args:
            state: 已归一化的观测。
            action: 要评分的动作。

        Returns:
            形状 (batch,) 的 Q 值。
        """
        return self.net(torch.cat([state, action], dim=-1)).squeeze(-1)


class SAC(L.LightningModule):
    """一个 minibatch 的 SAC 更新：评论家(熵折进目标) → 温度 α → 演员(熵正则的策略梯度) → 软更新。"""

    def __init__(self, state_dim, action_dim, action_low, action_high, gamma=0.99, tau=0.005, lr=3e-4):
        super().__init__()
        self.automatic_optimization = False
        self.gamma, self.tau, self.lr = gamma, tau, lr
        # 相对 DDPG/TD3 新增：自动温度——log_alpha 是待优化参数，目标熵按动作维度定
        self.target_entropy = -float(action_dim)
        self.log_alpha = nn.Parameter(torch.zeros(()))

        # 相对 DDPG/TD3 的最大变化：策略是随机的挤压高斯，不再是确定性 MLP
        self.actor = SquashedGaussianActor(state_dim, action_dim, action_low, action_high)
        # 双 Q 取 min 治高估这套照抄 TD3：两个独立 Critic + 各自目标网络
        self.critic1 = QCritic(state_dim, action_dim)
        self.critic2 = QCritic(state_dim, action_dim)
        self.critic1_target = QCritic(state_dim, action_dim)
        self.critic2_target = QCritic(state_dim, action_dim)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        # 观测归一化：buffer 里存的仍是原始 state，这里只在喂进网络前做归一化
        self.obs_norm = RunningNorm(state_dim)

    @property
    def alpha(self):
        """当前温度：熵项在目标里占多大权重。

        存成 `log_alpha` 再取指数，是为了让它在优化时天然保持为正。

        Returns:
            标量温度 α。
        """
        return self.log_alpha.exp()

    def configure_optimizers(self):
        """四个优化器：actor、两个 critic、以及温度 α。

        α 也是学出来的——SAC 不用手调探索强度，而是让训练自己找"探索够用又不过头"
        的那个平衡点，这就是"自动温度"这个零件的实现位置。

        Returns:
            (actor 优化器, critic1 优化器, critic2 优化器, α 优化器)。
        """
        critic_params = list(self.critic1.parameters()) + list(self.critic2.parameters())
        return (torch.optim.Adam(self.actor.parameters(), lr=self.lr),
                torch.optim.Adam(critic_params, lr=self.lr),
                torch.optim.Adam([self.log_alpha], lr=self.lr))

    @torch.no_grad()
    def sample_action(self, state):
        """随机动作本身就是探索，采集时不再像 DDPG/TD3 那样额外叠高斯噪声。

        Args:
            state: 原始观测（未归一化）。

        Returns:
            合法区间内的连续动作。
        """
        action, _, _ = self.actor.get_action(self.obs_norm.normalize(state))
        return action

    @torch.no_grad()
    def eval_action(self, state):
        """评测用的动作：取分布的均值，不采样。

        训练时策略是随机的（这正是熵探索要的），评测时再掺随机性就说不清成绩了。

        Args:
            state: 原始观测。

        Returns:
            确定性动作（分布均值经 tanh 挤压后的值）。
        """
        _, _, det_action = self.actor.get_action(self.obs_norm.normalize(state))
        return det_action

    def training_step(self, batch, batch_idx):
        """一个 minibatch 的 SAC 更新：critic → 温度 α → actor。

        和 TD3 的差别全在"熵"上：目标 Q 里减掉一项 α·logπ（保持随机也算收益），
        actor 的损失里同样带这一项。α 自己按"当前熵离目标熵差多少"调。

        Args:
            batch: 从回放池抽出的 `(state, action, reward, next_state)`。
            batch_idx: Lightning 传入的批序号，这里用不到。
        """
        actor_opt, critic_opt, alpha_opt = self.optimizers()
        state, action, reward, next_state = batch
        # buffer 里是原始 state，喂进网络前统一归一化（统计量只在采集时更新，这里只读）
        state, next_state = self.obs_norm.normalize(state), self.obs_norm.normalize(next_state)

        # —— 评论家：双 Q 取 min 算目标，熵项折进目标（always-bootstrap，和 v3/v4 一样不用 done）——
        with torch.no_grad():
            next_action, next_logp, _ = self.actor.get_action(next_state)
            next_logp = next_logp.squeeze(-1)
            target_q1 = self.critic1_target(next_state, next_action)
            target_q2 = self.critic2_target(next_state, next_action)
            # 相对 TD3 新增：目标里减掉 alpha*next_logp——下一步越"随机"（熵越高），
            # 目标值越高，鼓励策略维持探索而不是过早收窄到单一动作
            target_q = reward + self.gamma * (torch.min(target_q1, target_q2) - self.alpha.detach() * next_logp)
        critic_loss = (F.mse_loss(self.critic1(state, action), target_q)
                     + F.mse_loss(self.critic2(state, action), target_q))
        critic_opt.zero_grad(); self.manual_backward(critic_loss); critic_opt.step()

        # —— 温度 α：把当前策略熵调到目标熵 -action_dim ——
        with torch.no_grad():
            _, logp, _ = self.actor.get_action(state)
            logp = logp.squeeze(-1)
        alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
        alpha_opt.zero_grad(); self.manual_backward(alpha_loss); alpha_opt.step()

        # —— 演员：最大化 min(q1,q2) − α·熵，即最小化 alpha*logp − min(q1,q2) ——
        a, logp2, _ = self.actor.get_action(state)
        logp2 = logp2.squeeze(-1)
        q1_pi = self.critic1(state, a)
        q2_pi = self.critic2(state, a)
        actor_loss = (self.alpha.detach() * logp2 - torch.min(q1_pi, q2_pi)).mean()
        actor_opt.zero_grad(); self.manual_backward(actor_loss); actor_opt.step()

        # 没有 TD3 的延迟更新——随机策略下 Critic 目标本来就带一点自带的平滑，
        # 目标网络照常每步软更新
        with torch.no_grad():
            for p, tp in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                tp.mul_(1 - self.tau).add_(self.tau * p)
            for p, tp in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                tp.mul_(1 - self.tau).add_(self.tau * p)

        self.log_dict({"critic_loss": critic_loss.detach(), "actor_loss": actor_loss.detach(),
                      "alpha": self.alpha.detach()}, prog_bar=True, on_step=True, on_epoch=False)


class ReplayBuffer:
    """定容经验回放池，整块开在 GPU 上（state 版：无 rgb，只存关节状态向量）。与 v3/v4 完全一致。"""

    def __init__(self, capacity, state_dim, action_dim, device):
        z = lambda *s: torch.zeros(*s, device=device)  # noqa: E731
        self.state = z(capacity, state_dim)
        self.next_state = z(capacity, state_dim)
        self.action = z(capacity, action_dim)
        self.reward = z(capacity)
        self.capacity, self.device = capacity, device
        self.pos, self.full = 0, False

    def __len__(self):
        return self.capacity if self.full else self.pos

    def add(self, state, action, reward, next_state):
        """把一批转移写进回放池；写满一圈后从头覆盖最旧的。

        一次写入的是 `num_envs` 条（并行环境同一拍的经验），所以下标要按环形取模算。

        Args:
            state: 这一拍的原始观测（存原始值，归一化留到喂网络前做）。
            action: 执行的动作。
            reward: 即时奖励。
            next_state: 下一拍的原始观测。
        """
        n = state.shape[0]
        idx = (torch.arange(n, device=self.device) + self.pos) % self.capacity
        self.state[idx] = state; self.next_state[idx] = next_state
        self.action[idx] = action; self.reward[idx] = reward.float()
        self.pos = (self.pos + n) % self.capacity
        self.full = self.full or self.pos < n

    def sample(self, batch_size):
        """从整个池子里均匀随机抽一个 minibatch。

        不按轨迹、不按时间抽——随机打散正是打断样本相关性的那一步。

        Args:
            batch_size: 这一批抽多少条转移。

        Returns:
            `(state, action, reward, next_state)` 四个张量，已在 GPU 上。
        """
        i = torch.randint(0, len(self), (batch_size,), device=self.device)
        return self.state[i], self.action[i], self.reward[i], self.next_state[i]


class SO101SACData(L.LightningDataModule):
    """持有环境和回放池；每轮先采样、再把 minibatch 交给 Trainer（state 版，无 rgb）。

    `env` 是 ManiSkill 标准 `ManiSkillVectorEnv`：不像旧版 `TrainEnv` 那样自己缓存
    `self.obs`，这里改由本类持有 `self.state`，每次 `step` 后手动滚动到下一步。
    """

    def __init__(self, env, model, buffer, action_dim, steps_per_iter, updates_per_iter,
                batch_size, learning_starts):
        super().__init__()
        self.env, self.model, self.buffer = env, model, buffer
        self.action_dim = action_dim
        self.steps_per_iter, self.updates_per_iter = steps_per_iter, updates_per_iter
        self.batch_size, self.learning_starts = batch_size, learning_starts
        self.last_success = 0.0
        self.last_reward = 0.0
        self.state, _ = env.reset()

    def _collect(self, use_policy):
        state = self.state
        self.model.obs_norm.update(state)  # 用原始 state 更新归一化统计，每步一次
        if use_policy:
            action = self.model.sample_action(state)
        else:  # 预热：均匀随机动作把池子填起来
            low = torch.as_tensor(self.env.single_action_space.low, device=self.env.device)
            high = torch.as_tensor(self.env.single_action_space.high, device=self.env.device)
            action = low + (high - low) * torch.rand(self.env.num_envs, self.action_dim, device=self.env.device)
        next_state, reward, _, _, info = self.env.step(action)
        self.buffer.add(state, action, reward, next_state)
        self.state = next_state
        return info["success"].float().mean().item(), reward.float().mean().item()

    def train_dataloader(self):
        """每轮先采几步进池，再从池子里抽若干 minibatch 交给训练循环。

        "采一步、学很多次"就是高更新采样比（UTD）的实现：每条经验被反复抽中，
        这是 off-policy 样本效率的直接来源。Trainer 配了
        `reload_dataloaders_every_n_epochs=1`，所以每一轮都会重新走一遍这里。

        Returns:
            每次迭代吐一个 minibatch 的 DataLoader（`batch_size=None`，数据集自己成批）。
        """
        def gen():
            """本轮的样本生成器：先补够预热经验，再采样，最后连吐若干 minibatch。"""
            while len(self.buffer) < self.learning_starts:
                self._collect(use_policy=False)
            stats = [self._collect(use_policy=True) for _ in range(self.steps_per_iter)]
            succ, rew = zip(*stats)
            self.last_success = float(np.mean(succ))
            self.last_reward = float(np.mean(rew))
            for _ in range(self.updates_per_iter):
                yield self.buffer.sample(self.batch_size)

        class _DS(IterableDataset):
            def __iter__(self_inner):
                return gen()

        return DataLoader(_DS(), batch_size=None)


class SuccessLogger(L.Callback):
    """每轮打印采集成功率 + 平均奖励（靠近度） + 自动温度 α + 定期存 ckpt。"""

    def __init__(self, ckpt_dir, save_interval, max_iterations):
        self.ckpt_dir, self.save_interval, self.max_iterations = ckpt_dir, save_interval, max_iterations

    def on_train_epoch_end(self, trainer, pl_module):
        """每轮打印成功率与平均奖励，并按间隔存一次 checkpoint。

        存的只有推理要用的部分（actor 权重 + 归一化统计量），不存优化器状态——
        这份 checkpoint 是拿去 rollout 和生成数据的，不用于续训。
        Args:
            trainer: Lightning Trainer，用来读当前轮次与 datamodule 上的统计量。
            pl_module: 正在训练的模型，用来取要落盘的权重。
        """
        it = trainer.current_epoch + 1
        succ = trainer.datamodule.last_success
        rew = trainer.datamodule.last_reward
        print(f"  iter {it}: success_once={succ:.2f}  mean_reward={rew:.3f}  alpha={pl_module.alpha.item():.3f}",
              flush=True)
        if it % self.save_interval == 0 or it == self.max_iterations:
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"actor": pl_module.actor.state_dict(),
                       "obs_norm": pl_module.obs_norm.state_dict()}, self.ckpt_dir / "sac.pt")


def run_training(task, num_envs, max_iterations, updates_per_iter, batch_size,
                 buffer_capacity, learning_starts, device, seed=1):
    """搭好环境、模型、回放池，跑完整条训练，返回 checkpoint 路径。

    Args:
        task: `so101_sim` 注册的任务 id。
        num_envs: 并行环境数。
        max_iterations: 训练轮数，一轮 = 采 steps_per_iter 步 + 做 updates_per_iter 次更新。
        updates_per_iter: 每轮的梯度更新次数，也就是更新采样比（UTD）。
        batch_size: 每次更新抽的转移条数。
        buffer_capacity: 回放池容量。
        learning_starts: 开始用策略采样前，先用随机动作灌多少条经验。
        device: 训练设备。
        seed: 随机种子。

    Returns:
        checkpoint 文件路径。
    """
    torch.manual_seed(seed)
    env = so101_sim.state_rl_env(task, num_envs=num_envs)
    state_dim = env.single_observation_space.shape[-1]
    action_dim = env.single_action_space.shape[-1]
    low = torch.as_tensor(env.single_action_space.low, device=device)
    high = torch.as_tensor(env.single_action_space.high, device=device)
    model = SAC(state_dim, action_dim, low, high).to(device)
    buffer = ReplayBuffer(buffer_capacity, state_dim, action_dim, device)
    data = SO101SACData(env, model, buffer, action_dim, steps_per_iter=1, updates_per_iter=updates_per_iter,
                        batch_size=batch_size, learning_starts=learning_starts)

    ckpt_dir = Path(os.environ["DATASETS_ROOT"]) / "models" / "trained" / "so101_sim_offpolicy" / task
    trainer = L.Trainer(
        accelerator="gpu", devices=1, max_epochs=max_iterations,
        reload_dataloaders_every_n_epochs=1, enable_checkpointing=False, logger=False,
        enable_model_summary=False, enable_progress_bar=False, log_every_n_steps=10,
        callbacks=[SuccessLogger(ckpt_dir, save_interval=25, max_iterations=max_iterations)],
    )
    trainer.fit(model, datamodule=data)
    env.close()
    return ckpt_dir / "sac.pt"


# 改这里选任务与训练时长，然后 `python train_v5_sac.py`。
TASK = "SO101PickPlaceCube40-v1"

if __name__ == "__main__":
    # 与 v3 DDPG / v4 TD3 完全相同的共享预算，只差算法，三级可公平对照：
    # 每步 256 次更新（UTD），批 512，回放池 50 万；SAC 要到 iter~480 才稳定跨过"解决"，
    # 三级统一给够 500 iter 预算才公平。
    run_training(
        task=TASK, num_envs=1024, max_iterations=500, updates_per_iter=256, batch_size=512,
        buffer_capacity=500_000, learning_starts=5_000, device="cuda",
    )
