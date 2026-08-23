"""教学版视觉分布式 SAC（squint）训练入口：连续控制阶梯第六级，也是最后一级。

承接 v5：v5 SAC 已经用"随机策略 + 最大熵"把 success_once 从 0 拉到 0.99，但代价是
要跑到 iter~480 才稳定跨过这道坎——Critic 只学一个**标量** Q（回报的期望值），样本
效率就这样了；而且 v5 看的是 state（关节角度这类底层量），换成真实场景根本拿不到。
v6 从两个方向同时升级：

1. **输入换成 16×16 像素**：`CNNEncoder` 直接吃 RGB 图，不再依赖仿真给的关节状态——
   这才是真机能落地的输入形式，也顺带证明"纯视觉也能从零学会操作"。
2. **Critic 换成 C51 分布式**：`C51TwinQ` 不再只估一个 Q 数值，而是估"回报落在
   101 个支点上的概率分布"，用分类交叉熵而不是 MSE 去拟合分布式贝尔曼目标
   （`categorical_target`）。学一整条分布比学一个期望值提供的梯度信号更丰富，
   这正是 squint 能更快更稳收敛的关键——同样是 SAC 的骨架（随机策略＋最大熵＋自动
   温度，照抄 v5 不动），只把 Critic 从"一个数"换成"一条分布"，success≈0.99 到得
   更快、也更稳。**这套 C51 算法本身已验证到 0.99，本文件不改它**。

v6 还有一个"顺便"的身份：练出来的策略不只是教学终点，其 rollout 轨迹换个外观就是
VLA 课要的仿真数据——见同目录 `datagen/`（`rollout.py` 加载这里存的 ckpt 跑轨迹，
`replay.py` 换外观重渲，`to_lerobot.py` 转 LeRobotDataset）。

**这一级不需要 v3/v4/v5 那样外挂的 `RunningNorm`**：v3/v4/v5 的观测是原始关节状态，
数值到 ~200 会把 actor 的 tanh 分支饱和死，才需要在线维护 running mean/var 手动
归一化；v6 的视觉特征在 `Projection` 里已经过 `nn.LayerNorm`（`rgb_proj`/`state_proj`
两路都是 `Linear → LayerNorm → 激活`），LayerNorm 本身就把每层输出重新拉回零均值
单位方差，相当于归一化自带在网络结构里，不用再单独挂一层。

off-policy 的数据流和 on-policy（rl/1_1 的 PPO）不同：这里有个**经验回放池**，
每一轮先用当前策略采几步塞进池子，再从池子里随机抽若干 minibatch 做 SAC 更新——
"边玩边学、旧经验反复用"正是 off-policy 样本高效的来源。

四件套：环境 `so101_sim.visual_rl_env(...)`（与 lerobot 评测共用
同一个 so101_sim 环境定义）＋ 本文件顶部的网络/更新 `VisualSAC`（LightningModule）
＋ 数据 `SO101SACData`（LightningDataModule，持有环境和回放池）＋ 本文件的 `trainer.fit`。
自包含单文件，和 v3/v4/v5 一样：模型在前，`ReplayBuffer`/`trainer.fit` 在后。
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


def mlp(sizes, out_activation=False):
    layers = []
    for i in range(len(sizes) - 1):
        layers += [nn.Linear(sizes[i], sizes[i + 1])]
        if i < len(sizes) - 2 or out_activation:
            layers += [nn.LayerNorm(sizes[i + 1]), nn.ReLU()]
    return nn.Sequential(*layers)


class CNNEncoder(nn.Module):
    """16×16 RGB → 1024 维特征。归一化到 [-0.5, 0.5] 再过两层卷积。"""

    def __init__(self):
        super().__init__()
        self.repr_dim = 1024
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=1), nn.ReLU(),
            nn.Flatten(),
        )

    def forward(self, rgb):
        x = rgb.permute(0, 3, 1, 2).float() / 255.0 - 0.5
        return self.conv(x)


class Projection(nn.Module):
    """视觉特征压到 50、关节状态放大到 256，再拼接。两路都是 `Linear → LayerNorm → 激活`——
    这个 LayerNorm 就是 v6 不需要外挂 `RunningNorm` 的原因（见文件顶部说明）。"""

    def __init__(self, feat_dim, state_dim):
        super().__init__()
        self.repr_dim = 50 + 256
        self.rgb_proj = nn.Sequential(nn.Linear(feat_dim, 50), nn.LayerNorm(50), nn.Tanh())
        self.state_proj = nn.Sequential(nn.Linear(state_dim, 256), nn.LayerNorm(256), nn.ReLU())

    def forward(self, feat, state):
        return torch.cat([self.rgb_proj(feat), self.state_proj(state)], dim=-1)


class Actor(nn.Module):
    """挤压高斯策略：输出动作均值与 log 标准差，采样后 tanh 压进 [low, high]。
    和 v5 的 `SquashedGaussianActor` 同一套写法，只是喂进 trunk 前先过 `Projection`
    把视觉特征和关节状态拼起来。"""

    LOG_STD_MIN, LOG_STD_MAX = -5, 2

    def __init__(self, feat_dim, state_dim, action_dim, action_low, action_high):
        super().__init__()
        self.proj = Projection(feat_dim, state_dim)
        self.trunk = mlp([self.proj.repr_dim, 256, 256, 256], out_activation=True)
        self.fc_mean = nn.Linear(256, action_dim)
        self.fc_logstd = nn.Linear(256, action_dim)
        self.register_buffer("action_scale", (action_high - action_low) / 2.0)
        self.register_buffer("action_bias", (action_high + action_low) / 2.0)

    def _mean_logstd(self, feat, state):
        x = self.trunk(self.proj(feat, state))
        mean = self.fc_mean(x)
        log_std = torch.tanh(self.fc_logstd(x))
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_action(self, feat, state):
        """随机动作 + log 概率（含 tanh 雅可比修正）+ 确定性均值动作。"""
        mean, log_std = self._mean_logstd(feat, state)
        normal = torch.distributions.Normal(mean, log_std.exp())
        x = normal.rsample()
        y = torch.tanh(x)
        action = y * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x) - torch.log(self.action_scale * (1 - y.pow(2)) + 1e-6)
        det_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob.sum(-1, keepdim=True), det_action


class C51TwinQ(nn.Module):
    """分布式 C51 孪生 Q：每个 Q 估的是"回报落在 num_atoms 个支点上的概率分布"（不是单个数值）。

    相对 v3/v4/v5 的标量 `QCritic`（拼接后过 MLP 直出一个数）：这里最后一层出
    `num_atoms=101` 维 logits，softmax 成一条分布，训练目标不是 MSE 回归一个值，
    而是让预测分布去逼近"贝尔曼目标分布"（交叉熵，见下面 `categorical_target`）。
    工程上唯一"教学化"的地方：研究版用 vmap 把两个 Q 网络叠一起跑，这里就老老实实
    写成两个独立网络（`q1`/`q2`）。
    """

    def __init__(self, feat_dim, state_dim, action_dim, num_atoms=101, v_min=-20.0, v_max=20.0):
        super().__init__()
        self.num_atoms, self.v_min, self.v_max = num_atoms, v_min, v_max
        self.register_buffer("support", torch.linspace(v_min, v_max, num_atoms))
        self.proj1 = Projection(feat_dim, state_dim)
        self.proj2 = Projection(feat_dim, state_dim)
        head = lambda p: mlp([p.repr_dim + action_dim, 512, 512, 512, num_atoms])  # noqa: E731
        self.q1, self.q2 = head(self.proj1), head(self.proj2)

    def logits(self, feat, state, action):
        """两个 Q 网络的原始 logits，堆成 [2, batch, num_atoms]。"""
        l1 = self.q1(torch.cat([self.proj1(feat, state), action], dim=-1))
        l2 = self.q2(torch.cat([self.proj2(feat, state), action], dim=-1))
        return torch.stack([l1, l2], dim=0)

    def expected_q(self, feat, state, action):
        """把分布折算回期望 Q 值：[2, batch]。"""
        probs = F.softmax(self.logits(feat, state, action), dim=-1)
        return torch.sum(probs * self.support, dim=-1)

    def categorical_target(self, feat, state, action, rewards, discount):
        """C51 的分布式贝尔曼目标：把下一步的分布沿支点平移·投影回来。[2, batch, num_atoms]。"""
        delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        target_z = (rewards.unsqueeze(-1) + discount * self.support).clamp(self.v_min, self.v_max)  # [B,atoms]
        b = (target_z - self.v_min) / delta_z
        lower, upper = b.floor().long(), b.ceil().long()
        is_int = upper == lower
        lower = torch.where((lower > 0) & is_int, lower - 1, lower)
        upper = torch.where((lower == 0) & is_int, upper + 1, upper)

        next_dist = F.softmax(self.logits(feat, state, action), dim=-1)  # [2,B,atoms]
        lo = lower.unsqueeze(0).expand_as(next_dist)
        up = upper.unsqueeze(0).expand_as(next_dist)
        b_e = b.unsqueeze(0).expand_as(next_dist)
        proj = torch.zeros_like(next_dist)
        proj.scatter_add_(2, lo, next_dist * (up.float() - b_e))
        proj.scatter_add_(2, up, next_dist * (b_e - lo.float()))
        return proj


class VisualSAC(L.LightningModule):
    """一个 minibatch 的 SAC 更新：评论家(C51 交叉熵) → 温度 α → 演员(均值 Q) → 软更新目标网络。

    骨架和 v5 `SAC`（随机策略＋最大熵＋自动温度）完全一致，唯二区别：评论家从 MSE
    换成 C51 交叉熵（`critic_loss` 那一段），以及输入多了个共享 `encoder` 把 RGB 编码
    成特征、和关节状态一起喂进 `Actor`/`C51TwinQ`（视觉部分只训一份，评论家的梯度
    顺带把它教好）。
    """

    def __init__(self, state_dim, action_dim, action_low, action_high, gamma=0.9, tau=0.01,
                 lr=3e-4, policy_frequency=1):
        super().__init__()
        self.automatic_optimization = False
        self.gamma, self.tau, self.lr = gamma, tau, lr
        self.policy_frequency = policy_frequency
        self.target_entropy = -float(action_dim)

        self.encoder = CNNEncoder()
        self.actor = Actor(self.encoder.repr_dim, state_dim, action_dim, action_low, action_high)
        self.critic = C51TwinQ(self.encoder.repr_dim, state_dim, action_dim)
        self.critic_target = C51TwinQ(self.encoder.repr_dim, state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.log_alpha = nn.Parameter(torch.zeros(()))

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def configure_optimizers(self):
        critic_params = list(self.encoder.parameters()) + list(self.critic.parameters())
        return (torch.optim.Adam(self.actor.parameters(), lr=self.lr),
                torch.optim.Adam(critic_params, lr=self.lr),
                torch.optim.Adam([self.log_alpha], lr=self.lr))

    @torch.no_grad()
    def sample_action(self, rgb, state):
        action, _, _ = self.actor.get_action(self.encoder(rgb), state)
        return action

    @torch.no_grad()
    def eval_action(self, rgb, state):
        _, _, det = self.actor.get_action(self.encoder(rgb), state)
        return det

    def training_step(self, batch, batch_idx):
        actor_opt, critic_opt, alpha_opt = self.optimizers()
        rgb, state, action, reward, next_rgb, next_state = batch

        # —— 评论家（C51 交叉熵）：熵项折进奖励，目标分布来自目标网络 ——（always bootstrap）
        with torch.no_grad():
            next_feat = self.encoder(next_rgb)
            next_action, next_logp, _ = self.actor.get_action(next_feat, next_state)
            rewards_ent = reward - self.gamma * self.alpha * next_logp.squeeze(-1)
            target_dist = self.critic_target.categorical_target(next_feat, next_state, next_action,
                                                                rewards_ent, self.gamma)
        feat = self.encoder(rgb)
        log_probs = F.log_softmax(self.critic.logits(feat, state, action), dim=-1)
        critic_loss = -(target_dist * log_probs).sum(-1).mean(-1).sum()  # 两个 Q 的交叉熵之和
        critic_opt.zero_grad(); self.manual_backward(critic_loss); critic_opt.step()

        # —— 温度 α：把熵自动调到目标熵 ——
        with torch.no_grad():
            _, logp, _ = self.actor.get_action(feat.detach(), state)
        alpha_loss = (-self.log_alpha.exp() * (logp + self.target_entropy)).mean()
        alpha_opt.zero_grad(); self.manual_backward(alpha_loss); alpha_opt.step()

        # —— 演员（最大化均值 Q − α·熵；编码器截断，只由评论家教）——
        if batch_idx % self.policy_frequency == 0:
            feat_d = feat.detach()
            pi, log_pi, _ = self.actor.get_action(feat_d, state)
            critic_value = self.critic.expected_q(feat_d, state, pi).mean(0)  # 两个 Q 取均值（No CDQ）
            actor_loss = (self.alpha.detach() * log_pi.squeeze(-1) - critic_value).mean()
            actor_opt.zero_grad(); self.manual_backward(actor_loss); actor_opt.step()

        # —— 目标网络软更新 ——
        with torch.no_grad():
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.mul_(1 - self.tau).add_(self.tau * p)

        self.log_dict({"critic_loss": critic_loss.detach(), "alpha": self.alpha.detach()},
                      prog_bar=True, on_step=True, on_epoch=False)


class ReplayBuffer:
    """定容经验回放池，整块开在 GPU 上；每步把所有并行环境的转移滚动写入。"""

    def __init__(self, capacity, image_size, state_dim, action_dim, device):
        z = lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt, device=device)  # noqa: E731
        self.rgb = z(capacity, image_size, image_size, 3, dt=torch.uint8)
        self.next_rgb = z(capacity, image_size, image_size, 3, dt=torch.uint8)
        self.state = z(capacity, state_dim)
        self.next_state = z(capacity, state_dim)
        self.action = z(capacity, action_dim)
        self.reward = z(capacity)
        self.capacity, self.device = capacity, device
        self.pos, self.full = 0, False

    def __len__(self):
        return self.capacity if self.full else self.pos

    def add(self, rgb, state, action, reward, next_rgb, next_state):
        n = rgb.shape[0]
        idx = (torch.arange(n, device=self.device) + self.pos) % self.capacity
        self.rgb[idx] = rgb.to(torch.uint8); self.next_rgb[idx] = next_rgb.to(torch.uint8)
        self.state[idx] = state; self.next_state[idx] = next_state
        self.action[idx] = action; self.reward[idx] = reward.float()
        self.pos = (self.pos + n) % self.capacity
        self.full = self.full or self.pos < n

    def sample(self, batch_size):
        i = torch.randint(0, len(self), (batch_size,), device=self.device)
        return self.rgb[i], self.state[i], self.action[i], self.reward[i], self.next_rgb[i], self.next_state[i]


class SO101SACData(L.LightningDataModule):
    """持有环境和回放池；每轮先采样、再把 minibatch 交给 Trainer。

    `env` 是 ManiSkill 标准 `ManiSkillVectorEnv`：不像旧版 `TrainEnv` 那样自己缓存
    `self.obs`，这里改由本类持有 `self.obs`，每次 `step` 后手动滚动到下一步。
    """

    def __init__(self, env, model, buffer, action_dim, steps_per_iter, updates_per_iter,
                batch_size, learning_starts):
        super().__init__()
        self.env, self.model, self.buffer = env, model, buffer
        self.action_dim = action_dim
        self.steps_per_iter, self.updates_per_iter = steps_per_iter, updates_per_iter
        self.batch_size, self.learning_starts = batch_size, learning_starts
        self.last_success = 0.0
        self.obs, _ = env.reset()

    def _collect(self, use_policy):
        rgb, state = self.obs["rgb"], self.obs["state"]
        if use_policy:
            action = self.model.sample_action(rgb, state)
        else:  # 预热：均匀随机动作把池子填起来
            low = torch.as_tensor(self.env.single_action_space.low, device=self.env.device)
            high = torch.as_tensor(self.env.single_action_space.high, device=self.env.device)
            action = low + (high - low) * torch.rand(self.env.num_envs, self.action_dim, device=self.env.device)
        next_obs, reward, _, _, info = self.env.step(action)
        self.buffer.add(rgb, state, action, reward, next_obs["rgb"], next_obs["state"])
        self.obs = next_obs
        return info["success"].float().mean().item()

    def train_dataloader(self):
        def gen():
            while len(self.buffer) < self.learning_starts:
                self._collect(use_policy=False)
            succ = [self._collect(use_policy=True) for _ in range(self.steps_per_iter)]
            self.last_success = float(np.mean(succ))
            for _ in range(self.updates_per_iter):
                yield self.buffer.sample(self.batch_size)

        class _DS(IterableDataset):
            def __iter__(self_inner):
                return gen()

        return DataLoader(_DS(), batch_size=None)


class SuccessLogger(L.Callback):
    """每轮打印采集成功率 + 定期存 ckpt（encoder+actor，供评测/复用）。"""

    def __init__(self, ckpt_dir, save_interval, max_iterations):
        self.ckpt_dir, self.save_interval, self.max_iterations = ckpt_dir, save_interval, max_iterations

    def on_train_epoch_end(self, trainer, pl_module):
        it = trainer.current_epoch + 1
        succ = trainer.datamodule.last_success
        print(f"  iter {it}: success_once={succ:.2f}  alpha={pl_module.alpha.item():.3f}", flush=True)
        if it % self.save_interval == 0 or it == self.max_iterations:
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"encoder": pl_module.encoder.state_dict(),
                        "actor": pl_module.actor.state_dict()}, self.ckpt_dir / "sac_ckpt.pt")


def run_training(task, num_envs, max_iterations, updates_per_iter, batch_size,
                 buffer_capacity, learning_starts, image_size, device, seed=1):
    torch.manual_seed(seed)
    env = so101_sim.visual_rl_env(task, num_envs=num_envs, image_size=image_size)
    action_dim = env.single_action_space.shape[-1]
    state_dim = env.observation_space["state"].shape[-1]
    low = torch.as_tensor(env.single_action_space.low, device=device)
    high = torch.as_tensor(env.single_action_space.high, device=device)
    model = VisualSAC(state_dim, action_dim, low, high).to(device)
    buffer = ReplayBuffer(buffer_capacity, image_size, state_dim, action_dim, device)
    data = SO101SACData(env, model, buffer, action_dim, steps_per_iter=1, updates_per_iter=updates_per_iter,
                        batch_size=batch_size, learning_starts=learning_starts)

    ckpt_dir = Path(os.environ["DATASETS_ROOT"]) / "models" / "trained" / "so101_sim_sac" / task
    trainer = L.Trainer(
        accelerator="gpu", devices=1, max_epochs=max_iterations,
        reload_dataloaders_every_n_epochs=1, enable_checkpointing=False, logger=False,
        enable_model_summary=False, enable_progress_bar=False, log_every_n_steps=10,
        callbacks=[SuccessLogger(ckpt_dir, save_interval=25, max_iterations=max_iterations)],
    )
    trainer.fit(model, datamodule=data)
    env.close()
    return ckpt_dir / "sac_ckpt.pt"


# 改这里选任务与训练时长，然后 `python train_v6_squint.py`。
TASK = "SO101PickPlaceCube40-v1"

if __name__ == "__main__":
    # 沿用 squint 的成熟配方：每步 256 次更新（UTD），批 512，回放池 50 万。
    run_training(
        task=TASK, num_envs=1024, max_iterations=200, updates_per_iter=256, batch_size=512,
        buffer_capacity=500_000, learning_starts=5_000, image_size=16, device="cuda",
    )
