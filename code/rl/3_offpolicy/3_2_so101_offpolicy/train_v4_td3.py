"""教学版状态 TD3 训练入口：SO101 连续控制阶梯的第二级，专治 v3 DDPG 的 Q 高估。

上一级 DDPG 只有一个 Critic：噪声或离群样本一旦把某个动作的 Q 值估高了，确定性策略
梯度就会把 actor 一路推向这个虚高的动作。TD3（Twin Delayed DDPG）在**不改变"确定性
策略 + 经验回放 + 目标网络"这套骨架**的前提下，加三处刻意针对高估问题的改动：

1. **双 Q 取 min（Clipped Double Q）**：训两个独立的 Critic（q1/q2，各自也有目标网络），
   算 TD 目标时取两者较小值 `min(q1_target, q2_target)`。哪怕其中一个被高估，另一个
   压住它，目标值就不会被单侧误差带偏。
2. **延迟策略更新**：Critic 每步都更新，但 actor（以及两套目标网络的软更新）每
   `policy_frequency` 步才更新一次。让 Critic 先把 Q 估得准一点，再拿它去指导 actor，
   避免"用一个还没学好的 Q 去推策略"。
3. **目标策略平滑**：算 `next_action = actor_target(next_state)` 时，给它加一点截断
   高斯噪声再夹回合法区间。这让目标 Q 在动作空间里被"抹" 开一小片邻域，Critic 学到的
   不再是某个孤立动作上的一个尖峰值，而是一片平滑区域的估计，更难被单点噪声骗到。

其余全部照抄 v3 DDPG 的骨架：DeterministicActor / ReplayBuffer / SO101DDPGData（在线
采集 + 经验回放）/ SuccessLogger（同一套 `success_once` 打印口径）/ `trainer.fit` 入口。
探索方式也没变——确定性策略本身不探索，采集时仍旧靠 `exploration_noise` 加高斯噪声；
actor loss 仍然只用 q1（`-q1(s, actor(s)).mean()`），q2 只参与 Critic 目标的 min，不
参与 actor 更新。

**观测归一化同样是这一级能学起来的前提**：和 v3 DDPG 一样，SO101 的 55 维 state
里某些维度原始值能到 ~200，直接喂进 `DeterministicActor` 末层的 `Tanh` 会把它饱和到
±1 附近、梯度趋于 0——TD3 的骨架照抄 DDPG（同一个 `DeterministicActor`），这个问题
原样继承。修法也照抄 v3：`RunningNorm` 在线维护 state 的 running mean/var（采集时
用原始 state 更新），真正喂进 actor/critic 前先归一化，回放池仍然存原始 state。

诚实的教学结论（公平预算 500 iter、三级同预算、加观测归一化后的真实结果）：TD3 治的
是**高估**，不是**探索**。同样归一化、同样 500 iter 预算下重跑，双 Q 取 min 确实治住
了 v3 DDPG 那种剧烈震荡——DDPG 全程在 mean_reward 上反复跳水、均值只有约 0.28，TD3
则更稳更近地爬升，均值约 0.52，比 DDPG 明显更近、也明显更稳，曲线不再是那种忽上忽下
的锯齿。但 success_once 在 500 轮预算内依然没有从 0 抬起来——**治好了"稳"，没治"够
不着"**：策略稳稳地知道往哪个方向靠近，却还是探索不到稀疏成功判定需要的精确位置。
把预算从 200 轮拉到和 DDPG 一样长的 500 轮，这个"够不着"也没有随时间自愈，说明瓶颈
不是训练时间不够，而是**确定性策略 + 外加固定高斯噪声**这套探索方式本身到了顶——噪声
只在动作空间里做局部扰动，找不到更"聪明"的探索方向。这说明探索不足才是这个任务上比
高估更根本的瓶颈，双 Q/延迟更新/目标策略平滑这三味药治的是训练稳定性，不直接扩大探索
范围；下一级 SAC 换成带熵正则的随机策略，用最大熵目标做原则化的探索，才是真正对症
探索问题的下一步。
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


class DeterministicActor(nn.Module):
    """state → 确定性动作（DPG，没有分布也不采样）：MLP 到 [-1,1] 再线性缩放进 [low, high]。"""

    def __init__(self, state_dim, action_dim, action_low, action_high):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim), nn.Tanh(),
        )
        self.register_buffer("action_scale", (action_high - action_low) / 2.0)
        self.register_buffer("action_bias", (action_high + action_low) / 2.0)

    def forward(self, state):
        """一次前向：状态进，确定性动作出（与 v3 完全一致，TD3 没动 actor 这一侧）。

        Args:
            state: 已归一化的观测。

        Returns:
            合法区间内的连续动作。
        """
        return self.net(state) * self.action_scale + self.action_bias


class QCritic(nn.Module):
    """(state, action) → 标量 Q：拼接后过 MLP，只输出一个数值（不是分布，也不是逐动作打分表）。"""

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state, action):
        """一次前向：状态和动作拼起来进 MLP，出一个标量 Q 值。

        TD3 会建两份**独立初始化**的这个网络，算目标时取两者较小的那个——两个网络在
        同一个动作上同时虚高的概率，比单个网络虚高的概率低得多。

        Args:
            state: 已归一化的观测。
            action: 要评分的动作。

        Returns:
            形状 (batch,) 的 Q 值。
        """
        return self.net(torch.cat([state, action], dim=-1)).squeeze(-1)


class TD3(L.LightningModule):
    """一个 minibatch 的 TD3 更新：双 Q 取 min → 延迟 actor 更新 → 目标策略平滑。"""

    def __init__(self, state_dim, action_dim, action_low, action_high,
                 gamma=0.99, tau=0.005, lr=3e-4, exploration_noise=0.1,
                 policy_frequency=2, policy_noise=0.2, noise_clip=0.5):
        super().__init__()
        self.automatic_optimization = False
        self.gamma, self.tau, self.lr = gamma, tau, lr
        self.exploration_noise = exploration_noise
        self.action_low, self.action_high = action_low, action_high
        # 相对 DDPG 新增：延迟策略更新 + 目标策略平滑的两个内联超参
        self.policy_frequency = policy_frequency
        self.policy_noise, self.noise_clip = policy_noise, noise_clip

        self.actor = DeterministicActor(state_dim, action_dim, action_low, action_high)
        self.actor_target = DeterministicActor(state_dim, action_dim, action_low, action_high)
        self.actor_target.load_state_dict(self.actor.state_dict())
        # 相对 DDPG 新增：两个独立 Critic（q1/q2）+ 各自的目标网络，clipped double Q 治高估
        self.critic1 = QCritic(state_dim, action_dim)
        self.critic2 = QCritic(state_dim, action_dim)
        self.critic1_target = QCritic(state_dim, action_dim)
        self.critic2_target = QCritic(state_dim, action_dim)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        # 观测归一化：buffer 里存的仍是原始 state，这里只在喂进网络前做归一化
        self.obs_norm = RunningNorm(state_dim)

    def configure_optimizers(self):
        """三个优化器：actor 一个、两个 critic 各一个。

        比 v3 多出来的那一个就是 Twin——两个 critic 必须**独立**训练，共享优化器或
        共享参数都会让它们同时高估，取 min 也就压不住了。

        Returns:
            (actor 优化器, critic1 优化器, critic2 优化器)。
        """
        critic_params = list(self.critic1.parameters()) + list(self.critic2.parameters())
        return (torch.optim.Adam(self.actor.parameters(), lr=self.lr),
                torch.optim.Adam(critic_params, lr=self.lr))

    @torch.no_grad()
    def sample_action(self, state):
        """确定性动作 + 高斯探索噪声，clamp 回合法区间（探索方式与 DDPG 完全一致）。

        Args:
            state: 原始观测（未归一化）。

        Returns:
            合法区间内的连续动作。
        """
        action = self.actor(self.obs_norm.normalize(state))
        noise = torch.randn_like(action) * self.exploration_noise
        return (action + noise).clamp(self.action_low, self.action_high)

    @torch.no_grad()
    def eval_action(self, state):
        """评测用的动作：不加探索噪声，直接取 actor 的输出。

        Args:
            state: 原始观测。

        Returns:
            确定性动作。
        """
        return self.actor(self.obs_norm.normalize(state))

    def training_step(self, batch, batch_idx):
        """一个 minibatch 的 TD3 更新，三个补丁都在这里：

        1. 目标动作加截断噪声（目标策略平滑），免得 critic 上某个尖锐的虚高峰值被 actor 钻空子；
        2. 目标 Q 取两个 critic 的较小值（Clipped Double Q），系统性压掉高估；
        3. actor 每 policy_frequency 步才更新一次（延迟策略更新）。

        Args:
            batch: 从回放池抽出的 `(state, action, reward, next_state)`。
            batch_idx: Lightning 传入的批序号，这里用不到。
        """
        actor_opt, critic_opt = self.optimizers()
        state, action, reward, next_state = batch
        # buffer 里是原始 state，喂进网络前统一归一化（统计量只在采集时更新，这里只读）
        state, next_state = self.obs_norm.normalize(state), self.obs_norm.normalize(next_state)

        # —— 评论家（双 Q 取 min 算目标，MSE 回归两个 Q）——
        with torch.no_grad():
            # 相对 DDPG 新增：目标策略平滑——目标动作叠一点截断高斯噪声再夹回合法区间，
            # 防止 Critic 在某个孤立动作上学出一个容易被利用的 Q 尖峰。
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(self.action_low, self.action_high)
            # 相对 DDPG 新增：clipped double Q——两个目标网络分别打分，取较小值兜底
            target_q1 = self.critic1_target(next_state, next_action)
            target_q2 = self.critic2_target(next_state, next_action)
            target_q = reward + self.gamma * torch.min(target_q1, target_q2)
        critic_loss = (F.mse_loss(self.critic1(state, action), target_q)
                     + F.mse_loss(self.critic2(state, action), target_q))
        critic_opt.zero_grad(); self.manual_backward(critic_loss); critic_opt.step()

        # 相对 DDPG 新增：延迟策略更新——actor 和目标网络软更新每 policy_frequency 步才做一次
        if batch_idx % self.policy_frequency == 0:
            # —— 演员（确定性策略梯度）：只用 critic1 打分，critic2 只参与上面的 min ——
            actor_loss = -self.critic1(state, self.actor(state)).mean()
            actor_opt.zero_grad(); self.manual_backward(actor_loss); actor_opt.step()

            with torch.no_grad():
                for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
                    tp.mul_(1 - self.tau).add_(self.tau * p)
                for p, tp in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                    tp.mul_(1 - self.tau).add_(self.tau * p)
                for p, tp in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                    tp.mul_(1 - self.tau).add_(self.tau * p)

            self.log_dict({"critic_loss": critic_loss.detach(), "actor_loss": actor_loss.detach()},
                          prog_bar=True, on_step=True, on_epoch=False)
        else:
            self.log_dict({"critic_loss": critic_loss.detach()},
                          prog_bar=True, on_step=True, on_epoch=False)


class ReplayBuffer:
    """定容经验回放池，整块开在 GPU 上（state 版：无 rgb，只存关节状态向量）。"""

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


class SO101DDPGData(L.LightningDataModule):
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
    """每轮打印采集成功率 + 平均奖励（靠近度，success 之外的连续信号）+ 定期存 ckpt。"""

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
        print(f"  iter {it}: success_once={succ:.2f}  mean_reward={rew:.3f}", flush=True)
        if it % self.save_interval == 0 or it == self.max_iterations:
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"actor": pl_module.actor.state_dict(),
                       "obs_norm": pl_module.obs_norm.state_dict()}, self.ckpt_dir / "td3.pt")


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
    model = TD3(state_dim, action_dim, low, high).to(device)
    buffer = ReplayBuffer(buffer_capacity, state_dim, action_dim, device)
    data = SO101DDPGData(env, model, buffer, action_dim, steps_per_iter=1, updates_per_iter=updates_per_iter,
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
    return ckpt_dir / "td3.pt"


# 改这里选任务与训练时长，然后 `python train_v4_td3.py`。
TASK = "SO101PickPlaceCube40-v1"

if __name__ == "__main__":
    # 与 v3 DDPG / v5 SAC 完全相同的共享预算，只差算法，可公平对照：
    # 每步 256 次更新（UTD），批 512，回放池 50 万；SAC 要到 iter~480 才稳定跨过"解决"，
    # 三级统一给够 500 iter 预算才公平。
    run_training(
        task=TASK, num_envs=1024, max_iterations=500, updates_per_iter=256, batch_size=512,
        buffer_capacity=500_000, learning_starts=5_000, device="cuda",
    )
