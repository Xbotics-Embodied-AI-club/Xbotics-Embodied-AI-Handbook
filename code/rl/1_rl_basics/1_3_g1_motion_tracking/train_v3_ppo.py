"""动作跟随任务上的第三版：完整 PPO，也是本组两条任务线共同的收官版本。

三块让训练稳住的机制一个不少：概率比值裁剪（clip）、同一段 rollout 切成多个
minibatch 反复训几轮、按 KL 散度自适应调整学习率，外加对 value 估值的对称裁剪。
行走线的同名脚本证明了这套配方在简单任务上就已经领先；换到"逐帧贴住一段参考动作"
这种更难的任务上，v1/v2 明显跟不动，这套机制的必要性才真正显出来。

讲义对应：第14讲 6.4 节（代码走读）与 6.6 节（实验加码）。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterator

import lightning as L
import torch
import wandb
from torch import nn
from torch.utils.data import DataLoader, IterableDataset

# 与本组其余脚本一致：按脚本自身所在目录取同级模块，直接 `python xxx.py` 就能跑。
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env import BeyondMimicEnv  # noqa: E402
from motion import MotionClip  # noqa: E402


def default_checkpoint_root(run_name: str) -> Path:
    """给出这次训练存权重的目录。

    权重是大文件，按仓库约定落在共享数据根下、不进 git，所以路径从环境变量拼出来。

    Args:
        run_name: 本次训练的名字，同时用作目录名。

    Returns:
        存放 checkpoint 的目录路径。
    """
    datasets_root = Path(os.environ["DATASETS_ROOT"])
    return datasets_root / "models" / "trained" / "xbotics_rl_beyondmimic" / run_name


def compute_gae(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """用广义优势估计（GAE）把一段 rollout 换算成每步的优势与回报目标。

    Args:
        rewards: 形状 (T, N) 的即时奖励。
        dones: 形状 (T, N)，回合在该步结束则为 1。
        values: 形状 (T, N, 1) 的 critic 估值。
        next_value: 形状 (N, 1)，最后一步之后那个状态的估值。
        gamma: 折扣因子。
        lam: GAE 系数。

    Returns:
        (advantages, returns)。
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


def adapt_learning_rate_to_kl(
    optimizer: torch.optim.Optimizer,
    kl: torch.Tensor,
    desired_kl: float,
    min_lr: float = 1.0e-5,
    max_lr: float = 1.0e-2,
) -> float:
    """按新旧策略的 KL 距离自动收放学习率。

    clip 只让「走太远」无利可图，并不真的锁住步长；再加这一层兜底。

    Args:
        optimizer: 要调整的优化器。
        kl: 本次更新前后策略的 KL 距离。
        desired_kl: 希望维持的 KL 水平。
        min_lr: 学习率下限。
        max_lr: 学习率上限。

    Returns:
        调整后的学习率。
    """
    current_lr = optimizer.param_groups[0]["lr"]
    kl_value = float(kl.detach().cpu())
    if kl_value > 2.0 * desired_kl:
        current_lr = max(min_lr, current_lr / 1.5)
    elif 0.0 < kl_value < 0.5 * desired_kl:
        current_lr = min(max_lr, current_lr * 1.5)
    optimizer.param_groups[0]["lr"] = current_lr
    return current_lr


def describe_motion_dataset(motion_file: Path) -> dict[str, str | int | float | list[float]]:
    """读一遍参考动作文件，汇总成一份可打印的概况。

    Args:
        motion_file: 参考动作 npz 路径。

    Returns:
        含帧数、帧率、时长与几帧关节角样本的字典。
    """
    motion = MotionClip.load(motion_file, device="cpu")
    middle_frame = motion.num_frames // 2
    return {
        "motion_file": str(motion_file),
        "frames": motion.num_frames,
        "fps": motion.fps,
        "duration_s": round(motion.duration_s, 2),
        "joint_dim": int(motion.joint_pos.shape[1]),
        "body_count": int(motion.body_pos_w.shape[1]),
        "first_joint_pos": motion.joint_snapshot(0),
        "middle_joint_pos": motion.joint_snapshot(middle_frame),
        "last_joint_pos": motion.joint_snapshot(motion.num_frames - 1),
    }


def print_motion_dataset_preview(motion_file: Path) -> None:
    """把参考动作的概况打到终端，开训前肉眼确认数据没读错。

    Args:
        motion_file: 参考动作 npz 路径。
    """
    preview = describe_motion_dataset(motion_file)
    print("Motion dataset preview")
    for key, value in preview.items():
        print(f"{key}: {value}")


def save_checkpoint(
    path: Path,
    model: "ActorCritic",
    optimizer: torch.optim.Optimizer,
    iteration: int,
    training_settings: dict[str, str | int | float],
) -> None:
    """存一份可续训、也可直接拿去评测的权重。

    Args:
        path: 目标文件路径。
        model: 要保存的模型。
        optimizer: 当前优化器。
        iteration: 当前是第几次迭代。
        training_settings: 本次训练的全部设置。
    """
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


class BeyondMimicRolloutDataset(IterableDataset):
    """在线数据集：每轮先从仿真器采一段轨迹，再把它切成若干 minibatch 逐个交给训练循环。"""

    def __init__(
        self,
        env,
        model: "ActorCritic",
        num_steps_per_env: int,
        gamma: float,
        lam: float,
        num_learning_epochs: int,
        num_mini_batches: int,
    ) -> None:
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

    def sample_rollout(self) -> dict[str, torch.Tensor]:
        """用当前策略采一段轨迹，并记下旧策略的对数概率与分布参数。

        这些量必须在采样时就存下来：训练阶段要拿它们算概率比值和 KL，那时策略已经变了。

        Returns:
            含整段 rollout 与 GAE 优势的字典。
        """
        env = self.env
        num_envs = env.num_envs
        # 从持久环境读出当前观测，沿用上一轮训练后的仿真状态。
        obs, critic_obs = env.get_observations()
        obs_steps = torch.zeros(self.num_steps_per_env, num_envs, obs.shape[1], device=env.device)
        critic_obs_steps = torch.zeros(self.num_steps_per_env, num_envs, critic_obs.shape[1], device=env.device)
        actions_steps = torch.zeros(self.num_steps_per_env, num_envs, self.model.action_dim, device=env.device)
        log_prob_steps = torch.zeros(self.num_steps_per_env, num_envs, device=env.device)
        value_steps = torch.zeros(self.num_steps_per_env, num_envs, 1, device=env.device)
        reward_steps = torch.zeros(self.num_steps_per_env, num_envs, device=env.device)
        done_steps = torch.zeros(self.num_steps_per_env, num_envs, device=env.device)
        action_mean_steps = torch.zeros(self.num_steps_per_env, num_envs, self.model.action_dim, device=env.device)
        action_std_steps = torch.zeros(self.num_steps_per_env, num_envs, self.model.action_dim, device=env.device)
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
        # advantage 在整段 rollout 上只归一化一次，再切 minibatch。
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
            "reward_mean": torch.tensor(reward_sum / self.num_steps_per_env, device=env.device),
        }

    def iter_mini_batches(self, rollout: dict[str, torch.Tensor]) -> Iterator[dict[str, torch.Tensor]]:
        """把一整段 rollout 打散成若干 minibatch，重复吐几轮。

        PPO 的数据复用就实现在这里，Lightning 那边一行不用改。

        Args:
            rollout: `sample_rollout` 的产出。

        Yields:
            一个个 minibatch 字典。
        """
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


class BeyondMimicData(L.LightningDataModule):
    """LightningDataModule 持有持久仿真环境，并在每轮把新的采样数据集交给 Trainer。"""

    def __init__(
        self,
        motion_file: Path,
        num_envs: int,
        num_steps_per_env: int,
        device: str,
        seed: int,
        gamma: float,
        lam: float,
        num_learning_epochs: int,
        num_mini_batches: int,
    ) -> None:
        super().__init__()
        self.motion_file = motion_file
        self.num_envs = num_envs
        self.num_steps_per_env = num_steps_per_env
        self.device = device
        self.seed = seed
        self.gamma = gamma
        self.lam = lam
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.model = None
        self.env = None

    def attach_model(self, model: "ActorCritic") -> None:
        """把模型挂到数据模块上。

        模型要到 Trainer 起来之后才存在，所以数据模块先建、模型后挂。

        Args:
            model: 采样时要用的策略网络。
        """
        self.model = model

    def setup(self, stage: str) -> None:
        """训练开始前建好权重目录并起一个 W&B run。

        Args:
            stage: Lightning 传入的阶段名，只在 fit 阶段做事。
        """
        if stage != "fit" or self.env is not None:
            return
        torch.manual_seed(self.seed)
        self.env = BeyondMimicEnv(self.motion_file, num_envs=self.num_envs, device=self.device, seed=self.seed)
        self.model.to(self.env.device)
        self.env.reset()

    def train_dataloader(self) -> DataLoader:
        # 每个 epoch 重新构造数据集 = 用刚更新过的策略采一段全新 rollout。
        """每个 epoch 重建一次数据集，保证数据永远来自当前策略。

        Returns:
            包着在线数据集的 DataLoader。
        """
        dataset = BeyondMimicRolloutDataset(
            env=self.env,
            model=self.model,
            num_steps_per_env=self.num_steps_per_env,
            gamma=self.gamma,
            lam=self.lam,
            num_learning_epochs=self.num_learning_epochs,
            num_mini_batches=self.num_mini_batches,
        )
        return DataLoader(dataset, batch_size=None)


class ActorCritic(nn.Module):
    """标准 PyTorch 模型：actor 输出动作分布，critic 估计状态价值。"""

    def __init__(
        self,
        obs_dim: int,
        critic_obs_dim: int,
        action_dim: int,
        actor_hidden_dims: tuple[int, ...] = (512, 256, 128),
        critic_hidden_dims: tuple[int, ...] = (512, 256, 128),
        init_noise_std: float = 1.0,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.normalizer_epsilon = 1.0e-2
        self.register_buffer("actor_mean", torch.zeros(obs_dim, dtype=torch.float32))
        self.register_buffer("actor_var", torch.ones(obs_dim, dtype=torch.float32))
        self.register_buffer("actor_count", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("critic_mean", torch.zeros(critic_obs_dim, dtype=torch.float32))
        self.register_buffer("critic_var", torch.ones(critic_obs_dim, dtype=torch.float32))
        self.register_buffer("critic_count", torch.tensor(0.0, dtype=torch.float32))
        self.actor = self.build_mlp(obs_dim, actor_hidden_dims, action_dim)
        self.critic = self.build_mlp(critic_obs_dim, critic_hidden_dims, 1)
        self.std = nn.Parameter(torch.full((action_dim,), init_noise_std))

    @staticmethod
    def build_mlp(input_dim: int, hidden_dims: tuple[int, ...], output_dim: int) -> nn.Sequential:
        """堆一串「线性层 + ELU」，最后一层不带激活。

        Args:
            input_dim: 输入维度。
            hidden_dims: 各隐藏层宽度。
            output_dim: 输出维度。

        Returns:
            拼好的 `nn.Sequential`。
        """
        layers: list[nn.Module] = []
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ELU())
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, output_dim))
        return nn.Sequential(*layers)

    @torch.no_grad()
    def update_actor_normalizer(self, obs: torch.Tensor) -> None:
        """用新采到的观测更新 actor 侧归一化统计量。

        Args:
            obs: 一批 actor 观测。
        """
        self.actor_mean, self.actor_var, self.actor_count = self.update_running_stats(
            obs,
            self.actor_mean,
            self.actor_var,
            self.actor_count,
        )

    @torch.no_grad()
    def update_critic_normalizer(self, critic_obs: torch.Tensor) -> None:
        """同上，更新的是 critic 侧统计量。

        Args:
            obs: 一批 critic 观测。
        """
        self.critic_mean, self.critic_var, self.critic_count = self.update_running_stats(
            critic_obs,
            self.critic_mean,
            self.critic_var,
            self.critic_count,
        )

    def update_running_stats(
        self,
        x: torch.Tensor,
        mean: torch.Tensor,
        var: torch.Tensor,
        count: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """按新一批样本增量更新均值与方差。

        Args:
            x: 一批观测。
            mean: 当前均值。
            var: 当前方差。
            count: 已累计样本数。

        Returns:
            (新均值, 新方差, 新计数)。
        """
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = torch.tensor(float(x.shape[0]), device=x.device)
        new_count = count + batch_count
        rate = batch_count / new_count
        delta = batch_mean - mean
        new_mean = mean + rate * delta
        new_var = var + rate * (batch_var - var + delta * (batch_mean - new_mean))
        return new_mean, new_var, new_count

    def normalize_actor(self, obs: torch.Tensor) -> torch.Tensor:
        """把 actor 观测标准化。

        Args:
            obs: 原始观测。

        Returns:
            标准化后的观测。
        """
        return (obs - self.actor_mean) / (torch.sqrt(self.actor_var) + self.normalizer_epsilon)

    def normalize_critic(self, critic_obs: torch.Tensor) -> torch.Tensor:
        """把 critic 观测标准化。

        Args:
            obs: 原始观测。

        Returns:
            标准化后的观测。
        """
        return (critic_obs - self.critic_mean) / (torch.sqrt(self.critic_var) + self.normalizer_epsilon)

    def action_distribution(self, obs: torch.Tensor) -> torch.distributions.Normal:
        """给出当前状态下的动作分布。

        Args:
            obs: 一批 actor 观测。

        Returns:
            逐关节独立的高斯分布。
        """
        mean = self.actor(self.normalize_actor(obs))
        std = self.std.clamp_min(1.0e-6).expand_as(mean)
        return torch.distributions.Normal(mean, std)

    def value(self, critic_obs: torch.Tensor) -> torch.Tensor:
        """critic 给状态打分。

        Args:
            critic_obs: 一批 critic 观测。

        Returns:
            形状 (N, 1) 的状态价值估计。
        """
        return self.critic(self.normalize_critic(critic_obs))

    def act(
        self,
        obs: torch.Tensor,
        critic_obs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """采样阶段用：采一个动作，并带回训练要用的量。

        Args:
            obs: actor 观测。
            critic_obs: critic 观测。

        Returns:
            (动作, 对数概率, 状态价值, 分布均值, 分布标准差)。
        """
        distribution = self.action_distribution(obs)
        actions = distribution.sample()
        log_prob = distribution.log_prob(actions).sum(dim=-1)
        value = self.value(critic_obs)
        return actions, log_prob, value, distribution.mean, distribution.stddev

    def evaluate(
        self,
        obs: torch.Tensor,
        critic_obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """训练阶段用：拿当前策略重新评估一批已采好的动作。

        Args:
            obs: actor 观测。
            critic_obs: critic 观测。
            actions: 采样阶段执行过的动作。

        Returns:
            (对数概率, 熵, 状态价值, 分布均值)。
        """
        distribution = self.action_distribution(obs)
        log_prob = distribution.log_prob(actions).sum(dim=-1)
        entropy = distribution.entropy().sum(dim=-1)
        value = self.value(critic_obs)
        return log_prob, entropy, value, distribution.mean

    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        """部署 / 评测时用：直接取分布均值。

        Args:
            obs: actor 观测。

        Returns:
            确定性动作。
        """
        return self.actor(self.normalize_actor(obs))


class BeyondMimicLightningPPO(L.LightningModule):
    """LightningModule 只负责一个 minibatch 的 PPO loss、记录与保存；更新循环交给 Lightning。"""

    def __init__(
        self,
        model: ActorCritic,
        run_name: str,
        max_iterations: int,
        save_interval: int,
        checkpoint_dir: Path,
        training_settings: dict[str, str | int | float],
        wandb_project: str,
        wandb_mode: str,
    ) -> None:
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
        self.entropy_coef = 0.005
        self.learning_rate = 1.0e-3
        self.desired_kl = 0.01

    def setup(self, stage: str) -> None:
        """训练开始前建好权重目录并起一个 W&B run。

        Args:
            stage: Lightning 传入的阶段名，只在 fit 阶段做事。
        """
        if stage != "fit":
            return

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.wandb_run = wandb.init(
            project=self.wandb_project,
            name=self.run_name,
            mode=self.wandb_mode,
            dir=self.checkpoint_dir.as_posix(),
            config={**self.training_settings, "trainer": "lightning"},
        )

    def configure_optimizers(self) -> torch.optim.Optimizer:
        # 保留对原始 optimizer 的引用，供 KL 自适应学习率和 checkpoint 直接使用。
        """把全部参数交给一个 Adam。

        Returns:
            Adam 优化器。
        """
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return self.optimizer

    def on_train_epoch_start(self) -> None:
        """每轮开始时清空本轮的指标缓存。"""
        self.epoch_records = []

    def training_step(self, mini_batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """算一个 minibatch 的 PPO loss，并把指标攒进本轮缓存。

        Args:
            mini_batch: 数据集吐出的一个 minibatch。
            batch_idx: Lightning 传入的批序号，本实现用不到。

        Returns:
            本步的 loss。
        """
        loss, policy_loss, value_loss, entropy_loss, kl = self.loss_for_batch(mini_batch)
        # 把本 minibatch 的 KL 暂存，交给 on_before_optimizer_step 调学习率。
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

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        # 原生 hook：反向之后、optimizer.step 之前，按 KL 调整学习率。
        """每次真正迈步之前，先按 KL 把学习率调好。

        Args:
            optimizer: Lightning 传入的优化器。
        """
        adapt_learning_rate_to_kl(self.optimizer, self.latest_kl, self.desired_kl)

    def on_train_epoch_end(self) -> None:
        """把本轮各 minibatch 的指标平均后上报，并按需存盘。"""
        iteration = self.current_epoch + 1
        metrics = {key: sum(r[key] for r in self.epoch_records) / len(self.epoch_records) for key in self.epoch_records[0]}
        metrics["lr"] = self.optimizer.param_groups[0]["lr"]
        wandb.log(metrics, step=iteration)

        if iteration % self.save_interval == 0 or iteration == self.max_iterations:
            self.latest_checkpoint = self.checkpoint_dir / f"model_{iteration}.pt"
            save_checkpoint(self.latest_checkpoint, self.model, self.optimizer, iteration, self.training_settings)

    def loss_for_batch(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """算一个 minibatch 上的三项 loss 与 KL。

        KL 在 `no_grad` 下算：它只用来调学习率，不参与反向传播。

        Args:
            batch: 一个 minibatch。

        Returns:
            (总 loss, 策略 loss, 价值 loss, 熵, KL)。
        """
        log_probs, entropy, values, action_means = self.model.evaluate(
            batch["obs"],
            batch["critic_obs"],
            batch["actions"],
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
        clipped = (torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                   * batch["advantages"])
        policy_loss = -torch.min(unclipped, clipped).mean()
        value_loss = self.value_loss(values, batch["values"], batch["returns"])
        entropy_loss = entropy.mean()
        loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_loss
        return loss, policy_loss, value_loss, entropy_loss, kl

    def value_loss(self, values: torch.Tensor, old_values: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
        """critic 的回归损失，可选对新估值做对称裁剪。

        Args:
            values: 当前 critic 的估值。
            old_values: 采样时记下的旧估值。
            returns: 回归目标。

        Returns:
            标量损失。
        """
        if not self.use_clipped_value_loss:
            return torch.square(values - returns).mean()

        value_clipped = old_values + (values - old_values).clamp(-self.clip_param, self.clip_param)
        return torch.max(torch.square(values - returns), torch.square(value_clipped - returns)).mean()

    def teardown(self, stage: str) -> None:
        """训练结束后收尾，把 W&B run 正常关掉。

        Args:
            stage: Lightning 传入的阶段名。
        """
        if self.wandb_run is not None:
            self.wandb_run.finish()


def run_training(
    motion_file: Path,
    run_name: str,
    num_envs: int,
    max_iterations: int,
    num_steps_per_env: int,
    save_interval: int,
    device: str,
    seed: int = 1,
    checkpoint_dir: Path | None = None,
    wandb_project: str = "rl_class",
    wandb_mode: str = "online",
) -> Path:
    """起一次完整训练：建环境、建模型、交给 Trainer 跑完。

    Args:
        motion_file: 要跟随的参考动作文件。
        run_name: 本次训练的名字，用作权重目录名与 W&B run 名。
        num_envs: 并行环境数。
        max_iterations: 训练迭代次数。
        num_steps_per_env: 每个环境每轮采多少步。
        save_interval: 每多少次迭代存一次权重。
        device: 仿真与训练所在设备。
        seed: 随机种子。
        checkpoint_dir: 权重目录，给 None 时按 run_name 自动拼。
        wandb_project: W&B 项目名。
        wandb_mode: W&B 模式。

    Returns:
        最后一次存盘的 checkpoint 路径。
    """
    checkpoint_dir = checkpoint_dir or default_checkpoint_root(run_name)
    gamma = 0.99
    lam = 0.95
    num_learning_epochs = 5
    num_mini_batches = 4
    training_settings = {
        "motion_file": str(motion_file),
        "run_name": run_name,
        "num_envs": num_envs,
        "max_iterations": max_iterations,
        "num_steps_per_env": num_steps_per_env,
        "save_interval": save_interval,
        "device": device,
        "seed": seed,
        "checkpoint_dir": str(checkpoint_dir),
        "wandb_project": wandb_project,
        "wandb_mode": wandb_mode,
        "gamma": gamma,
        "lam": lam,
        "num_learning_epochs": num_learning_epochs,
        "num_mini_batches": num_mini_batches,
    }

    torch.manual_seed(seed)
    data = BeyondMimicData(
        motion_file=motion_file,
        num_envs=num_envs,
        num_steps_per_env=num_steps_per_env,
        device=device,
        seed=seed,
        gamma=gamma,
        lam=lam,
        num_learning_epochs=num_learning_epochs,
        num_mini_batches=num_mini_batches,
    )
    policy = ActorCritic(obs_dim=160, critic_obs_dim=286, action_dim=29)
    data.attach_model(policy)
    model = BeyondMimicLightningPPO(
        model=policy,
        run_name=run_name,
        max_iterations=max_iterations,
        save_interval=save_interval,
        checkpoint_dir=checkpoint_dir,
        training_settings=training_settings,
        wandb_project=wandb_project,
        wandb_mode=wandb_mode,
    )
    # 一个 epoch = 一次 PPO iteration：每轮重载 dataloader 采新 rollout，梯度裁剪由 Trainer 负责。
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


def main() -> None:
    """按本模块的默认设置跑一次完整 PPO 训练。"""
    group_root = Path(__file__).resolve().parents[1]

    # 主要修改这一段：数据、轮数、环境数和 W&B 模式。
    motion_file = group_root / "data/g1_reference_motions/marshal-arts.npz"
    run_name = "beyondmimic-marshal-arts-lightning-10000"
    num_envs = 4096
    max_iterations = 10000
    num_steps_per_env = 24
    save_interval = 1000
    device = "cuda:0"
    seed = 1
    checkpoint_dir = default_checkpoint_root(run_name)
    wandb_project = "rl_class"
    wandb_mode = "online"

    print_motion_dataset_preview(motion_file)

    run_training(
        motion_file=motion_file,
        run_name=run_name,
        num_envs=num_envs,
        max_iterations=max_iterations,
        num_steps_per_env=num_steps_per_env,
        save_interval=save_interval,
        device=device,
        seed=seed,
        checkpoint_dir=checkpoint_dir,
        wandb_project=wandb_project,
        wandb_mode=wandb_mode,
    )


if __name__ == "__main__":
    main()
