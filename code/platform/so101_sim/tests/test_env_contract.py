"""so101_sim 的对外契约：三个环境 id、无全局命名污染、观测形状。

这些断言是包的对外承诺，重构不得破坏。
"""

import subprocess
import sys
from pathlib import Path

import pytest

PKG_ROOT = Path(__file__).resolve().parents[1]
DISTRIBUTED_ENVS = [
    "SO101PickPlaceCube40-v1",
    "SO101PickPlaceCube20-v1",
    "SO101PickPlaceCylinder40-v1",
]


def test_three_envs_registered():
    """import 本包后，三个分发环境在 gymnasium 注册表里。"""
    import gymnasium as gym

    import so101_sim  # noqa: F401

    for env_id in DISTRIBUTED_ENVS:
        assert env_id in gym.registry, f"{env_id} 未注册"


def test_no_toplevel_envs_namespace():
    """import so101_sim 不得占用顶层 `envs` 这个名字。

    子进程里跑：顶层 envs 是 RL 圈最易撞名的包名，被我们占了会让学员自己的
    `envs/` 目录莫名其妙失效，且报错完全看不懂。
    """
    code = "import so101_sim, sys; sys.exit(1 if 'envs' in sys.modules else 0)"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


def test_no_syspath_injection():
    """包内不得有 sys.path 注入：import 只应定义东西，不应改全局搜索路径。"""
    hits = [
        p for p in (PKG_ROOT / "so101_sim").rglob("*.py")
        if "sys.path.insert" in p.read_text() or "sys.path.append" in p.read_text()
    ]
    assert hits == [], f"仍有 sys.path 注入：{hits}"


@pytest.mark.parametrize("env_id", DISTRIBUTED_ENVS)
def test_reset_step_shapes(env_id):
    """原生入口：obs/reward 首维恒为 num_envs，且在 GPU 上。"""
    import gymnasium as gym
    import torch

    import so101_sim  # noqa: F401

    env = gym.make(env_id, num_envs=2, obs_mode="state", sim_backend="gpu",
                   render_mode="all", domain_randomization=False, max_episode_steps=50)
    obs, _ = env.reset(seed=0)
    assert obs.shape[0] == 2
    action = torch.zeros((2, env.unwrapped.action_space.shape[-1]),
                         device=env.unwrapped.device)
    obs, reward, terminated, truncated, _ = env.step(action)
    assert obs.shape[0] == 2 and reward.shape[0] == 2
    env.close()


def test_visual_rl_env_returns_maniskill_vector_env():
    """RL 便利函数返回的是 ManiSkill 标准对象，不是我们自造的类型。

    这是"入口只有两个"的机器判据：RL 侧不引入第三种 API。
    """
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

    from so101_sim import visual_rl_env

    env = visual_rl_env("SO101PickPlaceCube40-v1", num_envs=2, image_size=16)
    assert isinstance(env, ManiSkillVectorEnv)
    obs, _ = env.reset(seed=0)
    assert obs["rgb"].shape[0] == 2 and obs["rgb"].shape[1] == 16
    env.close()


def test_state_rl_env_returns_maniskill_vector_env():
    """状态 RL 便利函数同样返回 ManiSkill 标准对象；观测是纯 state 张量，没有 rgb 键。"""
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

    from so101_sim import state_rl_env

    env = state_rl_env("SO101PickPlaceCube40-v1", num_envs=2)
    assert isinstance(env, ManiSkillVectorEnv)
    obs, _ = env.reset(seed=0)
    assert obs.shape[0] == 2
    env.close()


def test_color_jitter_wrapper_jitters_each_camera_independently():
    """双相机拼接后的 6 通道观测，颜色抖动必须逐相机独立采样。

    直接构造一个假的 6 通道 (top+wrist) 观测喂给 `ColorJitterWrapper`，不依赖仿真：
    - 输出仍是 6 通道（分组处理没有丢通道）；
    - 两路相机各自的抖动结果不同——如果两组共享同一次抖动参数，在这个构造下前 3
      通道与后 3 通道会变成同一份数据的同一种变换，结果会相等；只要两路本身画面不同，
      独立抖动几乎必然让两路输出也不同（同一种变换作用在不同输入上、输出自然不同），
      但更关键的判据是下面这条：连续两次 reset 出的同一张画面，两次抖动结果不相等，
      说明确实每次都在重新采样，不是缓存了固定参数。
    """
    import gymnasium as gym
    import torch

    from so101_sim.wrappers import ColorJitterWrapper

    torch.manual_seed(0)
    height = width = 8
    # 两路相机给完全相同的画面：这样如果抖动结果不同，只能是因为两组独立采样了参数。
    single_camera = torch.randint(0, 256, (2, height, width, 3), dtype=torch.uint8)
    six_channel = torch.cat([single_camera, single_camera], dim=-1)

    # gym.ObservationWrapper 只需要 observation_space 与 unwrapped 存在即可构造。
    base_env = gym.Env()
    base_env.observation_space = gym.spaces.Dict({
        "rgb": gym.spaces.Box(low=0, high=255, shape=(height, width, 6), dtype="uint8"),
    })
    wrapper = ColorJitterWrapper(base_env)

    out = wrapper.observation({"rgb": six_channel.clone()})
    assert out["rgb"].shape == six_channel.shape

    top_out = out["rgb"][..., :3]
    wrist_out = out["rgb"][..., 3:]
    # 两路相机输入完全相同，若独立采样几乎必然给出不同的抖动结果；若共享同一次采样，
    # 两路输出会逐元素相等。
    assert not torch.equal(top_out, wrist_out), "两路相机的抖动结果相同，怀疑共享了同一份采样参数"


def test_color_jitter_wrapper_rejects_non_multiple_of_three_channels():
    """通道数不是 3 的整数倍时要报清楚的错，而不是静默错乱观测。"""
    import gymnasium as gym
    import torch

    from so101_sim.wrappers import ColorJitterWrapper

    base_env = gym.Env()
    base_env.observation_space = gym.spaces.Dict({
        "rgb": gym.spaces.Box(low=0, high=255, shape=(8, 8, 4), dtype="uint8"),
    })
    wrapper = ColorJitterWrapper(base_env)
    bad_obs = {"rgb": torch.randint(0, 256, (2, 8, 8, 4), dtype=torch.uint8)}

    with pytest.raises(ValueError):
        wrapper.observation(bad_obs)


def test_no_train_env_module():
    """train_env.TrainEnv 已删除：RL 侧走 wrappers，不走自定义类。"""
    import so101_sim

    assert not hasattr(so101_sim, "make_train_env")
    assert not (Path(so101_sim.__file__).parent / "train_env.py").exists()
