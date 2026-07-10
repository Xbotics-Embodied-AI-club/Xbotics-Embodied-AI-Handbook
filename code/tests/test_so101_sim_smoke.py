"""so101_sim 冒烟测试：squint SO101 仿真接入 lerobot 的最小闭环自检。

覆盖 import → 注册 → make → reset/step → lerobot make_env 通路。需要 CUDA（ManiSkill GPU 后端）。
运行：`python -m pytest tests/test_so101_sim_smoke.py`（so101_sim 已 editable 安装，无需 PYTHONPATH）。
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mani_skill")

if not torch.cuda.is_available():
    pytest.skip("so101_sim 需要 CUDA（ManiSkill GPU 后端）", allow_module_level=True)

EXPECTED_TASKS = {
    "SO101ReachCube-v1",
    "SO101ReachCan-v1",
    "SO101LiftCube-v1",
    "SO101LiftCan-v1",
    "SO101PlaceCube-v1",
    "SO101PlaceCan-v1",
    "SO101StackCube-v1",
    "SO101StackCan-v1",
}


def test_import_registers_tasks_and_entrypoint():
    import so101_sim  # noqa: F401  导入即注册
    from gymnasium.envs.registration import registry
    from mani_skill.utils.registration import REGISTERED_ENVS

    assert "SO101Sim-v1" in registry
    assert EXPECTED_TASKS.issubset(set(REGISTERED_ENVS))


def test_env_make_reset_step_obs_format():
    import gymnasium as gym

    import so101_sim  # noqa: F401

    env = gym.make(
        "SO101Sim-v1",
        task="SO101ReachCube-v1",
        obs_type="pixels_agent_pos",
        observation_width=128,
        observation_height=128,
    )
    try:
        obs, info = env.reset(seed=0)
        assert info == {"is_success": False}
        assert obs["pixels"]["base_camera"].shape == (128, 128, 3)
        assert obs["agent_pos"].shape == (6,)
        assert env.action_space.shape == (6,)

        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert isinstance(reward, float)
        assert isinstance(info["is_success"], bool)
    finally:
        env.close()


def test_lerobot_make_env_generic_branch():
    """lerobot 的 make_env 走通用分支：import package_name → gym.make(gym_id)。"""
    from lerobot.envs.configs import So101SimEnv
    from lerobot.envs.factory import make_env

    cfg = So101SimEnv(task="SO101ReachCube-v1")
    assert cfg.type == "so101_sim"
    assert cfg.package_name == "so101_sim"
    assert cfg.gym_id == "SO101Sim-v1"

    envs = make_env(cfg, n_envs=1)
    vec = envs["so101_sim"][0]
    try:
        vec.reset(seed=0)
        vec.step(vec.action_space.sample())
    finally:
        vec.close()
