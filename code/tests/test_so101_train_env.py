"""so101_sim 训练环境冒烟测试：`make_train_env` 的 visual / state 两种 obs_mode 各跑一步。

覆盖：reset/step 闭环、obs 形状与 dtype、state 模式下确实没有 rgb 键。
需要 CUDA（ManiSkill GPU 后端）。运行：`uv run python -m pytest tests/test_so101_train_env.py -v`
（so101_sim 已 editable 安装，无需 PYTHONPATH）。
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mani_skill")

if not torch.cuda.is_available():
    pytest.skip("so101_sim 训练环境需要 CUDA（ManiSkill GPU 后端）", allow_module_level=True)


def test_visual_obs_mode_reset_and_step():
    import so101_sim

    env = so101_sim.make_train_env("SO101ReachCube-v1", num_envs=4, obs_mode="visual")
    try:
        obs = env.reset()
        assert obs["rgb"].shape == (4, 16, 16, 3)
        assert obs["rgb"].dtype == torch.uint8
        assert obs["state"].shape[0] == 4

        action = torch.zeros(4, env.action_dim, device=env.device)
        next_obs, reward, terminated, truncated, success = env.step(action)
        assert next_obs["rgb"].shape == (4, 16, 16, 3)
        assert next_obs["rgb"].dtype == torch.uint8
        assert next_obs["state"].shape[0] == 4
        assert success.dtype == torch.bool
    finally:
        env.close()


def test_state_obs_mode_reset_and_step_has_no_rgb():
    import so101_sim

    env = so101_sim.make_train_env("SO101ReachCube-v1", num_envs=4, obs_mode="state")
    try:
        obs = env.reset()
        assert obs["state"].shape[0] == 4
        assert "rgb" not in obs

        action = torch.zeros(4, env.action_dim, device=env.device)
        next_obs, reward, terminated, truncated, success = env.step(action)
        assert next_obs["state"].shape[0] == 4
        assert "rgb" not in next_obs
    finally:
        env.close()
