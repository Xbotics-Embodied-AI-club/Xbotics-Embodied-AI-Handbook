"""so101_kit 冒烟测试：KIT 机器人 + top/wrist 双相机接入的最小闭环自检。

覆盖 import → 注册 KIT 任务/机器人 → make → reset 出双相机 → STEP→mesh 资产已就位。
需要 CUDA（ManiSkill GPU 后端）。运行：`python -m pytest tests/test_so101_kit_smoke.py`。
"""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mani_skill")

if not torch.cuda.is_available():
    pytest.skip("so101_kit 需要 CUDA（ManiSkill GPU 后端）", allow_module_level=True)

KIT_TASKS = {"SO101KitReachCube-v1", "SO101KitPlaceCube-v1"}


def test_import_registers_kit_tasks_and_agent():
    import so101_sim  # noqa: F401  导入即注册
    from mani_skill.utils.registration import REGISTERED_ENVS
    from mani_skill.agents.registration import REGISTERED_AGENTS

    assert KIT_TASKS.issubset(set(REGISTERED_ENVS))
    assert "so101_kit" in REGISTERED_AGENTS


def test_kit_env_dual_camera_640x480_and_6dof():
    import gymnasium as gym

    import so101_sim  # noqa: F401

    env = gym.make(
        "SO101KitReachCube-v1",
        num_envs=1,
        obs_mode="rgb",
        render_mode="all",
        sim_backend="gpu",
        domain_randomization=False,
        sensor_configs=dict(width=640, height=480),
    )
    try:
        obs, info = env.reset(seed=0)
        sensors = obs["sensor_data"]
        assert {"top", "wrist"}.issubset(set(sensors.keys()))
        assert tuple(sensors["top"]["rgb"].shape) == (1, 480, 640, 3)
        assert tuple(sensors["wrist"]["rgb"].shape) == (1, 480, 640, 3)
        assert env.action_space.shape == (6,)

        robot = env.unwrapped.agent.robot
        assert [j.name for j in robot.active_joints] == [
            "shoulder_pan", "shoulder_lift", "elbow_flex",
            "wrist_flex", "wrist_roll", "gripper",
        ]
        assert "top_camera_optical_frame" in robot.links_map
        assert "wrist_camera_optical_frame" in robot.links_map
    finally:
        env.close()


def test_step_meshes_present():
    """convert_step.py 产物（4 件物体 visual+collision）已随包就位。"""
    import so101_sim

    objects = Path(so101_sim.__file__).parent / "robots" / "kit_assets" / "objects"
    for name in ("cube_2", "cube_4", "cylinder_4", "bin_2"):
        assert (objects / f"{name}_visual.glb").exists(), name
        assert (objects / f"{name}_collision.obj").exists(), name
