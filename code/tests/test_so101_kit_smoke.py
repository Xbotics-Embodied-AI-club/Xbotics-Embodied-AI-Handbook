"""so101_sim 冒烟测试：机器人 + top/wrist 双相机接入的最小闭环自检。

覆盖 import → 注册三个分发场景与机器人 → make → reset 出双相机 → STEP→mesh 资产已就位。
需要 CUDA（ManiSkill GPU 后端）。运行：`python -m pytest tests/test_so101_kit_smoke.py`。

★ 这里只认**唯一那套**机器人与几何。早先本文件断言 `SO101KitReachCube-v1` /
`SO101KitPlaceCube-v1` 两个环境和 `so101_kit` 这个 agent 存在 —— 那是「KIT 版」与
「裸臂版」并存时期的接口，两套几何对同一批关节给出不同限位，而取到哪一套取决于注册了
哪个 uid。并存本身就是错的，现在只剩一份 URDF、一个机器人 uid，所以断言也跟着收敛。
"""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mani_skill")

if not torch.cuda.is_available():
    pytest.skip("so101_sim 需要 CUDA（ManiSkill GPU 后端）", allow_module_level=True)

# 三个分发场景，就是全部入口；不做参数化派生，也不为某种用途另开孪生。
TASKS = {"SO101PickPlaceCube40-v1", "SO101PickPlaceCube20-v1", "SO101PickPlaceCylinder40-v1"}
# 机器人只有这两个 uid：基础版与压进真机速度包线的版本，共用同一份几何。
AGENTS = {"so101", "so101_kit_slow"}


def test_import_registers_tasks_and_agent():
    import so101_sim  # noqa: F401  导入即注册
    from mani_skill.agents.registration import REGISTERED_AGENTS
    from mani_skill.utils.registration import REGISTERED_ENVS

    assert TASKS.issubset(set(REGISTERED_ENVS))
    assert AGENTS.issubset(set(REGISTERED_AGENTS))


def test_env_dual_camera_640x480_and_6dof():
    import gymnasium as gym

    import so101_sim  # noqa: F401

    env = gym.make(
        "SO101PickPlaceCube40-v1",
        num_envs=1,
        obs_mode="rgb",
        render_mode="all",
        sim_backend="gpu",
        domain_randomization=False,
    )
    obs, _ = env.reset(seed=0)
    sensors = obs["sensor_data"]
    assert {"top", "wrist"}.issubset(set(sensors.keys()))
    # 不覆盖分辨率：用环境自己的标定值。相机内参就在 4:3 下标的，
    # 改宽高比而 fovy 不变会把水平视野压掉，画面与真机对不上。
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
    env.close()


def test_step_meshes_present():
    """convert_step.py 产物（4 件物体 visual+collision）已随包就位。"""
    import so101_sim

    objects = Path(so101_sim.__file__).parent / "robots" / "kit_assets" / "objects"
    for name in ("cube_2", "cube_4", "cylinder_4", "bin_2"):
        assert (objects / f"{name}_visual.glb").exists(), name
        assert (objects / f"{name}_collision.obj").exists(), name
