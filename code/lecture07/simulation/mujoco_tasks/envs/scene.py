from __future__ import annotations

from math import pi
from pathlib import Path

import mujoco

from .utils import actuators, collision, mesh_assets

LECTURE07_ROOT = Path(__file__).resolve().parents[3]
SIMULATION_ROOT = LECTURE07_ROOT / "simulation"
CODE_ROOT = LECTURE07_ROOT.parent
MODEL_DIR = SIMULATION_ROOT / "assets" / "models"

ARM_JOINT_NAMES = (
    f"{collision.ROBOT_PREFIX}shoulder_pan",
    f"{collision.ROBOT_PREFIX}shoulder_lift",
    f"{collision.ROBOT_PREFIX}elbow_flex",
    f"{collision.ROBOT_PREFIX}wrist_flex",
    f"{collision.ROBOT_PREFIX}wrist_roll",
)
GRIPPER_JOINT_NAME = f"{collision.ROBOT_PREFIX}gripper"
JOINT_NAMES = ARM_JOINT_NAMES + (GRIPPER_JOINT_NAME,)
EEF_BODY_NAME = f"{collision.ROBOT_PREFIX}gripper_frame_link"
# ``wrist_roll`` rotates this link about local +Z at its body origin.
WRIST_ROLL_BODY_NAME = f"{collision.ROBOT_PREFIX}gripper_link"
BASE_LINK_BODY_NAME = f"{collision.ROBOT_PREFIX}base_link"
HOME_QPOS = (0.0, 0.0, 0.0, pi / 2, -pi / 2, 60 * pi / 180)

TABLE_HALF_Z = 0.002
TABLE_X = 0.22
TABLE_Y = -0.02
TABLE_HALF_X = 0.25
TABLE_HALF_Y = 0.36
FLOOR_Z = -0.05

TABLE_TOP_Z = mesh_assets.compute_base_mesh_bottom_z(
    HOME_QPOS,
    is_collision_only_mesh=collision.is_robot_collision_only_mesh,
    name_gripper_collision_geoms=collision.name_gripper_collision_geoms,
)
TABLE_BODY_Z = TABLE_TOP_Z - TABLE_HALF_Z


def _grasp_viz_scene_xml() -> str:
    from mujoco_tasks.viz.scene_viz import VIEW_AZIMUTH, VIEW_ELEVATION

    return f"""
<mujoco model="lecture07_grasp_viz">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"/>
  <visual>
    <headlight diffuse="0.55 0.55 0.55" ambient="0.25 0.25 0.25"/>
    <global azimuth="{VIEW_AZIMUTH:.0f}" elevation="{VIEW_ELEVATION:.0f}"
            offwidth="1280" offheight="720"/>
  </visual>
  <asset>
    <material name="white_table" rgba="0.96 0.96 0.96 1"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" pos="0 0 {FLOOR_Z}" size="2 2 0.05" rgba="0.82 0.82 0.82 1"/>
    <body name="tabletop" pos="{TABLE_X} {TABLE_Y} {TABLE_BODY_Z}">
      <geom name="tabletop_visual" type="box" size="{TABLE_HALF_X} {TABLE_HALF_Y} {TABLE_HALF_Z}" material="white_table" contype="0" conaffinity="0"/>
    </body>
    <geom name="tabletop_geom" type="plane" pos="{TABLE_X} {TABLE_Y} {TABLE_TOP_Z}" size="{TABLE_HALF_X} {TABLE_HALF_Y} 0.01" rgba="0.96 0.96 0.96 1"/>
    <body name="robot_mount" pos="0 0 0"/>
  </worldbody>
</mujoco>
"""


def _attach_robot(scene_spec: mujoco.MjSpec) -> None:
    robot_spec = mesh_assets.load_robot_spec(
        name_gripper_collision_geoms=collision.name_gripper_collision_geoms,
    )
    mount = scene_spec.body("robot_mount")
    frame = mount.add_frame()
    scene_spec.attach(robot_spec, frame=frame, prefix=collision.ROBOT_PREFIX)

    for joint_name in ARM_JOINT_NAMES:
        actuators.add_position_actuator(scene_spec, joint_name, kp=actuators.ARM_ACTUATOR_KP)
    actuators.add_position_actuator(scene_spec, GRIPPER_JOINT_NAME, kp=actuators.GRIPPER_ACTUATOR_KP)


def compile_grasp_viz_scene_spec() -> mujoco.MjSpec:
    """Minimal scene: floor, table, and robot only."""

    from mujoco_tasks.viz.scene_viz import EEF_SITE_RGBA_GRASP_VIZ, add_eef_site

    scene_spec = mujoco.MjSpec.from_string(_grasp_viz_scene_xml())
    _attach_robot(scene_spec)
    add_eef_site(scene_spec.body(EEF_BODY_NAME), rgba=EEF_SITE_RGBA_GRASP_VIZ)
    return scene_spec


def _finalize_robot_model(model: mujoco.MjModel) -> mujoco.MjModel:
    actuators.configure_robot_actuators(model)
    collision.configure_gripper_collision(model)
    collision.configure_robot_visual_geoms(model)
    collision.configure_manipulation_contacts(model)
    collision.hide_robot_collision_meshes(model)
    collision.apply_robot_colors(model)
    return model


def build_grasp_viz_model(
    *,
    show_grasp: bool = True,
    show_grasp_axes: bool = True,
) -> mujoco.MjModel:
    """Compose a minimal grasp-visualization scene (floor, table, robot, grasps)."""

    scene_spec = compile_grasp_viz_scene_spec()
    if show_grasp:
        _attach_home_grasp_visualization(scene_spec, show_grasp_axes=show_grasp_axes)
    return _finalize_robot_model(scene_spec.compile())


def _attach_home_grasp_visualization(
    scene_spec: mujoco.MjSpec,
    *,
    show_grasp_axes: bool,
) -> None:
    from mujoco_tasks.viz.grasp_viz import add_grasp_visualization_on_wrist_roll, home_grasp_visual_params

    probe_model = _finalize_robot_model(scene_spec.compile())
    probe_data = mujoco.MjData(probe_model)
    center_in_wrist, width, depth, height = home_grasp_visual_params(probe_model, probe_data)
    add_grasp_visualization_on_wrist_roll(
        scene_spec,
        WRIST_ROLL_BODY_NAME,
        center_in_wrist=center_in_wrist,
        width=width,
        depth=depth,
        height=height,
        show_axes=show_grasp_axes,
    )


def _scene_xml(task: str) -> str:
    from . import bottle, cube
    from mujoco_tasks.viz.scene_viz import VIEW_AZIMUTH, VIEW_ELEVATION

    if task == "cube":
        extra_geoms = cube.ZONE_GEOMS_XML
        materials = cube.SCENE_MATERIALS_XML
    else:
        extra_geoms = ""
        materials = bottle.SCENE_MATERIALS_XML

    return f"""
<mujoco model="lecture07_{task}">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"/>
  <visual>
    <headlight diffuse="0.55 0.55 0.55" ambient="0.25 0.25 0.25"/>
    <global azimuth="{VIEW_AZIMUTH:.0f}" elevation="{VIEW_ELEVATION:.0f}"
            offwidth="1280" offheight="720"/>
  </visual>
  <asset>{materials}
  </asset>
  <worldbody>
    <geom name="floor" type="plane" pos="0 0 {FLOOR_Z}" size="2 2 0.05" rgba="0.82 0.82 0.82 1"/>
    <body name="tabletop" pos="{TABLE_X} {TABLE_Y} {TABLE_BODY_Z}">
      <geom name="tabletop_visual" type="box" size="{TABLE_HALF_X} {TABLE_HALF_Y} {TABLE_HALF_Z}" material="white_table" contype="0" conaffinity="0"/>
    </body>
    <geom name="tabletop_geom" type="plane" pos="{TABLE_X} {TABLE_Y} {TABLE_TOP_Z}" size="{TABLE_HALF_X} {TABLE_HALF_Y} 0.01" rgba="0.96 0.96 0.96 1"/>{extra_geoms}
    <body name="robot_mount" pos="0 0 0"/>
  </worldbody>
</mujoco>
"""


def compile_scene_spec(task: str = "cube") -> mujoco.MjSpec:
    """Build the Lecture 07 MuJoCo scene spec before compilation."""

    from . import bottle, cube
    from mujoco_tasks.viz.scene_viz import add_robot_frame_visuals, add_scene_camera_markers

    if task not in {"cube", "bottle"}:
        raise ValueError(f"unknown task {task!r}")

    # 1. 加载场景 XML：地面、桌面、机器人挂载点
    scene_spec = mujoco.MjSpec.from_string(_scene_xml(task))

    # 2. 添加任务物体（cube / bottle）
    if task == "cube":
        cube.add_task_objects(scene_spec)
    else:
        bottle.add_task_objects(scene_spec)

    # 3. 挂载 SO-101 机器人 URDF
    robot_spec = mesh_assets.load_robot_spec(
        name_gripper_collision_geoms=collision.name_gripper_collision_geoms,
    )
    mount = scene_spec.body("robot_mount")
    frame = mount.add_frame()
    scene_spec.attach(robot_spec, frame=frame, prefix=collision.ROBOT_PREFIX)

    # 4. 添加 base / TCP 坐标轴与 EEF 站点（仅可视化）
    add_robot_frame_visuals(
        scene_spec,
        base_body_name=BASE_LINK_BODY_NAME,
        eef_body_name=EEF_BODY_NAME,
    )

    # 5. 为机械臂与夹爪关节添加位置伺服执行器
    for joint_name in ARM_JOINT_NAMES:
        actuators.add_position_actuator(scene_spec, joint_name, kp=actuators.ARM_ACTUATOR_KP)
    actuators.add_position_actuator(scene_spec, GRIPPER_JOINT_NAME, kp=actuators.GRIPPER_ACTUATOR_KP)

    # 6. 添加默认相机调试标记（SHOW_VIEW_CAMERA_MARKER 开启时可见）
    add_scene_camera_markers(scene_spec)

    # 7. 配置夹爪与任务物体的接触对
    if task == "cube":
        collision.add_gripper_cube_contact_pairs(scene_spec)
    else:
        collision.add_gripper_bottle_contact_pairs(scene_spec)

    return scene_spec


def _finalize_model(model: mujoco.MjModel, task: str) -> mujoco.MjModel:
    from . import bottle

    model = _finalize_robot_model(model)
    if task == "bottle":
        bottle.configure_bottle_physics(model)
    return model


def build_model(
    task: str = "cube",
    *,
    show_grasp: bool = False,
    show_grasp_axes: bool = True,
    show_poses: bool = False,
) -> mujoco.MjModel:
    """Compose the Lecture 07 scene for the cube or bottle MuJoCo task."""

    scene_spec = compile_scene_spec(task)
    if show_poses:
        from mujoco_tasks.viz.pose_viz import add_pose_markers

        add_pose_markers(scene_spec, task)
    if show_grasp:
        _attach_home_grasp_visualization(scene_spec, show_grasp_axes=show_grasp_axes)
    return _finalize_model(scene_spec.compile(), task)
