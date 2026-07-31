"""Collision tuning for the SO-101 gripper, task objects, and manipulation contacts."""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

ROBOT_PREFIX = "so101_"

MANIPULATION_CONTACT_SOLREF = (0.55, 1.0)
MANIPULATION_CONTACT_SOLIMP = (0.92, 0.99, 0.005, 0.35, 1.0)
GRIPPER_JAW_CONTACT_SOLREF = (0.35, 1.0)
GRIPPER_JAW_CONTACT_SOLIMP = (0.94, 0.99, 0.004, 0.30, 1.0)
GRIPPER_JAW_FRICTION = (1.5, 0.05, 0.0005)
GRIPPER_DISABLE_COLLISION_MESH_NAMES = ("sts3215", "wrist_roll_follower")
GRIPPER_JAW_COLLISION_GEOM_NAMES = ("gripper_fixed_jaw_collision", "gripper_moving_jaw_collision")
GRIPPER_DISABLED_JAW_MESH_MARKERS = ("Fixed_part_1",)
GRIPPER_ACTIVE_JAW_MESH_MARKERS = ("Fixed_part_2",)
ROBOT_COLLISION_ONLY_MESH_MARKERS = (
    "_convex",
    "Fixed_part_1",
    "Fixed_part_2",
)
TASK_OBJECT_GEOMS = frozenset({"cube_geom", "bottle_body_geom", "bottle_neck_geom"})
GRIPPER_CUBE_CONTACT_SOLREF = (0.06, 1.0)
GRIPPER_CUBE_CONTACT_SOLIMP = (0.95, 0.99, 0.002, 0.40, 1.0)
GRIPPER_OBJECT_CONTACT_MARGIN = 0.0015
GRIPPER_OBJECT_CONTACT_FRICTION = (2.0, 2.0, 0.05, 0.0005, 0.0005)
GRIPPER_BOTTLE_CONTACT_SOLREF = (0.12, 1.0)
GRIPPER_BOTTLE_CONTACT_SOLIMP = (0.92, 0.99, 0.005, 0.35, 1.0)
GRIPPER_JAW_MESH_DEBUG_MARKERS = (
    ("Fixed_part_2", (1.0, 0.55, 0.15, 0.50)),
)
COLLISION_DEBUG_RGBA = {
    "bottle_body_geom": (0.15, 0.85, 1.0, 0.45),
    "bottle_neck_geom": (0.10, 0.65, 0.95, 0.45),
    "cube_geom": (0.15, 0.85, 1.0, 0.40),
    f"{ROBOT_PREFIX}{GRIPPER_JAW_COLLISION_GEOM_NAMES[0]}": (1.0, 0.55, 0.15, 0.50),
    f"{ROBOT_PREFIX}{GRIPPER_JAW_COLLISION_GEOM_NAMES[1]}": (1.0, 0.20, 0.15, 0.50),
}
ROBOT_MATERIALS = {
    "plastic": (0.95, 0.95, 0.95, 1.0),
    "motor": (0.10, 0.10, 0.10, 1.0),
}


def configure_stable_contact(geom: mujoco.MjsGeom) -> None:
    """Tune contact parameters to avoid explosive impulses on light touch."""

    geom.condim = 3
    geom.margin = 0.0008
    geom.solref = list(MANIPULATION_CONTACT_SOLREF)
    geom.solimp = list(MANIPULATION_CONTACT_SOLIMP)


def is_gripper_jaw_collision_geom(geom_name: str) -> bool:
    return any(
        geom_name == jaw_name or geom_name.endswith(f"_{jaw_name}")
        for jaw_name in GRIPPER_JAW_COLLISION_GEOM_NAMES
    )


def is_gripper_jaw_geom(body_name: str, mesh_name: str, geom_name: str = "") -> bool:
    del body_name, mesh_name
    return is_gripper_jaw_collision_geom(geom_name)


def is_robot_collision_only_mesh(mesh_name: str) -> bool:
    return any(marker in mesh_name for marker in ROBOT_COLLISION_ONLY_MESH_MARKERS)


def configure_manipulation_contacts(model: mujoco.MjModel) -> None:
    """Soften arm-body contacts with task objects; gripper jaws are configured separately."""

    solref = np.asarray(MANIPULATION_CONTACT_SOLREF, dtype=np.float64)
    solimp = np.asarray(MANIPULATION_CONTACT_SOLIMP, dtype=np.float64)
    for geom_id in range(model.ngeom):
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[geom_id]) or ""
        if model.geom_contype[geom_id] == 0:
            continue
        mesh_id = model.geom_dataid[geom_id]
        mesh_name = ""
        if mesh_id >= 0:
            mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id) or ""
        if is_gripper_jaw_geom(body_name, mesh_name, geom_name):
            continue
        if geom_name in TASK_OBJECT_GEOMS:
            continue
        if body_name.startswith(ROBOT_PREFIX):
            model.geom_solref[geom_id] = solref
            model.geom_solimp[geom_id] = solimp
            model.geom_margin[geom_id] = max(model.geom_margin[geom_id], 0.0008)


def configure_robot_visual_geoms(model: mujoco.MjModel) -> None:
    """Keep STL appearance geoms visual-only; convex proxies handle contact."""

    for geom_id in range(model.ngeom):
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[geom_id]) or ""
        if not body_name.startswith(ROBOT_PREFIX):
            continue

        mesh_id = model.geom_dataid[geom_id]
        if mesh_id < 0:
            continue

        mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id) or ""
        if is_robot_collision_only_mesh(mesh_name):
            continue
        if is_gripper_jaw_collision_geom(geom_name):
            continue

        model.geom_contype[geom_id] = 0
        model.geom_conaffinity[geom_id] = 0


def configure_gripper_collision(model: mujoco.MjModel) -> None:
    """Enable named jaw collision geoms; all other gripper meshes stay visual-only."""

    jaw_solref = np.asarray(GRIPPER_JAW_CONTACT_SOLREF, dtype=np.float64)
    jaw_solimp = np.asarray(GRIPPER_JAW_CONTACT_SOLIMP, dtype=np.float64)
    jaw_friction = np.asarray(GRIPPER_JAW_FRICTION, dtype=np.float64)

    for geom_id in range(model.ngeom):
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[geom_id]) or ""
        if "gripper" not in body_name and "moving_jaw" not in body_name:
            continue

        mesh_id = model.geom_dataid[geom_id]
        mesh_name = ""
        if mesh_id >= 0:
            mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id) or ""

        if any(marker in mesh_name for marker in GRIPPER_DISABLE_COLLISION_MESH_NAMES):
            model.geom_contype[geom_id] = 0
            model.geom_conaffinity[geom_id] = 0
            continue

        if not is_gripper_jaw_collision_geom(geom_name):
            model.geom_contype[geom_id] = 0
            model.geom_conaffinity[geom_id] = 0
            continue

        model.geom_solref[geom_id] = jaw_solref
        model.geom_solimp[geom_id] = jaw_solimp
        model.geom_margin[geom_id] = 0.001
        model.geom_friction[geom_id] = jaw_friction
        model.geom_condim[geom_id] = 3


def configure_collision_debug_view(model: mujoco.MjModel, *, enabled: bool = True) -> None:
    """Tint bottle and gripper collision geoms so they are visible in the viewer."""

    if not enabled:
        return

    for geom_id in range(model.ngeom):
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if geom_name in COLLISION_DEBUG_RGBA:
            model.geom_rgba[geom_id] = COLLISION_DEBUG_RGBA[geom_name]
            continue

        mesh_id = model.geom_dataid[geom_id]
        if mesh_id < 0:
            continue

        mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id) or ""
        for marker, rgba in GRIPPER_JAW_MESH_DEBUG_MARKERS:
            if marker in mesh_name:
                model.geom_rgba[geom_id] = rgba
                break


def configure_collision_debug_viewer(viewer: Any, *, enabled: bool = True) -> None:
    """Enable contact-point overlays in the passive MuJoCo viewer."""

    if not enabled:
        return

    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True


def name_gripper_collision_geoms(robot_spec: mujoco.MjSpec) -> None:
    """Give jaw collision geoms stable names for contact-pair tuning."""

    for geom in robot_spec.geoms:
        mesh_name = geom.meshname or ""
        geom_name = geom.name or ""
        body_name = geom.parent.name if geom.parent is not None else ""
        if "Fixed_part_2" in mesh_name and "simplified" not in mesh_name:
            geom.name = GRIPPER_JAW_COLLISION_GEOM_NAMES[0]
            continue
        if "moving_jaw" in body_name and "_proxy_" in geom_name and "moving_jaw_so101_v1" in mesh_name:
            geom.name = GRIPPER_JAW_COLLISION_GEOM_NAMES[1]


def _add_gripper_object_contact_pairs(
    scene_spec: mujoco.MjSpec,
    *,
    object_geom: str,
    solref: tuple[float, float],
    solimp: tuple[float, float, float, float, float],
    margin: float,
) -> None:
    """Configure jaw contacts independently from a task object's table contact."""

    for jaw_name in GRIPPER_JAW_COLLISION_GEOM_NAMES:
        pair = scene_spec.add_pair()
        pair.geomname1 = f"{ROBOT_PREFIX}{jaw_name}"
        pair.geomname2 = object_geom
        pair.solref = list(solref)
        pair.solimp = list(solimp)
        pair.margin = margin
        pair.friction = list(GRIPPER_OBJECT_CONTACT_FRICTION)


def add_gripper_cube_contact_pairs(scene_spec: mujoco.MjSpec) -> None:
    """Start cube jaw contact before penetration with a compliant response."""

    _add_gripper_object_contact_pairs(
        scene_spec,
        object_geom="cube_geom",
        solref=GRIPPER_CUBE_CONTACT_SOLREF,
        solimp=GRIPPER_CUBE_CONTACT_SOLIMP,
        margin=GRIPPER_OBJECT_CONTACT_MARGIN,
    )


def add_gripper_bottle_contact_pairs(scene_spec: mujoco.MjSpec) -> None:
    """Soften gripper-finger / bottle contacts independently from the stiff table support."""

    _add_gripper_object_contact_pairs(
        scene_spec,
        object_geom="bottle_body_geom",
        solref=GRIPPER_BOTTLE_CONTACT_SOLREF,
        solimp=GRIPPER_BOTTLE_CONTACT_SOLIMP,
        margin=0.0,
    )


def hide_robot_collision_meshes(model: mujoco.MjModel) -> None:
    """Hide convex / decomposed collision proxies in the normal viewer."""

    for geom_id in range(model.ngeom):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[geom_id]) or ""
        if not body_name.startswith(ROBOT_PREFIX):
            continue

        mesh_id = model.geom_dataid[geom_id]
        if mesh_id < 0:
            continue

        mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id) or ""
        if is_robot_collision_only_mesh(mesh_name):
            model.geom_rgba[geom_id] = (0.0, 0.0, 0.0, 0.0)


def apply_robot_colors(model: mujoco.MjModel) -> None:
    """Restore URDF material colors on visible robot meshes."""

    for geom_id in range(model.ngeom):
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if "_frame_" in geom_name and geom_name.endswith("_axis"):
            continue

        body_id = model.geom_bodyid[geom_id]
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if not body_name.startswith(ROBOT_PREFIX):
            continue

        mesh_id = model.geom_dataid[geom_id]
        mesh_name = ""
        if mesh_id >= 0:
            mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id) or ""
            if is_robot_collision_only_mesh(mesh_name):
                continue

        if "sts3215" in mesh_name:
            rgba = ROBOT_MATERIALS["motor"]
        else:
            rgba = ROBOT_MATERIALS["plastic"]
        model.geom_rgba[geom_id] = rgba
