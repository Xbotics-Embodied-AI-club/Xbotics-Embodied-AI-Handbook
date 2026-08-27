"""Lecture 07 bottle-to-box MuJoCo scene."""

from __future__ import annotations

import mujoco

from .utils.collision import configure_stable_contact
from .scene import MODEL_DIR, TABLE_TOP_Z, TABLE_X
from .utils.mesh_assets import add_textured_material

BOTTLE_OBJ = MODEL_DIR / "bottle.obj"
BOTTLE_TEXTURE = MODEL_DIR / "bottle_texture.png"
BOX_OBJ = MODEL_DIR / "box.obj"
BOX_TEXTURE = MODEL_DIR / "box_texture.png"

BOTTLE_HEIGHT = 0.14
BOTTLE_DIAMETER = 0.038
BOTTLE_BODY_RADIUS = BOTTLE_DIAMETER / 2
# Match the tapered mesh profile: wide body below the shoulder, narrower neck on top.
BOTTLE_BODY_HALF_HEIGHT = 0.055
BOTTLE_BODY_CENTER_Z = BOTTLE_BODY_HALF_HEIGHT
BOTTLE_NECK_RADIUS = 0.012
BOTTLE_NECK_HALF_HEIGHT = 0.015
BOTTLE_NECK_CENTER_Z = BOTTLE_HEIGHT - BOTTLE_NECK_HALF_HEIGHT
BOTTLE_MASS = 0.30
# MuJoCo 3.x 中 free joint 的阻尼是 3 维平动阻尼（旋转阻尼不支持），不再是标量。
BOTTLE_FREE_DAMPING = [5.0, 5.0, 5.0]
# Match the floor plane: stiff support, low bounce.
BOTTLE_CONTACT_SOLREF = (0.02, 1.0)
BOTTLE_CONTACT_SOLIMP = (0.95, 0.99, 0.001, 0.50, 1.0)
TABLE_CONTACT_SOLREF = (0.02, 1.0)
TABLE_CONTACT_SOLIMP = (0.95, 0.99, 0.001, 0.50, 1.0)
BOTTLE_X = TABLE_X
BOTTLE_Y = -0.12

BOX_LENGTH = 0.16
BOX_WIDTH = 0.115
BOX_HEIGHT = 0.028
BOX_X = TABLE_X
BOX_Y = 0.12
# The box.obj mesh is a hollow tray (four thin walls + a bottom plate).  MuJoCo
# meshes collide as a single convex hull, which would fill the interior, so the
# collision is built from thin box geoms instead and the mesh stays visual-only.
#

BOX_WALL_THICKNESS = 0.008
BOX_BOTTOM_THICKNESS = 0.008
BOX_COLLISION_RGBA = (0.65, 0.55, 0.45, 0.0)

SCENE_MATERIALS_XML = """
    <material name="white_table" rgba="0.96 0.96 0.96 1"/>"""


def add_task_objects(spec: mujoco.MjSpec) -> None:
    bottle_material = add_textured_material(spec, "bottle", BOTTLE_TEXTURE)
    box_material = add_textured_material(spec, "box", BOX_TEXTURE)

    bottle_mesh = spec.add_mesh()
    bottle_mesh.name = "lecture07_bottle_mesh"
    bottle_mesh.file = str(BOTTLE_OBJ)

    box_mesh = spec.add_mesh()
    box_mesh.name = "lecture07_box_mesh"
    box_mesh.file = str(BOX_OBJ)

    bottle_body = spec.worldbody.add_body(
        name="bottle",
        pos=[BOTTLE_X, BOTTLE_Y, TABLE_TOP_Z],
    )
    bottle_free = bottle_body.add_freejoint(name="bottle_free")
    bottle_free.damping = BOTTLE_FREE_DAMPING

    bottle_visual = bottle_body.add_geom(
        name="bottle_visual_geom",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="lecture07_bottle_mesh",
        material=bottle_material,
        mass=0.0,
    )
    bottle_visual.contype = 0
    bottle_visual.conaffinity = 0

    bottle_body_geom = bottle_body.add_geom(
        name="bottle_body_geom",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=[0.0, 0.0, BOTTLE_BODY_CENTER_Z],
        size=[BOTTLE_BODY_RADIUS, BOTTLE_BODY_HALF_HEIGHT, 0.0],
        rgba=[1.0, 1.0, 1.0, 0.0],
        mass=BOTTLE_MASS,
    )
    bottle_body_geom.friction = [0.8, 0.04, 0.0008]
    configure_stable_contact(bottle_body_geom)
    bottle_body_geom.solref = list(BOTTLE_CONTACT_SOLREF)
    bottle_body_geom.solimp = list(BOTTLE_CONTACT_SOLIMP)

    bottle_neck_geom = bottle_body.add_geom(
        name="bottle_neck_geom",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=[0.0, 0.0, BOTTLE_NECK_CENTER_Z],
        size=[BOTTLE_NECK_RADIUS, BOTTLE_NECK_HALF_HEIGHT, 0.0],
        rgba=[1.0, 1.0, 1.0, 0.0],
        mass=0.0,
    )
    bottle_neck_geom.friction = [0.8, 0.04, 0.0008]
    bottle_neck_geom.contype = 0
    bottle_neck_geom.conaffinity = 0

    box_body = spec.worldbody.add_body(
        name="box",
        pos=[BOX_X, BOX_Y, TABLE_TOP_Z],
    )
    box_geom = box_body.add_geom(
        name="box_geom",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="lecture07_box_mesh",
        material=box_material,
        mass=0.0,
    )
    box_geom.friction = [1.0, 0.05, 0.001]
    box_geom.contype = 0
    box_geom.conaffinity = 0

    _add_box_collision_geoms(spec)


def _add_box_collision_geoms(scene_spec: mujoco.MjSpec) -> None:
    """Add hollow-tray collision geoms (bottom + four thin walls) for the box."""

    x0 = BOX_X - BOX_LENGTH / 2
    x1 = BOX_X + BOX_LENGTH / 2
    y0 = BOX_Y - BOX_WIDTH / 2
    y1 = BOX_Y + BOX_WIDTH / 2
    z0 = TABLE_TOP_Z
    z1 = TABLE_TOP_Z + BOX_HEIGHT
    wall_half = BOX_WALL_THICKNESS / 2

    def add(name: str, pos: list[float], half: list[float]) -> None:
        geom = scene_spec.worldbody.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=pos,
            size=half,
            rgba=list(BOX_COLLISION_RGBA),
        )
        geom.mass = 0.0
        configure_stable_contact(geom)
        geom.friction = [1.2, 0.06, 0.001]

    add(
        "box_bottom",
        [BOX_X, BOX_Y, z0 + BOX_BOTTOM_THICKNESS / 2],
        [
            BOX_LENGTH / 2 - wall_half,
            BOX_WIDTH / 2 - wall_half,
            BOX_BOTTOM_THICKNESS / 2,
        ],
    )
    add(
        "box_wall_front",
        [BOX_X, y0 + wall_half, z0 + BOX_HEIGHT / 2],
        [BOX_LENGTH / 2 - wall_half, wall_half, BOX_HEIGHT / 2],
    )
    add(
        "box_wall_back",
        [BOX_X, y1 - wall_half, z0 + BOX_HEIGHT / 2],
        [BOX_LENGTH / 2 - wall_half, wall_half, BOX_HEIGHT / 2],
    )
    add(
        "box_wall_left",
        [x0 + wall_half, BOX_Y, z0 + BOX_HEIGHT / 2],
        [wall_half, BOX_WIDTH / 2 - wall_half, BOX_HEIGHT / 2],
    )
    add(
        "box_wall_right",
        [x1 - wall_half, BOX_Y, z0 + BOX_HEIGHT / 2],
        [wall_half, BOX_WIDTH / 2 - wall_half, BOX_HEIGHT / 2],
    )


def place_bottle_on_table(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Spawn the bottle resting on the table plane without a drop settle."""

    joint = model.joint("bottle_free")
    qadr = joint.qposadr[0]
    dadr = joint.dofadr[0]
    data.qpos[qadr : qadr + 3] = [BOTTLE_X, BOTTLE_Y, TABLE_TOP_Z]
    data.qpos[qadr + 3 : qadr + 7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[dadr : dadr + 6] = 0.0
    mujoco.mj_forward(model, data)


def configure_bottle_physics(model: mujoco.MjModel) -> None:
    """Re-apply bottle contact parameters after scene-wide contact tuning."""

    import numpy as np

    # MuJoCo 3.x 会把 free joint 的 3 维平动阻尼广播到全部 6 个自由度,将旋转清0
    bottle_dof = int(model.joint("bottle_free").dofadr[0])
    model.dof_damping[bottle_dof + 3 : bottle_dof + 6] = 0.0

    bottle_solref = np.asarray(BOTTLE_CONTACT_SOLREF, dtype=np.float64)
    bottle_solimp = np.asarray(BOTTLE_CONTACT_SOLIMP, dtype=np.float64)
    body_id = model.geom("bottle_body_geom").id
    model.geom_solref[body_id] = bottle_solref
    model.geom_solimp[body_id] = bottle_solimp
    model.geom_margin[body_id] = 0.0005
    model.geom_friction[body_id] = [0.9, 0.04, 0.0005]

    table_id = model.geom("tabletop_geom").id
    model.geom_solref[table_id] = np.asarray(TABLE_CONTACT_SOLREF, dtype=np.float64)
    model.geom_solimp[table_id] = np.asarray(TABLE_CONTACT_SOLIMP, dtype=np.float64)
    model.geom_friction[table_id] = [0.9, 0.04, 0.0005]
