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
BOTTLE_FREE_DAMPING = 5.0
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
