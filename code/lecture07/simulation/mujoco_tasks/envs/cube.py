"""Lecture 07 red-cube A-to-B MuJoCo scene."""

from __future__ import annotations

import mujoco

from .scene import TABLE_TOP_Z, TABLE_X

# 8 cm square frames: red at B(0.30, -0.12), blue at A(0.30, 0.12).
ZONE_RED_Y = -0.12
ZONE_BLUE_Y = 0.12
ZONE_HALF = 0.04
ZONE_LINE = 0.0015
ZONE_THICK = 0.001

CUBE_SIZE = 0.05
CUBE_HALF = CUBE_SIZE / 2
CUBE_X = 0.22
CUBE_Y = (ZONE_RED_Y + ZONE_BLUE_Y) / 2
ZONE_Z = TABLE_TOP_Z + ZONE_THICK

ZONE_GEOMS_XML = f"""
    <geom name="zone_a_front" type="box" pos="{TABLE_X} {ZONE_RED_Y - ZONE_HALF:.3f} {ZONE_Z}" size="{ZONE_HALF} {ZONE_LINE} {ZONE_THICK}" material="zone_red" contype="0" conaffinity="0"/>
    <geom name="zone_a_back"  type="box" pos="{TABLE_X} {ZONE_RED_Y + ZONE_HALF:.3f} {ZONE_Z}" size="{ZONE_HALF} {ZONE_LINE} {ZONE_THICK}" material="zone_red" contype="0" conaffinity="0"/>
    <geom name="zone_a_left"  type="box" pos="{TABLE_X - ZONE_HALF:.3f} {ZONE_RED_Y} {ZONE_Z}" size="{ZONE_LINE} {ZONE_HALF} {ZONE_THICK}" material="zone_red" contype="0" conaffinity="0"/>
    <geom name="zone_a_right" type="box" pos="{TABLE_X + ZONE_HALF:.3f} {ZONE_RED_Y} {ZONE_Z}" size="{ZONE_LINE} {ZONE_HALF} {ZONE_THICK}" material="zone_red" contype="0" conaffinity="0"/>
    <geom name="zone_b_front" type="box" pos="{TABLE_X} {ZONE_BLUE_Y - ZONE_HALF:.3f} {ZONE_Z}" size="{ZONE_HALF} {ZONE_LINE} {ZONE_THICK}" material="zone_blue" contype="0" conaffinity="0"/>
    <geom name="zone_b_back"  type="box" pos="{TABLE_X} {ZONE_BLUE_Y + ZONE_HALF:.3f} {ZONE_Z}" size="{ZONE_HALF} {ZONE_LINE} {ZONE_THICK}" material="zone_blue" contype="0" conaffinity="0"/>
    <geom name="zone_b_left"  type="box" pos="{TABLE_X - ZONE_HALF:.3f} {ZONE_BLUE_Y} {ZONE_Z}" size="{ZONE_LINE} {ZONE_HALF} {ZONE_THICK}" material="zone_blue" contype="0" conaffinity="0"/>
    <geom name="zone_b_right" type="box" pos="{TABLE_X + ZONE_HALF:.3f} {ZONE_BLUE_Y} {ZONE_Z}" size="{ZONE_LINE} {ZONE_HALF} {ZONE_THICK}" material="zone_blue" contype="0" conaffinity="0"/>"""

SCENE_MATERIALS_XML = """
    <material name="white_table" rgba="0.96 0.96 0.96 1"/>
    <material name="zone_red" rgba="0.65 0.03 0.03 1"/>
    <material name="zone_blue" rgba="0.02 0.12 0.75 1"/>
    <material name="red_cube" rgba="0.82 0.12 0.12 1"/>"""


def add_task_objects(spec: mujoco.MjSpec) -> None:
    cube_body = spec.worldbody.add_body(name="cube", pos=[CUBE_X, CUBE_Y, TABLE_TOP_Z])
    cube_free = cube_body.add_freejoint(name="cube_free")
    cube_free.damping = 1.0
    cube_geom = cube_body.add_geom(
        name="cube_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0.0, 0.0, CUBE_HALF],
        size=[CUBE_HALF, CUBE_HALF, CUBE_HALF],
        material="red_cube",
        mass=0.08,
    )
    cube_geom.friction = [1.0, 0.05, 0.001]
