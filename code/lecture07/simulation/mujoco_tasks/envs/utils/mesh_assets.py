"""OBJ texture registration and URDF mesh patching for Lecture 07 MuJoCo scenes."""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import mujoco

if TYPE_CHECKING:
    from collections.abc import Sequence

_LECTURE07_ROOT = Path(__file__).resolve().parents[4]
_CODE_ROOT = _LECTURE07_ROOT.parent
URDF_PATH = _CODE_ROOT / "platform" / "so101_sim" / "so101_sim" / "robots" / "so101_base" / "so101.urdf"

MESH_DIR = URDF_PATH.parent


def add_textured_material(spec: mujoco.MjSpec, prefix: str, texture_path: Path | None) -> str | None:
    """Register a MuJoCo material for a mesh texture; return the material name."""

    if texture_path is None:
        return None

    texture = spec.add_texture()
    texture.name = f"{prefix}_texture"
    texture.type = mujoco.mjtTexture.mjTEXTURE_2D
    texture.file = str(texture_path)

    material = spec.add_material()
    material.name = f"{prefix}_material"
    material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = texture.name
    return material.name


def _mesh_label_from_block(block: str) -> str:
    mesh_match = re.search(r'filename="([^"]+)"', block)
    if not mesh_match:
        return "geom"
    return Path(mesh_match.group(1)).stem.replace(".", "_").replace("-", "_")


def _link_slug(link_name: str) -> str:
    return link_name.removesuffix("_link").replace("-", "_")


def _prepare_link_collisions(link_body: str, link_name: str) -> str:
    """Promote visuals to named collision geoms and assign unique names to all proxies."""

    slug = _link_slug(link_name)
    used_names: set[str] = set()

    def unique_name(base: str) -> str:
        name = base
        suffix = 2
        while name in used_names:
            name = f"{base}_{suffix}"
            suffix += 1
        used_names.add(name)
        return name

    def visual_to_collision(match: re.Match[str]) -> str:
        inner = re.sub(r"\s*<material[^>]*/>\s*", "", match.group(1), flags=re.S)
        label = _mesh_label_from_block(inner)
        geom_name = unique_name(f"{slug}_visual_{label}")
        return f'<collision name="{geom_name}">{inner}</collision>'

    body = re.sub(r"<visual>(.*?)</visual>", visual_to_collision, link_body, flags=re.S)
    body = re.sub(r"\s*<visual>.*?</visual>", "", body, flags=re.S)

    def name_collision(match: re.Match[str]) -> str:
        attrs = match.group("attrs")
        inner = match.group("inner")
        if "name=" in attrs:
            return match.group(0)
        label = _mesh_label_from_block(inner)
        geom_name = unique_name(f"{slug}_proxy_{label}")
        return f'<collision name="{geom_name}"{attrs}>{inner}</collision>'

    return re.sub(
        r"<collision(?P<attrs>[^>]*)>(?P<inner>.*?)</collision>",
        name_collision,
        body,
        flags=re.S,
    )


def patch_gripper_collision_meshes(urdf_text: str) -> str:
    """Drop the inactive fixed-jaw collision mesh; use STL collision for the moving jaw."""

    patched = urdf_text.replace(
        "meshes/moving_jaw_so101_v1_convex.obj",
        "meshes/moving_jaw_so101_v1.stl",
    )
    patched = re.sub(
        r"\s*<collision>\s*<origin[^>]*/>\s*<geometry>\s*"
        r"<mesh filename=\"[^\"]*Fixed_part_1\.obj\"/>\s*</geometry>\s*</collision>",
        "",
        patched,
        flags=re.S,
    )
    return patched


def prepare_mujoco_urdf(urdf_path: Path = URDF_PATH) -> str:
    """Prepare SO-101 URDF with separate STL visuals and convex collision proxies.

    MuJoCo ignores URDF <visual> tags, so each visual block is duplicated as an
    extra <collision> entry. Post-compile we hide convex / Fixed_part geoms and
    disable contact on the STL appearance meshes.
    """

    urdf_text = urdf_path.read_text(encoding="utf-8")

    def patch_link(match: re.Match[str]) -> str:
        link_name = match.group(1)
        link_body = match.group(2)
        return f'<link name="{link_name}">{_prepare_link_collisions(link_body, link_name)}</link>'

    patched = re.sub(
        r'<link name="([^"]+)">(.*?)</link>',
        patch_link,
        urdf_text,
        flags=re.S,
    )
    return patch_gripper_collision_meshes(patched)


def load_robot_spec(
    *,
    name_gripper_collision_geoms: Callable[[mujoco.MjSpec], None] | None = None,
) -> mujoco.MjSpec:
    with tempfile.NamedTemporaryFile("w", suffix=".urdf", delete=False, encoding="utf-8") as handle:
        handle.write(prepare_mujoco_urdf())
        temp_path = handle.name
    robot_spec = mujoco.MjSpec.from_file(temp_path)
    robot_spec.meshdir = str(MESH_DIR)
    if name_gripper_collision_geoms is not None:
        name_gripper_collision_geoms(robot_spec)
    return robot_spec


def geom_mesh_bottom_z(model: mujoco.MjModel, data: mujoco.MjData, geom_id: int) -> float:
    """Return the lowest world-frame z of a mesh geom (not its bounding sphere)."""

    mesh_id = model.geom_dataid[geom_id]
    geom_pos = data.geom_xpos[geom_id]
    if mesh_id < 0:
        return float(geom_pos[2] - model.geom_rbound[geom_id])

    vertadr = model.mesh_vertadr[mesh_id]
    vertnum = model.mesh_vertnum[mesh_id]
    verts = model.mesh_vert[vertadr : vertadr + vertnum]
    geom_mat = data.geom_xmat[geom_id].reshape(3, 3)
    world = verts @ geom_mat.T + geom_pos
    return float(world[:, 2].min())


def compute_base_mesh_bottom_z(
    home_qpos: Sequence[float],
    *,
    is_collision_only_mesh: Callable[[str], bool],
    name_gripper_collision_geoms: Callable[[mujoco.MjSpec], None],
) -> float:
    """Lowest z of the fixed-base STL meshes in the home pose (used to seat the table)."""

    robot_spec = load_robot_spec(name_gripper_collision_geoms=name_gripper_collision_geoms)
    model = robot_spec.compile()
    data = mujoco.MjData(model)
    data.qpos[:] = home_qpos
    mujoco.mj_forward(model, data)
    return min(
        geom_mesh_bottom_z(model, data, geom_id)
        for geom_id in range(model.ngeom)
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[geom_id]) == "world"
        and model.geom_dataid[geom_id] >= 0
        and not is_collision_only_mesh(
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, model.geom_dataid[geom_id]) or ""
        )
    )
