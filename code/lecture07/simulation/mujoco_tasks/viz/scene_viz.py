"""Scene-level MuJoCo visualization helpers (axes, camera, TCP markers)."""

from __future__ import annotations

import mujoco
import numpy as np

# Default free-camera pose: face the robot from the +X axis (azimuth=180 in MuJoCo).
VIEW_LOOKAT = (0.10, 0.0, 0.10)
VIEW_DISTANCE = 1.10
VIEW_AZIMUTH = 180.0
VIEW_ELEVATION = -20.0
SHOW_VIEW_CAMERA_MARKER = False

EEF_SITE_RGBA = (0.12, 0.75, 0.18, 1.0)
EEF_SITE_RGBA_GRASP_VIZ = (0.12, 0.75, 0.18, 0.85)


def add_body_axis_frame(
    body: mujoco.MjsBody,
    prefix: str,
    *,
    length: float = 0.08,
    radius: float = 0.002,
) -> None:
    """Draw an RGB XYZ frame on a body (X=red, Y=green, Z=blue)."""

    axes = (
        ("x", [length, 0.0, 0.0], [1.0, 0.0, 0.0, 1.0]),
        ("y", [0.0, length, 0.0], [0.0, 1.0, 0.0, 1.0]),
        ("z", [0.0, 0.0, length], [0.0, 0.0, 1.0, 1.0]),
    )
    for axis_name, endpoint, rgba in axes:
        geom = body.add_geom(
            name=f"{prefix}_{axis_name}_axis",
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            fromto=[0.0, 0.0, 0.0, *endpoint],
            size=[radius, 0.0, 0.0],
            rgba=rgba,
        )
        geom.contype = 0
        geom.conaffinity = 0


def add_eef_site(
    eef_body: mujoco.MjsBody,
    *,
    name: str = "eef_site",
    size: float = 0.005,
    rgba: tuple[float, float, float, float] = EEF_SITE_RGBA,
) -> mujoco.MjsSite:
    """Add a visual site at the tool-center point."""

    site = eef_body.add_site(name=name, pos=[0.0, 0.0, 0.0], size=[size, size, size])
    site.rgba = list(rgba)
    return site


def add_robot_frame_visuals(
    scene_spec: mujoco.MjSpec,
    *,
    base_body_name: str,
    eef_body_name: str,
) -> None:
    """Add base/TCP axis frames and the EEF site to a robot scene."""

    base_body = scene_spec.body(base_body_name)
    add_body_axis_frame(base_body, "base_link_frame")

    eef_body = scene_spec.body(eef_body_name)
    add_body_axis_frame(eef_body, "tcp_frame", length=0.025, radius=0.0015)
    add_eef_site(eef_body)


def configure_scene_camera(camera: mujoco.MjvCamera) -> None:
    """Point the free camera at the robot from the +X side."""

    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = VIEW_LOOKAT
    camera.distance = VIEW_DISTANCE
    camera.azimuth = VIEW_AZIMUTH
    camera.elevation = VIEW_ELEVATION


def scene_camera_headpos() -> np.ndarray:
    """World-frame camera position matching MuJoCo's free-camera spherical model."""

    lookat = np.asarray(VIEW_LOOKAT, dtype=np.float64)
    azimuth = np.deg2rad(VIEW_AZIMUTH)
    elevation = np.deg2rad(VIEW_ELEVATION)
    horizontal = VIEW_DISTANCE * np.cos(elevation)
    return np.array(
        [
            lookat[0] - horizontal * np.cos(azimuth),
            lookat[1] - horizontal * np.sin(azimuth),
            lookat[2] + VIEW_DISTANCE * np.sin(-elevation),
        ],
        dtype=np.float64,
    )


def add_scene_camera_markers(spec: mujoco.MjSpec) -> None:
    """Visualize the default free-camera lookat point, head position, and view ray."""

    if not SHOW_VIEW_CAMERA_MARKER:
        return

    lookat = np.asarray(VIEW_LOOKAT, dtype=np.float64)
    headpos = scene_camera_headpos()

    target_body = spec.worldbody.add_body(name="view_camera_target", pos=lookat.tolist())
    target_geom = target_body.add_geom(
        name="view_camera_target_geom",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.012, 0.0, 0.0],
        rgba=[0.12, 0.65, 1.0, 0.95],
    )
    target_geom.contype = 0
    target_geom.conaffinity = 0

    camera_body = spec.worldbody.add_body(name="view_camera_head", pos=headpos.tolist())
    camera_geom = camera_body.add_geom(
        name="view_camera_head_geom",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.018, 0.0, 0.0],
        rgba=[1.0, 0.72, 0.08, 0.95],
    )
    camera_geom.contype = 0
    camera_geom.conaffinity = 0

    view_ray = spec.worldbody.add_geom(
        name="view_camera_ray",
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        fromto=[*headpos, *lookat],
        size=[0.004, 0.0, 0.0],
        rgba=[1.0, 0.85, 0.15, 0.75],
    )
    view_ray.contype = 0
    view_ray.conaffinity = 0


def make_scene_camera() -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    configure_scene_camera(camera)
    return camera
