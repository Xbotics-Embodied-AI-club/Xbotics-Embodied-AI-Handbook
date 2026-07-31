"""Gymnasium environment wrapper for Lecture 07 MuJoCo pick-place scenes."""

from __future__ import annotations

import threading
import time
from typing import Any

import gymnasium as gym
import mujoco
from mujoco import viewer as mujoco_viewer
import numpy as np
from gymnasium import spaces

from .utils.collision import configure_collision_debug_view, configure_collision_debug_viewer
from .scene import (
    HOME_QPOS,
    JOINT_NAMES,
    MODEL_DIR,
    build_model,
)
from .utils.mesh_assets import URDF_PATH
from ..viz.scene_viz import configure_scene_camera, make_scene_camera


class Lecture07MuJoCoEnv(gym.Env):
    """Single-environment MuJoCo scene for Lecture 07 pick-place demos."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        *,
        task: str,
        render_mode: str | None = None,
        show_collision_debug: bool = False,
        show_grasp: bool = False,
        show_grasp_axes: bool = True,
    ) -> None:
        super().__init__()
        self.task = task
        self.render_mode = render_mode
        self.show_collision_debug = show_collision_debug

        self.model = build_model(
            task,
            show_grasp=show_grasp,
            show_grasp_axes=show_grasp_axes,
        )
        if show_collision_debug:
            configure_collision_debug_view(self.model)
        self.data = mujoco.MjData(self.model)
        self._viewer: Any = None
        self._renderer: mujoco.Renderer | None = None
        self._scene_camera = make_scene_camera()

        self._joint_qpos_ids = np.array(
            [self.model.joint(name).qposadr[0] for name in JOINT_NAMES],
            dtype=np.int32,
        )
        self._joint_dof_ids = np.array(
            [self.model.joint(name).dofadr[0] for name in JOINT_NAMES],
            dtype=np.int32,
        )
        self._eef_site_id = self.model.site("eef_site").id

        action_dim = len(JOINT_NAMES)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(action_dim,),
            dtype=np.float32,
        )
        obs_dim = len(JOINT_NAMES) + 7
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self.reset(seed=0)

        print(f"task={self.task}")
        print(f"assets={MODEL_DIR}")
        print(f"observation_space={self.observation_space}")
        print(f"action_space={self.action_space}")

    def _eef_pose(self) -> tuple[np.ndarray, np.ndarray]:
        pos = self.data.site_xpos[self._eef_site_id].copy()
        mat = self.data.site_xmat[self._eef_site_id].reshape(3, 3)
        quat = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quat, mat.reshape(-1))
        return pos, quat

    def _build_observation(self) -> np.ndarray:
        joint_qpos = self.data.qpos[self._joint_qpos_ids].astype(np.float32)
        eef_pos, eef_quat = self._eef_pose()
        return np.concatenate([joint_qpos, eef_pos.astype(np.float32), eef_quat.astype(np.float32)])

    def _hold_robot_at_home(self) -> None:
        """Keep the arm fixed in HOME_QPOS during passive scene viewing."""

        self.data.qpos[self._joint_qpos_ids] = HOME_QPOS
        self.data.qvel[self._joint_dof_ids] = 0.0
        self.data.ctrl[:] = HOME_QPOS
        mujoco.mj_forward(self.model, self.data)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self._joint_qpos_ids] = HOME_QPOS
        self.data.ctrl[:] = HOME_QPOS
        mujoco.mj_forward(self.model, self.data)
        if self.task == "bottle":
            from . import bottle

            bottle.place_bottle_on_table(self.model, self.data)
        return self._build_observation(), {"task": self.task, "urdf": str(URDF_PATH)}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        delta_scale = np.array([0.02, 0.02, 0.02, 0.02, 0.02, 0.01], dtype=np.float64)
        current = self.data.qpos[self._joint_qpos_ids].copy()
        target = current + action * delta_scale
        for idx, joint_name in enumerate(JOINT_NAMES):
            low, high = self.model.joint(joint_name).range
            target[idx] = np.clip(target[idx], low, high)
        self.data.ctrl[:] = target

        collision = False
        for _ in range(25):
            mujoco.mj_step(self.model, self.data)
            if self.data.ncon > 0:
                for contact_idx in range(self.data.ncon):
                    geom1 = self.data.contact[contact_idx].geom1
                    geom2 = self.data.contact[contact_idx].geom2
                    name1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom1) or ""
                    name2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom2) or ""
                    pair = {name1, name2}
                    if pair & {"tabletop_geom", "floor"}:
                        continue
                    if "gripper" in name1 or "gripper" in name2 or "finger" in name1 or "finger" in name2:
                        continue
                    if any("so101_" in name for name in pair):
                        collision = True
                        break
                if collision:
                    break

        obs = self._build_observation()
        info = {"collision": collision, "task": self.task}
        return obs, 0.0, False, False, info

    def render(self) -> np.ndarray | None:
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            self._renderer.update_scene(self.data, camera=self._scene_camera)
            return self._renderer.render()
        if self.render_mode == "human":
            if self._viewer is None:
                self._viewer = mujoco_viewer.launch_passive(self.model, self.data)
                configure_scene_camera(self._viewer.cam)
                configure_collision_debug_viewer(self._viewer, enabled=self.show_collision_debug)
            self._viewer.sync()
        return None

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


def make_task_environment(
    task: str,
    *,
    render_mode: str | None = None,
    rotation_z: float = 0.0,
    show_collision_debug: bool = False,
    show_grasp: bool = False,
    show_grasp_axes: bool = True,
) -> Lecture07MuJoCoEnv:
    del rotation_z
    if task not in {"cube", "bottle"}:
        raise ValueError(f"unknown task {task!r}")
    return Lecture07MuJoCoEnv(
        task=task,
        render_mode=render_mode,
        show_collision_debug=show_collision_debug,
        show_grasp=show_grasp,
        show_grasp_axes=show_grasp_axes,
    )


def make_environment(
    task: str,
    *,
    render_mode: str | None = None,
    show_collision_debug: bool = False,
    show_grasp: bool = False,
    show_grasp_axes: bool = True,
) -> Lecture07MuJoCoEnv:
    """Create a Gym environment for the given Lecture 07 task."""

    return make_task_environment(
        task,
        render_mode=render_mode,
        show_collision_debug=show_collision_debug,
        show_grasp=show_grasp,
        show_grasp_axes=show_grasp_axes,
    )


def show_until_enter(env: Lecture07MuJoCoEnv) -> None:
    stop = threading.Event()

    def wait_for_enter() -> None:
        input("窗口已打开，按 Enter 关闭仿真... ")
        stop.set()

    threading.Thread(target=wait_for_enter, daemon=True).start()
    if env.render_mode != "human":
        env.render_mode = "human"

    while not stop.is_set():
        try:
            env._hold_robot_at_home()
            mujoco.mj_forward(env.model, env.data)
            env.render()
        except RuntimeError as exc:
            if "width and height must be positive" not in str(exc):
                raise
            stop.set()
            print("仿真窗口已关闭。")
            break
        time.sleep(0.02)
