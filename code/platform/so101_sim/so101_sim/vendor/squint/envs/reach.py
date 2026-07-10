from dataclasses import asdict, dataclass
from typing import Any, Optional, Sequence, Union

import dacite
import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

import mani_skill.envs.utils.randomization as randomization
from mani_skill.utils import common
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose
from .base_random_env import DefaultCameraEnv, DefaultRandomizationConfig

from .robot.so100 import SO100
from .robot.so101 import SO101

@dataclass
class ReachRandomizationConfig(DefaultRandomizationConfig):
    """Domain randomization config for Reach task, extending wrist camera randomization."""
    # Noisy joint positions for better sim2real
    robot_qpos_noise_std: float = np.deg2rad(5)
    # Cube-specific randomization
    cube_half_size_range: Sequence[float] = (0.022 / 2, 0.028 / 2)
    # Can-specific randomization
    can_half_radius_range: Sequence[float] = (0.028 / 2, 0.038 / 2)
    can_half_height_range: Sequence[float] = (0.05 / 2, 0.07 / 2)

    item_friction_range: Sequence[float] = (0.1, 0.5)
    item_density_range: Sequence[float] = (200, 200)
    randomize_item_color: bool = False


class Reach(DefaultCameraEnv):
    """
    **Task Description:**
    A simple task where the objective is to reach above the item with the SO100 arm's TCP.

    **Randomizations:**
    - the item's xy position is randomized on top of a table
    - the item's z-axis rotation is randomized to a random angle
    - the target goal position is above the item

    **Success Conditions:**
    - the TCP position is within `goal_thresh` euclidean distance of the goal position
    - the robot is static
    """

    SUPPORTED_ROBOTS = ["so100","so101"]
    SUPPORTED_OBS_MODES = ["none", "state", "state_dict", "rgb", "rgb+segmentation", "rgb+state", "rgb+segmentation+state",
                           "rgb+depth+segmentation", "rgb+depth+segmentation+state"]
    agent: Union[SO100, SO101]

    def __init__(
        self,
        *args,
        item_type="cube",
        robot_uids="so101",
        control_mode="pd_joint_target_delta_pos",
        domain_randomization_config: Union[
            ReachRandomizationConfig, dict
        ] = ReachRandomizationConfig(),
        domain_randomization=False,
        spawn_box_pos=[0.3, 0],
        spawn_box_half_size=0.2 / 2, 
        **kwargs,
    ):
        self.item_type = item_type
        self.above_item_goal_height = 0.02  # Goal height above item
        self.target_goal_thresh = 0.02 # Goal radius
        if robot_uids == "so100":
            self.base_z_rot = np.pi / 2
            self.rest_qpos = [0, 0, 0, np.pi / 2, np.pi / 2, 0]
        elif robot_uids == "so101":
            self.base_z_rot = 0
            self.rest_qpos = SO101.keyframes["start"].qpos.tolist()
            
        # Handle domain randomization config - merge with defaults
        self.domain_randomization_config = ReachRandomizationConfig()
        merged_domain_randomization_config = self.domain_randomization_config.dict()
        if isinstance(domain_randomization_config, dict):
            common.dict_merge(merged_domain_randomization_config, domain_randomization_config)
            self.domain_randomization_config = dacite.from_dict(
                data_class=ReachRandomizationConfig,
                data=merged_domain_randomization_config,
                config=dacite.Config(strict=True),
            )
        elif isinstance(domain_randomization_config, ReachRandomizationConfig):
            self.domain_randomization_config = domain_randomization_config

        self.spawn_box_pos = spawn_box_pos
        self.spawn_box_half_size = spawn_box_half_size

        super().__init__(
            *args,
            robot_uids=robot_uids,
            control_mode=control_mode,
            domain_randomization=domain_randomization,
            domain_randomization_config=self.domain_randomization_config,
            **kwargs,
        )

    def _load_agent(self, options: dict):
        super()._load_agent(
            options,
            sapien.Pose(p=[0, 0, 0], q=euler2quat(0, 0, self.base_z_rot)),
            build_separate=True
            if self.domain_randomization
            and self.domain_randomization_config.robot_color == "random"
            else False,
        )

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(self)
        self.table_scene.build()



        if self.item_type not in ["cube", "can"]:
            raise NotImplementedError(f"Unknown item_type: {self.item_type}")

        # Default values for item geometry
        colors = np.zeros((self.num_envs, 3))
        colors[:, 0] = 1  # red
        cfg = self.domain_randomization_config
        frictions = np.ones(self.num_envs) * (cfg.item_friction_range[0] + cfg.item_friction_range[1]) / 2
        densities = np.ones(self.num_envs) * (cfg.item_density_range[0] + cfg.item_density_range[1]) / 2

        if self.item_type == "cube":
            half_sizes = (
                np.ones(self.num_envs)
                * (
                    self.domain_randomization_config.cube_half_size_range[1]
                    + self.domain_randomization_config.cube_half_size_range[0]
                )
                / 2
            )
            if self.domain_randomization:
                half_sizes = self._batched_episode_rng.uniform(
                    low=cfg.cube_half_size_range[0],
                    high=cfg.cube_half_size_range[1],
                )
                if cfg.randomize_item_color:
                    colors = self._batched_episode_rng.uniform(low=0, high=1, size=(3,))
                frictions = self._batched_episode_rng.uniform(
                    low=cfg.item_friction_range[0],
                    high=cfg.item_friction_range[1],
                )
                densities = self._batched_episode_rng.uniform(
                    low=cfg.item_density_range[0],
                    high=cfg.item_density_range[1],
                )
            self.item_half_sizes = common.to_tensor(half_sizes, device=self.device)
            self.item_dimensions = torch.stack([self.item_half_sizes] * 3, dim=-1)

        elif self.item_type == "can":
            # Blue for cans
            colors[:, 0] = 0
            colors[:, 1] = 0
            colors[:, 2] = 1.0
            half_radii = (
                np.ones(self.num_envs)
                * (
                    self.domain_randomization_config.can_half_radius_range[1]
                    + self.domain_randomization_config.can_half_radius_range[0]
                )
                / 2
            )
            half_heights = (
                np.ones(self.num_envs)
                * (
                    self.domain_randomization_config.can_half_height_range[1]
                    + self.domain_randomization_config.can_half_height_range[0]
                )
                / 2
            )
            if self.domain_randomization:
                half_radii = self._batched_episode_rng.uniform(
                    low=cfg.can_half_radius_range[0],
                    high=cfg.can_half_radius_range[1],
                )
                half_heights = self._batched_episode_rng.uniform(
                    low=cfg.can_half_height_range[0],
                    high=cfg.can_half_height_range[1],
                )
                if cfg.randomize_item_color:
                    colors = self._batched_episode_rng.uniform(low=0, high=1, size=(3,))
                frictions = self._batched_episode_rng.uniform(
                    low=cfg.item_friction_range[0],
                    high=cfg.item_friction_range[1],
                )
                densities = self._batched_episode_rng.uniform(
                    low=cfg.item_density_range[0],
                    high=cfg.item_density_range[1],
                )
            self.item_half_radii = common.to_tensor(half_radii, device=self.device)
            self.item_half_heights = common.to_tensor(half_heights, device=self.device)
            self.item_half_sizes = self.item_half_heights
            self.item_dimensions = torch.stack([self.item_half_radii, self.item_half_radii, self.item_half_heights], dim=-1)

        colors = np.concatenate([colors, np.ones((self.num_envs, 1))], axis=-1)
        self.item_frictions = common.to_tensor(frictions, device=self.device)
        self.item_densities = common.to_tensor(densities, device=self.device)

        # Build items
        items = []
        for i in range(self.num_envs):
            builder = self.scene.create_actor_builder()
            friction = frictions[i]
            material = sapien.pysapien.physx.PhysxMaterial(
                static_friction=friction,
                dynamic_friction=friction,
                restitution=0,
            )

            if self.item_type == "cube":
                builder.add_box_collision(
                    half_size=[half_sizes[i]] * 3, material=material, density=densities[i]
                )
                builder.add_box_visual(
                    half_size=[half_sizes[i]] * 3,
                    material=sapien.render.RenderMaterial(base_color=colors[i]),
                )
                builder.initial_pose = sapien.Pose(p=[0, 0, half_sizes[i]])

            elif self.item_type == "can":
                cylinder_pose = sapien.Pose(q=euler2quat(0, np.pi / 2, 0))
                builder.add_cylinder_collision(
                    radius=half_radii[i], half_length=half_heights[i], material=material, density=densities[i],
                    pose=cylinder_pose
                )
                builder.add_cylinder_visual(
                    radius=half_radii[i],
                    half_length=half_heights[i],
                    material=sapien.render.RenderMaterial(base_color=colors[i]),
                    pose=cylinder_pose
                )
                builder.initial_pose = sapien.Pose(p=[0, 0, half_heights[i]])

            builder.set_scene_idxs([i])
            item = builder.build(name=f"item-{i}")
            items.append(item)
            self.remove_from_state_dict_registry(item)

        self.item = Actor.merge(items, name="item")
        self.add_to_state_dict_registry(self.item)

        # Set up greenscreening - keep robot and item visible
        if self.apply_greenscreen:
            self.remove_object_from_greenscreen(self.agent.robot)
            self.remove_object_from_greenscreen(self.item)

        # Robot rest pose
        self.rest_qpos = common.to_tensor(self.rest_qpos, device=self.device)
        # Table pose
        self.table_pose = Pose.create_from_pq(
            p=[-0.12 + 0.737, 0, -0.9196429], q=euler2quat(0, 0, np.pi / 2)
        )

        # Build camera mount
        self._load_camera_mount()

        # Randomize robot color
        self._randomize_robot_color()

        # Add goal site
        goal_builder = self.scene.create_actor_builder()
        goal_builder.add_sphere_visual(
            radius=0.01,
            material=sapien.render.RenderMaterial(base_color=[0, 1, 0, 1]),
        )
        goal_builder.initial_pose = sapien.Pose(p=[0, 0, 0.1])
        self.goal_site = goal_builder.build_kinematic(name="goal_site")
        self._hidden_objects.append(self.goal_site)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            self.table_scene.table.set_pose(self.table_pose)

            # Random initial qpos
            self.agent.robot.set_qpos(
                self.rest_qpos + torch.randn(size=(b, self.rest_qpos.shape[-1])) * self.domain_randomization_config.initial_qpos_noise_scale
            )
            self.agent.robot.set_pose(
                Pose.create_from_pq(p=[0, 0, 0], q=euler2quat(0, 0, self.base_z_rot))
            )

            # Initialize item position
            spawn_box_pos = self.agent.robot.pose.p + torch.tensor(
                [self.spawn_box_pos[0], self.spawn_box_pos[1], 0]
            )
            xyz = torch.zeros((b, 3))
            xyz[:, :2] = (
                torch.rand((b, 2)) * self.spawn_box_half_size * 2
                - self.spawn_box_half_size
            )
            xyz[:, :2] += spawn_box_pos[env_idx, :2]
            xyz[:, 2] = self.item_half_sizes[env_idx]
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.item.set_pose(Pose.create_from_pq(xyz, qs))

            # Set goal position (above item)
            goal_xyz = xyz.clone()
            goal_xyz[:, 2] += self.item_half_sizes[env_idx] + self.above_item_goal_height
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))


    def _get_obs_agent(self):
        qpos = self.agent.robot.get_qpos()
        # Adding joint noise for better sim2real
        if self.domain_randomization and self.domain_randomization_config.robot_qpos_noise_std > 0:
            noise = torch.randn_like(qpos) * self.domain_randomization_config.robot_qpos_noise_std
            qpos = qpos + noise
        obs = dict(noisy_qpos=qpos)
        controller_state = self.agent.controller.get_state()
        if len(controller_state) > 0:
            obs.update(controller=controller_state)
        return obs

    def _get_obs_extra(self, info: dict):
        obs = dict()
        if self.obs_mode_struct.state:
            obs.update(
                qvel=self.agent.robot.get_qvel(),
                goal_pos=self.goal_site.pose.p,
                is_reached=info["is_reached"],
                item_pose=self.item.pose.raw_pose,
                tcp_pose=self.agent.tcp_pose.raw_pose,
                tcp_to_item_pos=self.item.pose.p - self.agent.tcp_pos,
                tcp_to_goal_pos=self.goal_site.pose.p - self.agent.tcp_pos,
            )
            if self.domain_randomization:
                gripper_params = self.get_gripper_params()
                obs.update(
                    clean_qpos=self.agent.robot.get_qpos(),
                    item_dimensions=self.item_dimensions,
                    item_friction=self.item_frictions,
                    item_density=self.item_densities,
                    gripper_stiffness=gripper_params["gripper_stiffness"],
                    gripper_damping=gripper_params["gripper_damping"],
                )
        return obs

    def evaluate(self):
        tcp_to_goal_dist = torch.linalg.norm(
            self.goal_site.pose.p - self.agent.tcp_pose.p, axis=1
        )
        is_reached = tcp_to_goal_dist <= self.target_goal_thresh
        is_robot_static = self.agent.is_static()

        # Close gripper reward
        gripper_min, gripper_max = self.agent.robot.get_qlimits()[0, -1, :]
        gripper_closeness = (gripper_max - self.agent.robot.get_qpos()[:, -1]) / (gripper_max - gripper_min)
        is_gripper_closed = (1 - gripper_closeness) < 0.1

        # Contact checks
        robot_touching_table = self.agent.is_touching(self.table_scene.table)
        robot_touching_item = self.agent.is_touching(self.item)

        return {
            "success": is_reached & is_robot_static & is_gripper_closed,
            "is_reached": is_reached,
            "is_robot_static": is_robot_static,
            "robot_touching_table": robot_touching_table,
            "robot_touching_item": robot_touching_item,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        tcp_to_goal_dist = torch.linalg.norm(
            self.goal_site.pose.p - self.agent.tcp_pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_goal_dist)
        reward = reaching_reward

        # Close gripper reward
        gripper_min, gripper_max = self.agent.robot.get_qlimits()[0, -1, :]
        reward += (gripper_max - self.agent.robot.get_qpos()[:, -1]) / (gripper_max - gripper_min)
        
        # Static reward when reached
        reward += info["is_reached"].float()

        # Success bonus
        reward[info["success"]] = 4

        # Touching penalty
        reward -= 3 * info["robot_touching_table"].float()
        reward -= 1 * info["robot_touching_item"].float()

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 4




@register_env("SO101ReachCube-v1", max_episode_steps=50)
class ReachCube(Reach):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, item_type="cube", **kwargs)


@register_env("SO101ReachCan-v1", max_episode_steps=50)
class ReachCan(Reach):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, item_type="can", **kwargs)
