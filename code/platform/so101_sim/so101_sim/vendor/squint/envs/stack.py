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
class StackRandomizationConfig(DefaultRandomizationConfig):
    """Domain randomization config for Stack task, extending wrist camera randomization."""
    # Noisy joint positions for better sim2real
    robot_qpos_noise_std: float = np.deg2rad(5)
    # ItemA (red cube to be stacked) - always a cube
    itemA_half_size_range: Sequence[float] = (0.022 / 2, 0.028 / 2)
    # ItemB (green base - cube or can)
    cube_half_size_range: Sequence[float] = (0.025 / 2, 0.035 / 2)  # Slightly larger
    can_radius_range: Sequence[float] = (0.028 / 2, 0.038 / 2)
    can_half_height_range: Sequence[float] = (0.05 / 2, 0.07 / 2)

    item_friction_range: Sequence[float] = (0.1, 0.5)
    item_density_range: Sequence[float] = (200, 200)
    randomize_item_color: bool = False  # Keep colors distinct (red/blue)


class Stack(DefaultCameraEnv):
    """
    **Task Description:**
    Pick up itemA (red cube) and stack it on top of itemB (blue cube or can).

    **Randomizations:**
    - both items have their xy positions randomized (non-overlapping)
    - both items have their z-axis rotation randomized
    - item sizes are randomized within configured ranges

    **Success Conditions:**
    - itemA is on top of itemB
    - itemA is static
    - itemA is not being grasped
    - robot is static
    """

    SUPPORTED_ROBOTS = ["so100", "so101"]
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
            StackRandomizationConfig, dict
        ] = StackRandomizationConfig(),
        domain_randomization=False,
        spawn_box_pos=[0.3, 0],
        spawn_box_half_size=0.2 / 2,
        **kwargs,
    ):
        self.item_type = item_type  # Type of itemB (base)

        # Robot-specific configuration
        if robot_uids == "so100":
            self.base_z_rot = np.pi / 2
            self.rest_qpos = [0, 0, 0, np.pi / 2, np.pi / 2, 0]
        elif robot_uids == "so101":
            self.base_z_rot = 0
            self.rest_qpos = SO101.keyframes["start"].qpos.tolist()

        # Handle domain randomization config
        self.domain_randomization_config = StackRandomizationConfig()
        merged_domain_randomization_config = self.domain_randomization_config.dict()
        if isinstance(domain_randomization_config, dict):
            common.dict_merge(merged_domain_randomization_config, domain_randomization_config)
            self.domain_randomization_config = dacite.from_dict(
                data_class=StackRandomizationConfig,
                data=merged_domain_randomization_config,
                config=dacite.Config(strict=True),
            )
        elif isinstance(domain_randomization_config, StackRandomizationConfig):
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

        # Default friction and density
        cfg = self.domain_randomization_config
        frictions = np.ones(self.num_envs) * (cfg.item_friction_range[0] + cfg.item_friction_range[1]) / 2
        densities = np.ones(self.num_envs) * (cfg.item_density_range[0] + cfg.item_density_range[1]) / 2
        if self.domain_randomization:
            frictions = self._batched_episode_rng.uniform(
                low=cfg.item_friction_range[0],
                high=cfg.item_friction_range[1],
            )
            densities = self._batched_episode_rng.uniform(
                low=cfg.item_density_range[0],
                high=cfg.item_density_range[1],
            )
        self.item_frictions = common.to_tensor(frictions, device=self.device)
        self.item_densities = common.to_tensor(densities, device=self.device)

        # ========== Build ItemA (red cube - always a cube) ==========
        itemA_half_sizes = (
            np.ones(self.num_envs)
            * (
                self.domain_randomization_config.itemA_half_size_range[1]
                + self.domain_randomization_config.itemA_half_size_range[0]
            )
            / 2
        )
        if self.domain_randomization:
            itemA_half_sizes = self._batched_episode_rng.uniform(
                low=cfg.itemA_half_size_range[0],
                high=cfg.itemA_half_size_range[1],
            )
        self.itemA_half_sizes = common.to_tensor(itemA_half_sizes, device=self.device)
        self.itemA_dimensions = torch.stack([self.itemA_half_sizes] * 3, dim=-1)


        # ItemA colors (red)
        colorsA = np.zeros((self.num_envs, 4))
        colorsA[:, 0] = 1  # Red
        colorsA[:, 3] = 1  # Alpha

        itemsA = []
        for i in range(self.num_envs):
            builder = self.scene.create_actor_builder()
            friction = frictions[i]
            material = sapien.pysapien.physx.PhysxMaterial(
                static_friction=friction,
                dynamic_friction=friction,
                restitution=0,
            )
            builder.add_box_collision(
                half_size=[itemA_half_sizes[i]] * 3, material=material, density=densities[i]
            )
            builder.add_box_visual(
                half_size=[itemA_half_sizes[i]] * 3,
                material=sapien.render.RenderMaterial(base_color=colorsA[i]),
            )
            builder.initial_pose = sapien.Pose(p=[0.2, 0, itemA_half_sizes[i]])  # Offset to avoid collision with itemB at creation
            builder.set_scene_idxs([i])
            item = builder.build(name=f"itemA-{i}")
            itemsA.append(item)
            self.remove_from_state_dict_registry(item)

        self.itemA = Actor.merge(itemsA, name="itemA")
        self.add_to_state_dict_registry(self.itemA)

        # ========== Build ItemB (blue - cube or can) ==========
        # ItemB colors (blue)
        colorsB = np.zeros((self.num_envs, 4))
        colorsB[:, 2] = 1  # Blue
        colorsB[:, 3] = 1  # Alpha

        if self.item_type == "cube":
            itemB_half_sizes = (
                np.ones(self.num_envs)
                * (
                    self.domain_randomization_config.cube_half_size_range[1]
                    + self.domain_randomization_config.cube_half_size_range[0]
                )
                / 2
            )
            if self.domain_randomization:
                itemB_half_sizes = self._batched_episode_rng.uniform(
                    low=cfg.cube_half_size_range[0],
                    high=cfg.cube_half_size_range[1],
                )
            self.itemB_half_sizes = common.to_tensor(itemB_half_sizes, device=self.device)
            self.itemB_dimensions = torch.stack([self.itemB_half_sizes] * 3, dim=-1)

            itemsB = []
            for i in range(self.num_envs):
                builder = self.scene.create_actor_builder()
                friction = frictions[i]
                material = sapien.pysapien.physx.PhysxMaterial(
                    static_friction=friction,
                    dynamic_friction=friction,
                    restitution=0,
                )
                builder.add_box_collision(
                    half_size=[itemB_half_sizes[i]] * 3, material=material, density=densities[i]
                )
                builder.add_box_visual(
                    half_size=[itemB_half_sizes[i]] * 3,
                    material=sapien.render.RenderMaterial(base_color=colorsB[i]),
                )
                builder.initial_pose = sapien.Pose(p=[-0.2, 0, itemB_half_sizes[i]])  # Offset to avoid collision with itemA at creation
                builder.set_scene_idxs([i])
                item = builder.build(name=f"itemB-{i}")
                itemsB.append(item)
                self.remove_from_state_dict_registry(item)

        elif self.item_type == "can":
            itemB_half_radii = (
                np.ones(self.num_envs)
                * (
                    self.domain_randomization_config.can_radius_range[1]
                    + self.domain_randomization_config.can_radius_range[0]
                )
                / 2
            )
            itemB_half_heights = (
                np.ones(self.num_envs)
                * (
                    self.domain_randomization_config.can_half_height_range[1]
                    + self.domain_randomization_config.can_half_height_range[0]
                )
                / 2
            )
            if self.domain_randomization:
                itemB_half_radii = self._batched_episode_rng.uniform(
                    low=cfg.can_radius_range[0],
                    high=cfg.can_radius_range[1],
                )
                itemB_half_heights = self._batched_episode_rng.uniform(
                    low=cfg.can_half_height_range[0],
                    high=cfg.can_half_height_range[1],
                )
            self.itemB_half_radii = common.to_tensor(itemB_half_radii, device=self.device)
            self.itemB_half_heights = common.to_tensor(itemB_half_heights, device=self.device)
            self.itemB_half_sizes = self.itemB_half_heights
            self.itemB_dimensions = torch.stack([self.itemB_half_radii, self.itemB_half_radii, self.itemB_half_heights], dim=-1)


            itemsB = []
            for i in range(self.num_envs):
                builder = self.scene.create_actor_builder()
                friction = frictions[i]
                material = sapien.pysapien.physx.PhysxMaterial(
                    static_friction=friction,
                    dynamic_friction=friction,
                    restitution=0,
                )
                cylinder_pose = sapien.Pose(q=euler2quat(0, np.pi / 2, 0))
                builder.add_cylinder_collision(
                    radius=itemB_half_radii[i], half_length=itemB_half_heights[i],
                    material=material, density=densities[i], pose=cylinder_pose
                )
                builder.add_cylinder_visual(
                    radius=itemB_half_radii[i],
                    half_length=itemB_half_heights[i],
                    material=sapien.render.RenderMaterial(base_color=colorsB[i]),
                    pose=cylinder_pose
                )
                builder.initial_pose = sapien.Pose(p=[-0.2, 0, itemB_half_heights[i]])  # Offset to avoid collision with itemA at creation
                builder.set_scene_idxs([i])
                item = builder.build(name=f"itemB-{i}")
                itemsB.append(item)
                self.remove_from_state_dict_registry(item)

        self.itemB = Actor.merge(itemsB, name="itemB")
        self.add_to_state_dict_registry(self.itemB)

        # Set up greenscreening - keep robot, itemA, and itemB visible
        if self.apply_greenscreen:
            self.remove_object_from_greenscreen(self.agent.robot)
            self.remove_object_from_greenscreen(self.itemA)
            self.remove_object_from_greenscreen(self.itemB)

        # Convert rest_qpos to tensor
        self.rest_qpos = common.to_tensor(self.rest_qpos, device=self.device)
        # Table pose
        self.table_pose = Pose.create_from_pq(
            p=[-0.12 + 0.737, 0, -0.9196429], q=euler2quat(0, 0, np.pi / 2)
        )

        # Build camera mount
        self._load_camera_mount()

        # Randomize robot color
        self._randomize_robot_color()

        # Goal site
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

            # Sample positions
            spawn_center = self.agent.robot.pose.p + torch.tensor(
                [self.spawn_box_pos[0], self.spawn_box_pos[1], 0]
            )

            region = [
                [-self.spawn_box_half_size, -self.spawn_box_half_size],
                [self.spawn_box_half_size, self.spawn_box_half_size]
            ]
            sampler = randomization.UniformPlacementSampler(
                bounds=region, batch_size=b, device=self.device
            )

            # Collision radii for placement (use max half_size + margin)
            cfg = self.domain_randomization_config
            collision_margin = 0.01
            itemA_radius = cfg.itemA_half_size_range[1] + collision_margin
            if self.item_type == "cube":
                itemB_radius = cfg.cube_half_size_range[1] + collision_margin
            else:  # can
                itemB_radius = cfg.can_radius_range[1] + collision_margin

            itemA_xy_offset = sampler.sample(itemA_radius, 100)
            itemB_xy_offset = sampler.sample(itemB_radius, 100, verbose=False)

            # Set itemA pose (red cube)
            itemA_xyz = torch.zeros((b, 3))
            itemA_xyz[:, :2] = spawn_center[env_idx, :2] + itemA_xy_offset
            itemA_xyz[:, 2] = self.itemA_half_sizes[env_idx]
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.itemA.set_pose(Pose.create_from_pq(itemA_xyz, qs))

            # Set itemB pose (green base)
            itemB_xyz = torch.zeros((b, 3))
            itemB_xyz[:, :2] = spawn_center[env_idx, :2] + itemB_xy_offset
            itemB_xyz[:, 2] = self.itemB_half_sizes[env_idx]
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.itemB.set_pose(Pose.create_from_pq(itemB_xyz, qs))

            # Goal is on top of itemB
            goal_xyz = itemB_xyz.clone()
            goal_xyz[:, 2] += self.itemB_half_sizes[env_idx] + self.itemA_half_sizes[env_idx]
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
                tcp_pose=self.agent.tcp_pose.raw_pose,
                is_itemA_grasped=info["is_itemA_grasped"],
                itemA_pose=self.itemA.pose.raw_pose,
                itemB_pose=self.itemB.pose.raw_pose,
                tcp_to_itemA_pos=self.itemA.pose.p - self.agent.tcp_pos,
                tcp_to_itemB_pos=self.itemB.pose.p - self.agent.tcp_pos,
                itemA_to_itemB_pos=self.itemB.pose.p - self.itemA.pose.p,
            )
            if self.domain_randomization:
                gripper_params = self.get_gripper_params()
                obs.update(
                    clean_qpos=self.agent.robot.get_qpos(),
                    itemA_dimensions=self.itemA_dimensions,
                    itemB_dimensions=self.itemB_dimensions,
                    item_friction=self.item_frictions,
                    item_density=self.item_densities,
                    gripper_stiffness=gripper_params["gripper_stiffness"],
                    gripper_damping=gripper_params["gripper_damping"],
                )
        return obs

    def evaluate(self):
        posA = self.itemA.pose.p
        posB = self.itemB.pose.p

        offset = posA - posB
        xy_dist = torch.linalg.norm(offset[:, :2], axis=1)
        xy_flag = (xy_dist <= 0.02)

        # Z offset should be itemA_half + itemB_half
        expected_z_offset = self.itemA_half_sizes + self.itemB_half_sizes
        z_dist = torch.abs(offset[:, 2] - expected_z_offset)
        z_flag = (z_dist <= 0.01)

        is_itemA_on_itemB = xy_flag & z_flag

        itemA_vel = torch.linalg.norm(self.itemA.linear_velocity, axis=-1)
        is_itemA_static = itemA_vel <= 2e-2
        is_itemA_grasped = self.agent.is_grasping(self.itemA)
        is_itemA_lifted = self.itemA.pose.p[..., -1] >= (self.itemA_half_sizes + 1e-3)
        is_robot_static = self.agent.is_static()

        # Contact checks
        robot_touching_table = self.agent.is_touching(self.table_scene.table)
        robot_touching_itemA = self.agent.is_touching(self.itemA)

        success = is_itemA_on_itemB & is_itemA_static & (~robot_touching_itemA) & is_robot_static

        return {
            "xy_dist": xy_dist,
            "z_dist": z_dist,
            "itemA_vel": itemA_vel,

            "success": success,
            "is_itemA_on_itemB": is_itemA_on_itemB,
            "is_itemA_static": is_itemA_static,
            "is_itemA_grasped": is_itemA_grasped,
            "is_itemA_lifted": is_itemA_lifted,
            "is_robot_static": is_robot_static,
            "robot_touching_table": robot_touching_table,
            "robot_touching_itemA": robot_touching_itemA,
        }



    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        # Reaching reward (TCP to itemA)
        tcp_pos = self.agent.tcp_pose.p
        itemA_pos = self.itemA.pose.p
        tcp_to_itemA_dist = torch.linalg.norm(tcp_pos - itemA_pos, axis=1)
        reaching_reward = 2 * (1 - torch.tanh(5 * tcp_to_itemA_dist))
        reward = reaching_reward

        # Complex place reward (itemA to goal on top of itemB)
        itemB_pos = self.itemB.pose.p
        goal_z = itemB_pos[:, 2] + self.itemB_half_sizes + self.itemA_half_sizes
        goal_xyz = torch.cat([itemB_pos[:, :2], goal_z.unsqueeze(1)], dim=1)

        # Overall distance reward
        itemA_to_goal_dist = torch.linalg.norm(goal_xyz - itemA_pos, axis=1)
        place_reward_final = 1 - torch.tanh(5.0 * itemA_to_goal_dist)

        # XY and Z distance with far/close logic
        itemA_to_goal_dist_xy = torch.linalg.norm(goal_xyz[..., :2] - itemA_pos[..., :2], dim=1)
        # Far: target is 0.03m above the goal (encourages lifting before placing)
        itemA_to_goal_dist_z_far = torch.linalg.norm(
            (goal_xyz[..., 2:] + 0.03) - itemA_pos[..., 2:], dim=1
        )
        # Close: target is final position
        itemA_to_goal_dist_z_close = torch.linalg.norm(goal_xyz[..., 2:] - itemA_pos[..., 2:], dim=1)
        itemA_close_to_goal = (itemA_to_goal_dist_xy <= 0.04)
        itemA_to_goal_dist_z = torch.where(itemA_close_to_goal, itemA_to_goal_dist_z_close, itemA_to_goal_dist_z_far)
        place_reward_z = 1 - torch.tanh(10.0 * itemA_to_goal_dist_z)
        place_reward = place_reward_final + place_reward_z

        # Ungrasp reward (inverted from Reach's close gripper)
        gripper_min, gripper_max = self.agent.robot.get_qlimits()[0, -1, :]
        ungrasp_reward = (self.agent.robot.get_qpos()[:, -1] - gripper_min) / (gripper_max - gripper_min)

        # Grasped: 3 + place_reward
        reward[info["is_itemA_grasped"]] = (3 + place_reward)[info["is_itemA_grasped"]]

        # On itemB (still grasped): 4 + place_reward + gripper_openness
        is_on_itemB_and_grasped = info["is_itemA_on_itemB"] & info["robot_touching_itemA"]
        reward[is_on_itemB_and_grasped] = (4 + place_reward + ungrasp_reward)[is_on_itemB_and_grasped]

        # On itemB and released (not grasped): 7 + static_itemA_reward + static_robot_reward
        itemA_v = torch.linalg.norm(self.itemA.linear_velocity, axis=1)
        robot_v = torch.linalg.norm(self.agent.robot.get_qvel()[:, :-1], axis=1) 
        static_itemA_reward = 1 - torch.tanh(itemA_v * 10)
        static_robot_reward = 1 - torch.tanh(robot_v * 10)
        is_on_itemB_and_released = info["is_itemA_on_itemB"] & (~info["robot_touching_itemA"])
        reward[is_on_itemB_and_released] = (7 + (static_itemA_reward + static_robot_reward) / 2.0)[is_on_itemB_and_released]

        # Success
        reward[info["success"]] = 9

        # Penalties
        reward -= 6 * info["robot_touching_table"].float()
        reward -= 1 * (~info["is_itemA_lifted"]).float()  # Encourage picking item fast

        return reward




    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 9


@register_env("SO101StackCube-v1", max_episode_steps=50)
class StackCube(Stack):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, item_type="cube", **kwargs)


@register_env("SO101StackCan-v1", max_episode_steps=50)
class StackCan(Stack):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, item_type="can", **kwargs)
