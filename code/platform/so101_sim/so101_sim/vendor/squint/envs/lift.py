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
class LiftRandomizationConfig(DefaultRandomizationConfig):
    """Domain randomization config for Lift task, extending wrist camera randomization."""
    # Noisy joint positions for better sim2real
    robot_qpos_noise_std: float = np.deg2rad(5)
    # Cube-specific randomization
    cube_half_size_range: Sequence[float] = (0.022 / 2, 0.028 / 2)
    # Can-specific randomization
    can_radius_range: Sequence[float] = (0.028 / 2, 0.038 / 2)
    can_half_height_range: Sequence[float] = (0.05 / 2, 0.07 / 2)

    item_friction_range: Sequence[float] = (0.1, 0.5)
    item_density_range: Sequence[float] = (200, 200)
    randomize_item_color: bool = False


class Lift(DefaultCameraEnv):
    """
    **Task Description:**
    A simple task where the objective is to grasp a item with the SO100 arm and bring it up to a target rest pose.

    **Randomizations:**
    - the item's xy position is randomized on top of a table in a region of size [0.2, 0.2] x [-0.2, -0.2]. It is placed flat on the table
    - the item's z-axis rotation is randomized to a random angle

    **Success Conditions:**
    - the item is lifted, grasped, and the robot returns to a rest pose above the surface of the table
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
            LiftRandomizationConfig, dict
        ] = LiftRandomizationConfig(),
        domain_randomization=False,
        spawn_box_pos=[0.3, 0],
        spawn_box_half_size=0.2 / 2,
        **kwargs,
    ):
        self.item_type = item_type

        # Robot-specific configuration
        if robot_uids == "so100":
            self.base_z_rot = np.pi / 2
            self.rest_qpos = [0, 0, 0, np.pi / 2, np.pi / 2, 0]
        elif robot_uids == "so101":
            self.base_z_rot = 0
            self.rest_qpos = SO101.keyframes["start"].qpos.tolist()

        # Handle domain randomization config - merge with defaults
        self.domain_randomization_config = LiftRandomizationConfig()
        merged_domain_randomization_config = self.domain_randomization_config.dict()
        if isinstance(domain_randomization_config, dict):
            common.dict_merge(merged_domain_randomization_config, domain_randomization_config)
            self.domain_randomization_config = dacite.from_dict(
                data_class=LiftRandomizationConfig,
                data=merged_domain_randomization_config,
                config=dacite.Config(strict=True),
            )
        elif isinstance(domain_randomization_config, LiftRandomizationConfig):
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
        # load the robot arm at this initial pose
        super()._load_agent(
            options,
            sapien.Pose(p=[0, 0, 0], q=euler2quat(0, 0, self.base_z_rot)),
            build_separate=True
            if self.domain_randomization
            and self.domain_randomization_config.robot_color == "random"
            else False,
        )

    def _load_scene(self, options: dict):
        # we use a predefined table scene builder which simply adds a table and floor to the scene
        # where the 0, 0, 0 position is the center of the table
        self.table_scene = TableSceneBuilder(self)
        self.table_scene.build()


        if self.item_type not in ["cube", "can"]:
            raise NotImplementedError(f"Unknown item_type: {self.item_type}")

        # some default values for item geometry
        colors = np.zeros((self.num_envs, 3))
        colors[:, 0] = 1
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
            # randomize item sizes, colors, frictions, and densities
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
            colors[:, :] = 0
            colors[:, 2] = 1 # blue
            half_radii = (
                np.ones(self.num_envs)
                * (
                    self.domain_randomization_config.can_radius_range[1]
                    + self.domain_randomization_config.can_radius_range[0]
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
            # randomize can dimensions, colors, frictions, and densities
            if self.domain_randomization:
                half_radii = self._batched_episode_rng.uniform(
                    low=cfg.can_radius_range[0],
                    high=cfg.can_radius_range[1],
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
            self.item_half_sizes = self.item_half_heights  # For z-position in _initialize_episode
            self.item_dimensions = torch.stack([self.item_half_radii, self.item_half_radii, self.item_half_heights], dim=-1)

        colors = np.concatenate([colors, np.ones((self.num_envs, 1))], axis=-1)
        self.item_frictions = common.to_tensor(frictions, device=self.device)
        self.item_densities = common.to_tensor(densities, device=self.device)

        # build our items
        items = []
        for i in range(self.num_envs):
            # create a different item in each parallel environment
            # using our randomized colors, frictions, and sizes
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
                    material=sapien.render.RenderMaterial(
                        base_color=colors[i],
                    ),
                )
                builder.initial_pose = sapien.Pose(p=[0, 0, half_sizes[i]])

            elif self.item_type == "can":
                # SAPIEN cylinders have their axis along X by default, so we need to rotate
                # 90 degrees around Y-axis to make them stand upright (axis along Z)
                cylinder_pose = sapien.Pose(q=euler2quat(0, np.pi / 2, 0))
                builder.add_cylinder_collision(
                    radius=half_radii[i], half_length=half_heights[i], material=material, density=densities[i],
                    pose=cylinder_pose
                )
                builder.add_cylinder_visual(
                    radius=half_radii[i],
                    half_length=half_heights[i],
                    material=sapien.render.RenderMaterial(
                        base_color=colors[i],
                    ),
                    pose=cylinder_pose
                )
                builder.initial_pose = sapien.Pose(p=[0, 0, half_heights[i]])

            builder.set_scene_idxs([i])
            item = builder.build(name=f"item-{i}")
            items.append(item)
            self.remove_from_state_dict_registry(item)

        # since we are building many different items but simulating in parallel, we need to merge them into a single actor
        # so we can access each different item's information with a single object
        self.item = Actor.merge(items, name="item")
        self.add_to_state_dict_registry(self.item)

        # Set up greenscreening - keep robot and item visible
        if self.apply_greenscreen:
            self.remove_object_from_greenscreen(self.agent.robot)
            self.remove_object_from_greenscreen(self.item)

        # Convert rest_qpos to tensor
        self.rest_qpos = common.to_tensor(self.rest_qpos, device=self.device)
        # hardcoded pose for the table that places it such that the robot base is at 0 and on the edge of the table.
        self.table_pose = Pose.create_from_pq(
            p=[-0.12 + 0.737, 0, -0.9196429], q=euler2quat(0, 0, np.pi / 2)
        )

        # build the camera mount (from parent class)
        self._load_camera_mount()

        # randomize or set a fixed robot color (from parent class)
        self._randomize_robot_color()



    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        # we randomize the pose of the item accordingly so that the policy can learn to pick up the item from
        # many different orientations and positions.
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            # move the table back so that the robot is at 0 and on the edge of the table.
            self.table_scene.table.set_pose(self.table_pose)

            # sample a random initial joint configuration for the robot
            self.agent.robot.set_qpos(
                self.rest_qpos + torch.randn(size=(b, self.rest_qpos.shape[-1])) * self.domain_randomization_config.initial_qpos_noise_scale
            )
            self.agent.robot.set_pose(
                Pose.create_from_pq(p=[0, 0, 0], q=euler2quat(0, 0, self.base_z_rot))
            )

            # initialize the item at a random position and rotation around the z-axis
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
        # we ensure that the observation data is always retrievable in the real world, using only real world
        # available data (joint positions or the controllers target joint positions in this case).
        obs = dict()
        if self.obs_mode_struct.state:
            # state based policies can gain access to more information that helps learning
            obs.update(
                qvel=self.agent.robot.get_qvel(),
                is_item_grasped=info["is_item_grasped"],
                item_pose=self.item.pose.raw_pose,
                tcp_pos=self.agent.tcp_pose.raw_pose,
                tcp_to_item_grip_pos=self.item.pose.p - self.agent.tcp_pos,
                dist_to_rest_qpos=self.agent.controller._target_qpos[:, :-1] - self.rest_qpos[:-1],
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
        # evaluation function to generate some useful metrics/flags and evaluate the success of the task
        tcp_to_obj_dist = torch.linalg.norm(
            self.item.pose.p - self.agent.tcp_pos,
            axis=-1,
        )
        reached_object = tcp_to_obj_dist < 0.03
        is_item_grasped = self.agent.is_grasping(self.item)

        target_qpos = self.agent.controller._target_qpos.clone()  
        distance_to_rest_qpos = torch.linalg.norm(
            target_qpos[:, :-1] - self.rest_qpos[:-1], axis=-1
        )
        reached_rest_qpos = distance_to_rest_qpos < 0.2
        item_lifted = self.item.pose.p[..., -1] >= (self.item_half_sizes + 1e-3)

        # Contact checks
        robot_touching_table = self.agent.is_touching(self.table_scene.table)

        success = item_lifted & is_item_grasped & reached_rest_qpos

        return {
            "is_item_grasped": is_item_grasped,
            "reached_object": reached_object,
            "distance_to_rest_qpos": distance_to_rest_qpos,
            "robot_touching_table": robot_touching_table,
            "item_lifted": item_lifted,
            "success": success,
        }
    

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        # Reach item reward
        tcp_to_item_dist = torch.linalg.norm(self.item.pose.p - self.agent.tcp_pose.p, axis=1)
        reaching_reward = 1 - torch.tanh(5 * tcp_to_item_dist)
        reward = reaching_reward + info["is_item_grasped"]

        # Lift back reward
        place_reward = torch.exp(-2 * info["distance_to_rest_qpos"])
        reward += place_reward * info["is_item_grasped"]
        
        # Penalties
        reward -= 3 * info["robot_touching_table"].float()
        reward -= 1 * (~info["item_lifted"]).float()  # Encourage picking item fast
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        # for more stable RL we often also permit defining a noramlized reward function where you manually scale the reward down by its max value like so
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 3


@register_env("SO101LiftCube-v1", max_episode_steps=50)
class LiftCube(Lift):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, item_type="cube", **kwargs)

@register_env("SO101LiftCan-v1", max_episode_steps=50)
class LiftCan(Lift):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, item_type="can", **kwargs)
