from __future__ import annotations

import tempfile
import torch
import carb

from pink.tasks import FrameTask

import isaaclab.controllers.utils as ControllerUtils
import isaaclab.envs.mdp as base_mdp
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.controllers.pink_ik import NullSpacePostureTask, PinkIKControllerCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.openxr import ManusViveCfg, OpenXRDeviceCfg, XrCfg
from isaaclab.devices.openxr.retargeters.humanoid.unitree.inspire.g1_upper_body_retargeter import (
    UnitreeG1RetargeterCfg,
)
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.pink_actions_cfg import PinkInverseKinematicsActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils

from isaaclab_assets.robots.unitree import G1_INSPIRE_FTP_CFG  # isort: skip

from . import object_table_env_cfg
from . import mdp

from spark_tasks.data.bowl import BOWL_CFG
from spark_tasks.data.glassware import GLASSWARE_CFG


# Local BALL configuration (similar to Ego BALL_CFG)
BALL_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/Ball",
    spawn=sim_utils.SphereCfg(
        radius=0.006,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.0, 0.0)),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_linear_velocity=0.5,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=0.002,
            rest_offset=0.001,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.001),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.56, 0.35, 0.76)),
)


@configclass
class ObjectTableSceneCfg(object_table_env_cfg.ObjectTableSceneCfg):
    # bottle that initially contains balls
    bottle: RigidObjectCfg = GLASSWARE_CFG.replace(prim_path="/World/envs/env_.*/bottle")  # type: ignore
    bottle.init_state = RigidObjectCfg.InitialStateCfg(
        pos=(0.45, -0.3, 0.715),
        rot=(1.0, 0.0, 0.0, 0.0),
    )

    # bowl that balls are poured into
    bowl: RigidObjectCfg = BOWL_CFG.replace(prim_path="/World/envs/env_.*/bowl")  # type: ignore
    bowl.init_state = RigidObjectCfg.InitialStateCfg(
        pos=(0.54, 0.0, 0.715),
        rot=(1.0, 0.0, 0.0, 0.0),
    )
    bowl.spawn.scale = (0.6, 0.6, 0.6)  # type: ignore

    # balls (kept close to Ego layout, Z adapted to table height ~0.76) # 堆叠放置，启动后应该会自行打乱掉下来？
    ball1: RigidObjectCfg = BALL_CFG.replace(prim_path="/World/envs/env_.*/ball1")  # type: ignore
    ball1.init_state = RigidObjectCfg.InitialStateCfg(pos=(0.46, -0.31, 0.78))

    ball2: RigidObjectCfg = BALL_CFG.replace(prim_path="/World/envs/env_.*/ball2")  # type: ignore
    ball2.init_state = RigidObjectCfg.InitialStateCfg(pos=(0.46, -0.29, 0.78))

    ball3: RigidObjectCfg = BALL_CFG.replace(prim_path="/World/envs/env_.*/ball3")  # type: ignore
    ball3.init_state = RigidObjectCfg.InitialStateCfg(pos=(0.44, -0.31, 0.78))

    ball4: RigidObjectCfg = BALL_CFG.replace(prim_path="/World/envs/env_.*/ball4")  # type: ignore
    ball4.init_state = RigidObjectCfg.InitialStateCfg(pos=(0.44, -0.29, 0.78))

    ball5: RigidObjectCfg = BALL_CFG.replace(prim_path="/World/envs/env_.*/ball5")  # type: ignore
    ball5.init_state = RigidObjectCfg.InitialStateCfg(pos=(0.46, -0.31, 0.75))

    ball6: RigidObjectCfg = BALL_CFG.replace(prim_path="/World/envs/env_.*/ball6")  # type: ignore
    ball6.init_state = RigidObjectCfg.InitialStateCfg(pos=(0.46, -0.29, 0.75))

    ball7: RigidObjectCfg = BALL_CFG.replace(prim_path="/World/envs/env_.*/ball7")  # type: ignore
    ball7.init_state = RigidObjectCfg.InitialStateCfg(pos=(0.44, -0.31, 0.75))

    ball8: RigidObjectCfg = BALL_CFG.replace(prim_path="/World/envs/env_.*/ball8")  # type: ignore
    ball8.init_state = RigidObjectCfg.InitialStateCfg(pos=(0.44, -0.29, 0.75))

    robot: ArticulationCfg = G1_INSPIRE_FTP_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.80),
            joint_pos={
                "right_shoulder_pitch_joint": 0.0,
                "right_shoulder_roll_joint": 0.0,
                "right_shoulder_yaw_joint": 0.0,
                "right_elbow_joint": 0.0,
                "right_wrist_yaw_joint": 0.0,
                "right_wrist_roll_joint": 0.0,
                "right_wrist_pitch_joint": 0.0,
                "left_shoulder_pitch_joint": 0.0,
                "left_shoulder_roll_joint": 0.0,
                "left_shoulder_yaw_joint": 0.0,
                "left_elbow_joint": 0.0,
                "left_wrist_yaw_joint": 0.0,
                "left_wrist_roll_joint": 0.0,
                "left_wrist_pitch_joint": 0.0,
                "waist_.*": 0.0,
                ".*_hip_.*": 0.0,
                ".*_knee_.*": 0.0,
                ".*_ankle_.*": 0.0,
                ".*_thumb_.*": 0.0,
                ".*_index_.*": 0.0,
                ".*_middle_.*": 0.0,
                ".*_ring_.*": 0.0,
                ".*_pinky_.*": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
    )


@configclass
class ActionsCfg:
    pink_ik_cfg = PinkInverseKinematicsActionCfg(
        pink_controlled_joint_names=[
            ".*_shoulder_pitch_joint",
            ".*_shoulder_roll_joint",
            ".*_shoulder_yaw_joint",
            ".*_elbow_joint",
            ".*_wrist_yaw_joint",
            ".*_wrist_roll_joint",
            ".*_wrist_pitch_joint",
        ],
        hand_joint_names=[
            "L_index_proximal_joint",
            "L_middle_proximal_joint",
            "L_pinky_proximal_joint",
            "L_ring_proximal_joint",
            "L_thumb_proximal_yaw_joint",
            "R_index_proximal_joint",
            "R_middle_proximal_joint",
            "R_pinky_proximal_joint",
            "R_ring_proximal_joint",
            "R_thumb_proximal_yaw_joint",
            "L_index_intermediate_joint",
            "L_middle_intermediate_joint",
            "L_pinky_intermediate_joint",
            "L_ring_intermediate_joint",
            "L_thumb_proximal_pitch_joint",
            "R_index_intermediate_joint",
            "R_middle_intermediate_joint",
            "R_pinky_intermediate_joint",
            "R_ring_intermediate_joint",
            "R_thumb_proximal_pitch_joint",
            "L_thumb_intermediate_joint",
            "R_thumb_intermediate_joint",
            "L_thumb_distal_joint",
            "R_thumb_distal_joint",
        ],
        target_eef_link_names={
            "left_wrist": "left_wrist_yaw_link",
            "right_wrist": "right_wrist_yaw_link",
        },
        asset_name="robot",
        controller=PinkIKControllerCfg(
            articulation_name="robot",
            base_link_name="pelvis",
            num_hand_joints=24,
            show_ik_warnings=False,
            fail_on_joint_limit_violation=False,
            variable_input_tasks=[
                FrameTask(
                    "g1_29dof_rev_1_0_left_wrist_yaw_link",
                    position_cost=8.0,
                    orientation_cost=2.0,
                    lm_damping=10.0,
                    gain=0.5,
                ),
                FrameTask(
                    "g1_29dof_rev_1_0_right_wrist_yaw_link",
                    position_cost=8.0,
                    orientation_cost=2.0,
                    lm_damping=10.0,
                    gain=0.5,
                ),
                NullSpacePostureTask(
                    cost=0.5,
                    lm_damping=1.0,
                    controlled_frames=[
                        "g1_29dof_rev_1_0_left_wrist_yaw_link",
                        "g1_29dof_rev_1_0_right_wrist_yaw_link",
                    ],
                    controlled_joints=[
                        "left_shoulder_pitch_joint",
                        "left_shoulder_roll_joint",
                        "left_shoulder_yaw_joint",
                        "right_shoulder_pitch_joint",
                        "right_shoulder_roll_joint",
                        "right_shoulder_yaw_joint",
                        "waist_yaw_joint",
                        "waist_pitch_joint",
                        "waist_roll_joint",
                    ],
                    gain=0.3,
                ),
            ],
            fixed_input_tasks=[],
            xr_enabled=bool(carb.settings.get_settings().get("/app/xr/enabled")),
        ),
        enable_gravity_compensation=False,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        actions = ObsTerm(func=base_mdp.last_action)
        robot_joint_pos = ObsTerm(func=base_mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot")})
        robot_root_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("robot")})
        robot_root_rot = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("robot")})
        robot_links_state = ObsTerm(func=mdp.get_all_robot_link_state)

        left_eef_pos = ObsTerm(func=mdp.get_eef_pos, params={"link_name": "left_wrist_yaw_link"})
        left_eef_quat = ObsTerm(func=mdp.get_eef_quat, params={"link_name": "left_wrist_yaw_link"})
        right_eef_pos = ObsTerm(func=mdp.get_eef_pos, params={"link_name": "right_wrist_yaw_link"})
        right_eef_quat = ObsTerm(func=mdp.get_eef_quat, params={"link_name": "right_wrist_yaw_link"})
        hand_joint_state = ObsTerm(func=mdp.get_robot_joint_state, params={"joint_names": ["R_.*", "L_.*"]})

        bottle_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("bottle")})
        bottle_rot = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("bottle")})

        bowl_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("bowl")})
        bowl_rot = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("bowl")})

        ball1_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("ball1")})
        ball2_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("ball2")})
        ball3_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("ball3")})
        ball4_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("ball4")})
        ball5_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("ball5")})
        ball6_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("ball6")})
        ball7_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("ball7")})
        ball8_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("ball8")})

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=mdp.task_done_pour_balls,
        params={
            "bowl_cfg": SceneEntityCfg("bowl"),
            "ball1_cfg": SceneEntityCfg("ball1"),
            "ball2_cfg": SceneEntityCfg("ball2"),
            "ball3_cfg": SceneEntityCfg("ball3"),
            "ball4_cfg": SceneEntityCfg("ball4"),
            "ball5_cfg": SceneEntityCfg("ball5"),
            "ball6_cfg": SceneEntityCfg("ball6"),
            "ball7_cfg": SceneEntityCfg("ball7"),
            "ball8_cfg": SceneEntityCfg("ball8"),
            "surface_center_offset": (0.0, 0.0, 0.075),
            "surface_radius": 0.085,
            "min_poured_balls": 3,
            "angle_tolerance_deg": 10.0,
        },
    )


@configclass
class EventCfg:
    reset_all = EventTerm(func=base_mdp.reset_scene_to_default, mode="reset")

    reset_pour_balls = EventTerm(
        func=mdp.reset_pour_balls_objects,
        mode="reset",
        params={
            "bottle_cfg": SceneEntityCfg("bottle"),
            "bowl_cfg": SceneEntityCfg("bowl"),
            "ball1_cfg": SceneEntityCfg("ball1"),
            "ball2_cfg": SceneEntityCfg("ball2"),
            "ball3_cfg": SceneEntityCfg("ball3"),
            "ball4_cfg": SceneEntityCfg("ball4"),
            "ball5_cfg": SceneEntityCfg("ball5"),
            "ball6_cfg": SceneEntityCfg("ball6"),
            "ball7_cfg": SceneEntityCfg("ball7"),
            "ball8_cfg": SceneEntityCfg("ball8"),
        },
    )


@configclass
class G1InspireFTPEnvCfg(ManagerBasedRLEnvCfg):
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(
        num_envs=1,
        env_spacing=2.5,
        replicate_physics=True,
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events = EventCfg()

    commands = None
    rewards = None
    curriculum = None

    xr: XrCfg = XrCfg(
        anchor_pos=(0.0, 0.0, 0.0),
        anchor_rot=(0.70711, 0.0, 0.0, -0.70711),
    )

    temp_urdf_dir = tempfile.gettempdir()

    def __post_init__(self) -> None:
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = 2

        temp_urdf_output_path, temp_urdf_meshes_output_path = ControllerUtils.convert_usd_to_urdf(
            self.scene.robot.spawn.usd_path,
            self.temp_urdf_dir,
            force_conversion=True,
        )

        self.actions.pink_ik_cfg.controller.urdf_path = temp_urdf_output_path
        self.actions.pink_ik_cfg.controller.mesh_path = temp_urdf_meshes_output_path

        self.teleop_devices = DevicesCfg(
            devices={
                "handtracking": OpenXRDeviceCfg(
                    retargeters=[
                        UnitreeG1RetargeterCfg(
                            enable_visualization=True,
                            num_open_xr_hand_joints=2 * 26,
                            sim_device=self.sim.device,
                            hand_joint_names=self.actions.pink_ik_cfg.hand_joint_names,
                        )
                    ],
                    sim_device=self.sim.device,
                    xr_cfg=self.xr,
                ),
                "manusvive": ManusViveCfg(
                    retargeters=[
                        UnitreeG1RetargeterCfg(
                            enable_visualization=True,
                            num_open_xr_hand_joints=2 * 26,
                            sim_device=self.sim.device,
                            hand_joint_names=self.actions.pink_ik_cfg.hand_joint_names,
                        )
                    ],
                    sim_device=self.sim.device,
                    xr_cfg=self.xr,
                ),
            }
        )
