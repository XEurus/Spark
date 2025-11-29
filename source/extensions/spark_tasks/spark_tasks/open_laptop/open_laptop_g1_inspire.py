from __future__ import annotations
import tempfile
import torch
import carb

# 本文件定义了一个基于 ManagerBasedRLEnvCfg 的 "打开笔记本" 任务环境配置：
# - 机器人：IsaacLab 提供的 G1 Inspire（G1_INSPIRE_FTP_CFG）
# - 物体：EgoVLA/humanoid.tasks 中的 LAPTOP_CFG（笔记本）
# - 控制：Pink IK 控制 G1 上肢和双手手指

from pink.tasks import FrameTask

import isaaclab.controllers.utils as ControllerUtils
import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.controllers.pink_ik import NullSpacePostureTask, PinkIKControllerCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.openxr import ManusViveCfg, OpenXRDeviceCfg, XrCfg
from isaaclab.devices.openxr.retargeters.humanoid.unitree.inspire.g1_upper_body_retargeter import UnitreeG1RetargeterCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.pink_actions_cfg import PinkInverseKinematicsActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils import configclass

from . import mdp
from isaaclab_assets.robots.unitree import G1_INSPIRE_FTP_CFG  # isort: skip
# 这里使用的是 IsaacLab 里自带的 G1 Inspire 机器人配置
from humanoid.tasks.data.laptop import LAPTOP_CFG  # 这里复用 humanoid.tasks 中笔记本的 ArticulationCfg


@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """场景配置：定义本任务中有哪些物体，以及它们的初始状态。

    本场景包含：
    - 一个笔记本（laptop）：来自 LAPTOP_CFG，并放置到机器人前方桌面附近
    - 一个 G1 Inspire 机器人（robot）：站在原点附近
    - 一个地面平面（ground）
    - 一个环境光源（light）
    """

    # 笔记本：从 LAPTOP_CFG 复制一份，并修改在每个 env 下的 prim_path
    laptop: ArticulationCfg = LAPTOP_CFG.replace(
        prim_path="/World/envs/env_.*/Laptop",
    )
    # 设置笔记本在世界坐标中的初始位置和姿态
    laptop.init_state.pos = (0.6, 0.0, 1.03)
    laptop.init_state.rot = (0.5, 0.5, 0.5, 0.5)

    # 机器人：G1 Inspire，本任务中作为操作笔记本的主体
    robot: ArticulationCfg = G1_INSPIRE_FTP_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0, 0, 1.0),
            rot=(0.7071, 0, 0, 0.7071),
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

    # 地面平面：用于支撑机器人和物体
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=GroundPlaneCfg(),
    )

    # 环境光：增加场景整体照明
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class ActionsCfg:
    """动作配置：描述 RL 动作如何映射到 G1 Inspire 机器人。

    这里使用 Pink IK：
    - 控制上肢（肩、肘、腕）关节
    - 控制 24 个手指关节
    - 末端执行器是左右手腕的 yaw link
    """

    # Pink 逆运动学动作配置：描述被 IK 控制的关节以及 EEF 链接
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
        # 手指关节名列表：共 24 个关节（左右手各 12 个）
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
        # 指定左右手的末端执行器（EEF）链接名
        target_eef_link_names={
            "left_wrist": "left_wrist_yaw_link",
            "right_wrist": "right_wrist_yaw_link",
        },
        # Pink 控制的目标资产名称，对应 scene 中的 "robot"
        asset_name="robot",
        # Pink IK 控制器配置：描述 IK 解算的细节
        controller=PinkIKControllerCfg(
            articulation_name="robot",
            base_link_name="pelvis",
            num_hand_joints=24,
            show_ik_warnings=False,
            fail_on_joint_limit_violation=False,
            # variable_input_tasks：每一步根据 RL 动作更新的 IK 任务
            variable_input_tasks=[
                # 下面两条 FrameTask 的统一用法：
                # FrameTask(target_link_name, position_cost, orientation_cost, lm_damping, gain)
                # - target_link_name：需要控制的链接名（这里是左右手腕的 yaw link）
                # - position_cost：位置误差权重（越大越重视“到达目标位置”）
                # - orientation_cost：姿态误差权重（越大越重视“对齐目标朝向”）
                # - lm_damping：Levenberg-Marquardt 求解的阻尼项，越大越稳定但动作更“钝”
                # - gain：控制增益，决定朝着目标移动的速度
                # 左手腕的位姿任务：希望左手腕跟踪某个目标位置和朝向
                FrameTask(
                    "g1_29dof_rev_1_0_left_wrist_yaw_link",
                    position_cost=8.0,  # [cost] / [m]
                    orientation_cost=2.0,  # [cost] / [rad]
                    lm_damping=10,  # dampening for solver for step jumps
                    gain=0.5,
                ),
                FrameTask(
                    "g1_29dof_rev_1_0_right_wrist_yaw_link",
                    position_cost=8.0,  # [cost] / [m]
                    orientation_cost=2.0,  # [cost] / [rad]
                    lm_damping=10,  # dampening for solver for step jumps
                    gain=0.5,
                ),
                # NullSpacePostureTask(cost, lm_damping, controlled_frames, controlled_joints, gain)
                # - cost：偏离期望姿态时的“代价”权重
                # - lm_damping：该任务在 LM 求解中的阻尼
                # - controlled_frames：在哪些链接的空间里施加该 null-space 约束
                # - controlled_joints：要保持姿态的关节集合
                # - gain：收敛速度，越大越快回到期望姿态
                # 这里用于在 IK 的余空间里保持上身/肩部/腰部比较自然的姿态
                NullSpacePostureTask(
                    cost=0.5,
                    lm_damping=1,
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
            # fixed_input_tasks：本任务中没有固定输入任务
            fixed_input_tasks=[],
            # xr_enabled：根据应用设置决定是否启用 XR 支持
            xr_enabled=bool(carb.settings.get_settings().get("/app/xr/enabled")),
        ),
        enable_gravity_compensation=False,
    )


@configclass
class ObservationsCfg:
    """观测配置：定义策略可以看到哪些状态信息。"""

    @configclass
    class PolicyCfg(ObsGroup):
        """供策略使用的一组观测项。"""

        # 通用说明：
        # ObsTerm(func=..., params=...) 表示“如何计算一个观测项”，其中：
        # - func：一个接收 env 的函数（例如 base_mdp.joint_pos），返回一个张量
        # - params：额外参数，一般会传入 SceneEntityCfg("robot") / SceneEntityCfg("laptop") 等
        # ManagerBasedRLEnv 会在每个 step 调用这些 func 来构造观测字典。

        # 上一时间步的动作（有助于某些算法，如 RNN）
        actions = ObsTerm(func=base_mdp.last_action)

        # 机器人与笔记本的基础状态观测（只观测“状态本身”，不直接观测成功标志）
        robot_joint_pos = ObsTerm(
            func=base_mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )  # 机器人所有关节的位置
        robot_root_pos = ObsTerm(
            func=base_mdp.root_pos_w,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )  # 机器人根（基座）的世界坐标位置
        robot_root_rot = ObsTerm(
            func=base_mdp.root_quat_w,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )  # 机器人根（基座）的世界坐标旋转（四元数）

        # laptop_pos = ObsTerm(
        #     func=base_mdp.root_pos_w,
        #     params={"asset_cfg": SceneEntityCfg("laptop")},
        # )  # 笔记本根的世界坐标位置
        # laptop_rot = ObsTerm(
        #     func=base_mdp.root_quat_w,
        #     params={"asset_cfg": SceneEntityCfg("laptop")},
        # )  # 笔记本根的世界坐标旋转（四元数）
        # laptop_joint_pos = ObsTerm(
        #     func=base_mdp.joint_pos,
        #     params={"asset_cfg": SceneEntityCfg("laptop")},
        # )  # 笔记本所有关节的位置（包括盖子铰链角），供 termination / reward 等在 MDP 中使用

        # 末端执行器与手部关节相关观测
        left_eef_pos = ObsTerm(func=mdp.get_eef_pos, params={"link_name": "left_wrist_yaw_link"})
        left_eef_quat = ObsTerm(func=mdp.get_eef_quat, params={"link_name": "left_wrist_yaw_link"})
        right_eef_pos = ObsTerm(func=mdp.get_eef_pos, params={"link_name": "right_wrist_yaw_link"})
        right_eef_quat = ObsTerm(func=mdp.get_eef_quat, params={"link_name": "right_wrist_yaw_link"})
        hand_joint_state = ObsTerm(func=mdp.get_robot_joint_state, params={"joint_names": ["R_.*", "L_.*"]})

        # 物体相关观测：
        # - 物体在世界坐标系下的位置和姿态（四元数）
        # - 左/右手末端执行器（wrist_yaw_link）指向物体的相对位移向量
        # 这些信息用于帮助策略感知抓取/放置目标以及当前双手与物体之间的空间关系
        # object = ObsTerm(
        #     func=mdp.object_obs,
        #     params={
        #         "left_eef_link_name": "left_wrist_yaw_link",
        #         "right_eef_link_name": "right_wrist_yaw_link",
        #     },
        # )

        def __post_init__(self):
            # 不对观测添加噪声（corruption），保持干净信号
            self.enable_corruption = False
            # 不自动把所有 ObsTerm 拼成一个大向量，保持字典形式
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()  


@configclass
class TerminationsCfg:
    """终止条件配置：定义 episode 何时结束。"""
    # DoneTerm(func=..., params=...) 表示一个“是否 done”的判定项：
    # - func：一个接收 env 的函数（例如 mdp.time_out 或 mdp.task_done_open_laptop），返回 bool 张量
    # - params：传给该函数的额外参数，例如 SceneEntityCfg("laptop") 或时间步上限等

    # 基础的超时终止：当步数达到上限时结束 episode
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # 成功终止：当笔记本盖子打开足够大（超过关节上限的 70%）时结束 episode
    success = DoneTerm(
        func=mdp.task_done_open_laptop,
        params={
            "laptop_cfg": SceneEntityCfg("laptop"),
            "ratio": 0.7,
        },
    )


@configclass
class EventCfg:
    """事件配置：主要用于 reset 时重置场景和物体。"""

    # 将整个场景（机器人、物体等）重置为默认状态
    reset_all = EventTerm(func=base_mdp.reset_scene_to_default, mode="reset")

    # 在 reset 时，调用自定义的 reset_laptop_open：
    # - 将笔记本根状态放到 env 原点附近
    # - 按 EgoVLA 逻辑在 x/y 上做随机偏移或网格采样
    # - 重置关节位置到默认值并清零速度
    reset_laptop = EventTerm(
        func=mdp.reset_laptop_open,
        mode="reset",
        params={
            "laptop_cfg": SceneEntityCfg("laptop"),
            "randomize": True,
            "randomize_idx": -1,
            "randomize_range": 1.0,
        },
    )


@configclass
class Open_Laptop_G1_Inspire_EnvCfg(ManagerBasedRLEnvCfg):
    """打开笔记本任务的总环境配置。

    该类汇总了：
    - 场景配置（scene）
    - 动作配置（actions）
    - 观测配置（observations）
    - 终止条件（terminations）
    - 事件（events）
    并额外指定仿真参数、XR 配置以及 teleop 设备。
    """

    # 场景配置：1 个 env，env 之间间距 2.5 米，物理复用
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
    # 观测和动作管理器
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # 终止条件和事件管理器
    terminations: TerminationsCfg = TerminationsCfg()
    events = EventCfg()

    # 本任务暂不使用 commands / rewards / curriculum 管理器
    commands = None
    rewards = None
    curriculum = None

    # XR 锚点在世界坐标中的位置和姿态
    xr: XrCfg = XrCfg(
        anchor_pos=(0.0, 0.0, 0.0),
        anchor_rot=(1.0, 0.0, 0.0, 0.0),
    )

    # 用于存放从 USD 转为 URDF 后的临时文件目录
    temp_urdf_dir = tempfile.gettempdir()

    # 一个用于让机器人保持默认姿态的“空闲”动作
    idle_action = torch.tensor([
        -0.1487,
        0.2038,
        1.0952,
        0.707,
        0.0,
        0.0,
        0.707,
        0.1487,
        0.2038,
        1.0952,
        0.707,
        0.0,
        0.0,
        0.707,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ])

    def __post_init__(self):
        """在配置实例创建后自动调用，用来补充一些依赖场景的设置。"""
        # decimation：控制步长缩放（每控制一步仿真走多少小步）
        self.decimation = 6
        # 每个 episode 的最长时间（单位：秒）
        self.episode_length_s = 20.0
        # 仿真时间步长和渲染间隔
        self.sim.dt = 1 / 120
        self.sim.render_interval = 2

        # 将机器人 USD 模型转换为 URDF 和 mesh，以供 Pink IK 控制器使用
        temp_urdf_output_path, temp_urdf_meshes_output_path = ControllerUtils.convert_usd_to_urdf(
            self.scene.robot.spawn.usd_path, self.temp_urdf_dir, force_conversion=True
        )

        # 将生成的 URDF 和 mesh 路径写入 Pink IK 控制器
        self.actions.pink_ik_cfg.controller.urdf_path = temp_urdf_output_path
        self.actions.pink_ik_cfg.controller.mesh_path = temp_urdf_meshes_output_path

        # 远程操作设备配置：支持 OpenXR 手部追踪和 Manus 手套
        self.teleop_devices = DevicesCfg(
            devices={
                "handtracking": OpenXRDeviceCfg(
                    retargeters=[
                        UnitreeG1RetargeterCfg(
                            enable_visualization=True,
                            num_open_xr_hand_joints=2 * 26,
                            sim_device=self.sim.device,
                            hand_joint_names=self.actions.pink_ik_cfg.hand_joint_names,
                        ),
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
                        ),
                    ],
                    sim_device=self.sim.device,
                    xr_cfg=self.xr,
                ),
            },
        )