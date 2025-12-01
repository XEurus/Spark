import tempfile
import torch

import carb
from pink.tasks import FrameTask

import isaaclab.controllers.utils as ControllerUtils
import isaaclab.envs.mdp as base_mdp
from isaaclab.sensors.camera.camera_cfg import CameraCfg
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
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
from isaaclab.sim.schemas.schemas_cfg import MassPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

#from . import mdp

from isaaclab_assets.robots.unitree import G1_INSPIRE_FTP_CFG  # isort: skip

from humanoid.tasks.data.drawer.drawer import DRAWER_CFG
from humanoid.tasks.data.table.table import TABLE_CFG
from humanoid.tasks.data.scene.room import ROOM_CFG


# EGO_VLA_SCALE = (0.7,0.7,0.7)
# # EGO_VLA_FACE_DIRECTION_POS = (0, 0.55, 0)
# # EGO_VLA_FACE_DIRECTION_ROT = (0, 0, 0, 1)

##
# Scene definition
##
@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
        # # table
    # table = TABLE_CFG.replace(
    # prim_path="/World/envs/env_.*/Table")
    # table.spawn.scale = (0.7,0.7,0.7)
    # table.init_state = AssetBaseCfg.InitialStateCfg(pos=(0.6,0,0), rot=(0.70711,0,0,0.70711))
    # table: spawn as a static USD asset (no RigidBody required)
    table = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Table",
        spawn=UsdFileCfg(
            usd_path=TABLE_CFG.spawn.usd_path,
            activate_contact_sensors=False,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            scale=(0.7, 0.7, 0.7),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.6, 0, 0),
            rot=(0.70711, 0, 0, 0.70711),
        ),
    )

    # background 我这边打不开这个似乎 还在看什么问题 暂时注释了，，
    # room = ROOM_CFG.replace(prim_path="/World/envs/env_.*/Room")  # type: ignore
    # room.spawn.scale = (0.7,0.7,0.7)

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=GroundPlaneCfg(),
    )

    # Lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    main_camera = CameraCfg(
        prim_path="/World/envs/env_.*/fixed_camera",
        height=216,
        width=384,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            projection_type="pinhole",
            focal_length=4,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.05, 0.0, 1.27),
            rot=(0.65328, 0.2706, -0.2706, -0.65328),
            convention="opengl",
        ),
    )