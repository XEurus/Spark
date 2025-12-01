"""
Humanoid heat food environment.
"""

import gymnasium as gym

from .open_laptop_g1_inspire import Open_Laptop_G1_Inspire_EnvCfg

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="m2g-g1-close-drawer-inspire-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.m2g_g1_close_drawer_cfg:DrawerG1InspireFTPEnvCfg",
        "robomimic_bc_cfg_entry_point": f"{agents.__name__}:robomimic/bc_rnn_low_dim.json",
    },
    disable_env_checker=True,
)

gym.register(
    id="m2g-g1-open-drawer-inspire-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.m2g_g1_open_drawer_cfg:DrawerG1InspireFTPEnvCfg",
        "robomimic_bc_cfg_entry_point": f"{agents.__name__}:robomimic/bc_rnn_low_dim.json",
    },
    disable_env_checker=True,
)


# gym.register(
#     id="Isaac-PickPlace-GR1T2-Abs-v0",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.pickplace_gr1t2_env_cfg:PickPlaceGR1T2EnvCfg",
#         "robomimic_bc_cfg_entry_point": f"{agents.__name__}:robomimic/bc_rnn_low_dim.json",
#     },
#     disable_env_checker=True,
# )

# gym.register(
#     id="Isaac-NutPour-GR1T2-Pink-IK-Abs-v0",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.nutpour_gr1t2_pink_ik_env_cfg:NutPourGR1T2PinkIKEnvCfg",
#         "robomimic_bc_cfg_entry_point": f"{agents.__name__}:robomimic/bc_rnn_image_nut_pouring.json",
#     },
#     disable_env_checker=True,
# )

# gym.register(
#     id="Isaac-ExhaustPipe-GR1T2-Pink-IK-Abs-v0",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.exhaustpipe_gr1t2_pink_ik_env_cfg:ExhaustPipeGR1T2PinkIKEnvCfg",
#         "robomimic_bc_cfg_entry_point": f"{agents.__name__}:robomimic/bc_rnn_image_exhaust_pipe.json",
#     },
#     disable_env_checker=True,
# )

# gym.register(
#     id="Isaac-PickPlace-GR1T2-WaistEnabled-Abs-v0",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.pickplace_gr1t2_waist_enabled_env_cfg:PickPlaceGR1T2WaistEnabledEnvCfg",
#         "robomimic_bc_cfg_entry_point": f"{agents.__name__}:robomimic/bc_rnn_low_dim.json",
#     },
#     disable_env_checker=True,
# )

# gym.register(
#     id="Isaac-PickPlace-G1-InspireFTP-Abs-v0",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.pickplace_unitree_g1_inspire_hand_env_cfg:PickPlaceG1InspireFTPEnvCfg",
#         "robomimic_bc_cfg_entry_point": f"{agents.__name__}:robomimic/bc_rnn_low_dim.json",
#     },
#     disable_env_checker=True,
# )

gym.register(
    id="Spark-Open-Laptop-G1Inspire-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": Open_Laptop_G1_Inspire_EnvCfg},
)