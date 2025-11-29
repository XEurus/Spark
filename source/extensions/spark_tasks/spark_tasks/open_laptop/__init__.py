"""
Humanoid heat food environment.
"""

import gymnasium as gym

from .open_laptop_g1_inspire import Open_Laptop_G1_Inspire_EnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Spark-Open-Laptop-G1Inspire-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": Open_Laptop_G1_Inspire_EnvCfg},
)
