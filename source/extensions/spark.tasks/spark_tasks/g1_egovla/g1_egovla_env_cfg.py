"""Skeleton for a manager-based G1 Inspire environment cfg.

Concrete scene, observations, terminations, rewards, and events
will be filled in later.
"""

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass


@configclass
class G1EgoVLAEnvCfg(ManagerBasedRLEnvCfg):
    """Empty env cfg placeholder.

    To be populated with:
    - scene (G1 robot + Ego benchmark objects)
    - observations
    - actions
    - terminations
    - rewards
    - events
    """

    # TODO: define scene, observations, actions, terminations, rewards, events
    pass
