from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def get_eef_pos(env: "ManagerBasedRLEnv", link_name: str) -> torch.Tensor:
    """Return EEF position (x, y, z) in environment frame for the given robot link name."""
    body_pos_w = env.scene["robot"].data.body_pos_w
    link_idx = env.scene["robot"].data.body_names.index(link_name)
    eef_pos = body_pos_w[:, link_idx] - env.scene.env_origins
    return eef_pos


def get_eef_quat(env: "ManagerBasedRLEnv", link_name: str) -> torch.Tensor:
    """Return EEF orientation (w, x, y, z) in world frame for the given robot link name."""
    body_quat_w = env.scene["robot"].data.body_quat_w
    link_idx = env.scene["robot"].data.body_names.index(link_name)
    eef_quat = body_quat_w[:, link_idx]
    return eef_quat


def get_robot_joint_state(env: "ManagerBasedRLEnv", joint_names: list[str]) -> torch.Tensor:
    """Return joint positions for all joints that match the provided regex patterns."""
    indexes, _ = env.scene["robot"].find_joints(joint_names)
    indexes_tensor = torch.tensor(indexes, dtype=torch.long, device=env.device)
    joint_states = env.scene["robot"].data.joint_pos[:, indexes_tensor]
    return joint_states
