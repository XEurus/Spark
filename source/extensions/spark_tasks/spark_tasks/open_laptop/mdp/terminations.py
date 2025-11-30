from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def task_done_open_laptop(
    env: "ManagerBasedRLEnv",
    laptop_cfg: SceneEntityCfg = SceneEntityCfg("laptop"),
    ratio: float = 0.7,
) -> torch.Tensor:
    """
    判断笔记本是否被“充分打开”
    - 使用第一个关节（盖子）的上限 laptop_upper。如果当前角度超过上限的 70%（开得足够大），就认为任务成功。
    - 将对应观测的 success 标记为 1。
    """
    laptop = env.scene[laptop_cfg.name]
    laptop_upper = laptop.data.joint_limits[:, 0, 1]
    done = laptop.data.joint_pos[:, 0] > laptop_upper * ratio
    return done

# 似乎控制最终速度和位置是合理的，但是原benchmark中没有设置，所以不设置
# def laptop_velocity_below_threshold(
#     env: "ManagerBasedRLEnv",
#     laptop_cfg: SceneEntityCfg = SceneEntityCfg("laptop"),
#     hinge_vel_threshold: float = 0.05,
#     root_vel_threshold: float = 0.05,
# ) -> torch.Tensor:
#     """
#     同时约束笔记本转轴角速度和整体线速度不超过各自阈值。

#     返回:
#         done: bool 张量，True 表示该环境中转轴角速度和根线速度均低于阈值。
#     """
#     laptop = env.scene[laptop_cfg.name]

#     # 根线速度约束（最终/整体速度）
#     root_vel = laptop.data.root_vel_w  # [..., 3(+)]
#     lin_speed = torch.norm(root_vel[:, :3], dim=-1)
#     root_ok = lin_speed <= root_vel_threshold

#     # 转轴（假定第 0 号关节为笔记本转轴/盖子关节）的角速度约束
#     hinge_vel = torch.abs(laptop.data.joint_vel[:, 0])
#     hinge_ok = hinge_vel <= hinge_vel_threshold

#     done = torch.logical_and(root_ok, hinge_ok)
#     return done

# def _laptop_not_moved_too_far(env: "ManagerBasedRLEnv", laptop_cfg: SceneEntityCfg, max_disp: float = 0.1) -> torch.Tensor:
#         laptop = env.scene[laptop_cfg.name]
#         laptop_root_pos = laptop.data.root_pos_w
#         env_origins = env.scene.env_origins
#         laptop_disp = torch.norm(laptop_root_pos - env_origins, dim=-1)
#         done = laptop_disp <= max_disp
#         return done

# 子任务
def task_done_move_lid_open_laptop(
    env: "ManagerBasedRLEnv",
    laptop_cfg: SceneEntityCfg = SceneEntityCfg("laptop"),
    ratio: float = 0.15,
) -> torch.Tensor:
    """
    判断笔记本盖子是否“开始被打开”
    - 取出第一个关节（盖子）在关节上限处的值 laptop_upper,如果当前关节角度超过上限的 15%，则认为盖子已经开始被打开。
    - 对应观测的 move_lid_success 置为 1。
    """
    laptop = env.scene[laptop_cfg.name]
    laptop_upper = laptop.data.joint_limits[:, 0, 1]
    done = laptop.data.joint_pos[:, 0] > laptop_upper * ratio
    return done