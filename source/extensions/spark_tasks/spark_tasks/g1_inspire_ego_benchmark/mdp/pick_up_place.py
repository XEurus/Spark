# Copyright (c) 2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations
import torch
from typing import TYPE_CHECKING
import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg
if TYPE_CHECKING: 
    from isaaclab.envs import ManagerBasedEnv # 仅在类型检查阶段导入 ManagerBasedRLEnv，避免运行时循环依赖或不必要的导入开销

def reset_laptop_open(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,  # 需要重置的环境 ID 列表（张量形式）
    laptop_cfg: SceneEntityCfg = SceneEntityCfg("laptop"),  # 指定笔记本在场景中的配置，默认名称为 "laptop"
    randomize: bool = True,  # 是否对笔记本在 x/y 方向的位置进行随机扰动
    randomize_idx: int = -1,  # 如果 >= 0，使用固定网格上的确定位置；< 0 时使用连续随机采样
    randomize_range: float = 1.0,  # 随机扰动的范围缩放系数
) -> None:
    """
    重置笔记本（laptop）的状态，与 EgoVLA 中 OpenLaptopEnv._reset_idx 的逻辑类似。

    主要功能：
    - 将笔记本的根状态（位置 + 姿态）放在对应 env 的原点附近；
    - 可选地对笔记本在 x/y 平面上的位置进行随机扰动（连续随机或规则网格）；
    - 将笔记本的关节位置重置为默认值（并裁剪到软关节限制范围内），关节速度重置为 0。
    """
    # 从场景中根据配置名称获取笔记本这个关节体（articulation）对象
    laptop = env.scene[laptop_cfg.name]

    # ===== 根姿态重置部分 =====
    # 从默认根状态中取出对应 env_ids 的根状态，并做一次拷贝，避免直接修改底层缓存
    laptop_root_state = laptop.data.default_root_state[env_ids].clone()
    # 将根位置平移到对应环境的原点位置处（每个 env 的原点可能不同）
    laptop_root_state[:, 0:3] += env.scene.env_origins[env_ids, :]

    # ===== 位置随机化部分（可选）=====
    if randomize:
        # randomize_idx < 0：使用连续的随机扰动
        if randomize_idx < 0:
            # 在 [-0.1, 0.1] * randomize_range 范围内随机生成 x/y 偏移
            # rand_xy 的形状为 [num_envs, 2]，分别对应 x 和 y 的偏移量
            rand_xy = (0.20 * randomize_range) * torch.rand((len(env_ids), 2), device=laptop.device) - (
                0.1 * randomize_range
            )
            # 将随机偏移加到根状态的 x/y 坐标上
            laptop_root_state[:, 0:2] += rand_xy
        else:
            # randomize_idx >= 0：使用 100x100 的规则网格，在 [-0.1, 0.1] 范围内取一个确定位置
            # 计算网格列索引（相当于 x 方向）
            column_idx = randomize_idx // 100
            # 计算网格行索引（相当于 y 方向）
            row_idx = randomize_idx % 100
            # 根据行列索引映射到 [-0.1, 0.1] 区间中的具体坐标（y 方向）
            laptop_root_state[:, 1] += 0.1 - (0.2 / 99) * row_idx
            # 根据行列索引映射到 [-0.1, 0.1] 区间中的具体坐标（x 方向）
            laptop_root_state[:, 0] += 0.1 - (0.2 / 99) * column_idx

    # 将更新后的根状态一次性写回到仿真中，应用到所有指定 env_ids 的笔记本
    laptop.write_root_state_to_sim(laptop_root_state, env_ids=env_ids)

    # ===== 关节状态重置部分 =====
    # 取出这些环境中笔记本的默认关节位置（例如：盖子的初始开合角度等）
    joint_pos = laptop.data.default_joint_pos[env_ids]
    # 将关节位置裁剪到软关节限制范围内，防止越界导致不稳定
    joint_pos = torch.clamp(
        joint_pos,
        laptop.data.soft_joint_pos_limits[0, :, 0],  # 每个关节的下限
        laptop.data.soft_joint_pos_limits[0, :, 1],  # 每个关节的上限
    )
    # 创建与 joint_pos 相同形状的零张量，作为初始关节速度（所有关节速度设为 0）
    joint_vel = torch.zeros_like(joint_pos)
    # 将关节位置和速度写入仿真，用于重置笔记本的关节状态
    laptop.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    print("reset laptop")

def reset_drawer_closed(env, env_ids, asset_cfg: SceneEntityCfg):
    drawer = env.scene[asset_cfg.name]

    bottom_joint_idx = drawer.find_joints(["bottom_joint"])[0][0]
    top_joint_idx = drawer.find_joints(["top_joint"])[0][0]

    # joint limits
    joint_limits = drawer.data.joint_limits
    bottom_close = joint_limits[env_ids, bottom_joint_idx, 1]
    top_close = joint_limits[env_ids, top_joint_idx, 1]

    # joint states
    joint_pos = drawer.data.joint_pos.clone()
    joint_vel = drawer.data.joint_vel.clone()

    joint_pos[env_ids, bottom_joint_idx] = bottom_close
    joint_pos[env_ids, top_joint_idx] = top_close

    drawer.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    return {}

def reset_object_poses_nut_pour(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    sorting_beaker_cfg: SceneEntityCfg = SceneEntityCfg("sorting_beaker"),
    factory_nut_cfg: SceneEntityCfg = SceneEntityCfg("factory_nut"),
    sorting_bowl_cfg: SceneEntityCfg = SceneEntityCfg("sorting_bowl"),
    sorting_scale_cfg: SceneEntityCfg = SceneEntityCfg("sorting_scale"),
):
    """Reset the asset root states to a random position and orientation uniformly within the given ranges.

    Args:
        env: The RL environment instance.
        env_ids: The environment IDs to reset the object poses for.
        sorting_beaker_cfg: The configuration for the sorting beaker asset.
        factory_nut_cfg: The configuration for the factory nut asset.
        sorting_bowl_cfg: The configuration for the sorting bowl asset.
        sorting_scale_cfg: The configuration for the sorting scale asset.
        pose_range: The dictionary of pose ranges for the objects. Keys are
                    ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``.
    """
    # extract the used quantities (to enable type-hinting)
    sorting_beaker = env.scene[sorting_beaker_cfg.name]
    factory_nut = env.scene[factory_nut_cfg.name]
    sorting_bowl = env.scene[sorting_bowl_cfg.name]
    sorting_scale = env.scene[sorting_scale_cfg.name]

    # get default root state
    sorting_beaker_root_states = sorting_beaker.data.default_root_state[env_ids].clone()
    factory_nut_root_states = factory_nut.data.default_root_state[env_ids].clone()
    sorting_bowl_root_states = sorting_bowl.data.default_root_state[env_ids].clone()
    sorting_scale_root_states = sorting_scale.data.default_root_state[env_ids].clone()

    # get pose ranges
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=sorting_beaker.device)

    # randomize sorting beaker and factory nut together
    rand_samples = math_utils.sample_uniform(
        ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=sorting_beaker.device
    )
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    positions_sorting_beaker = (
        sorting_beaker_root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
    )
    positions_factory_nut = factory_nut_root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
    orientations_sorting_beaker = math_utils.quat_mul(sorting_beaker_root_states[:, 3:7], orientations_delta)
    orientations_factory_nut = math_utils.quat_mul(factory_nut_root_states[:, 3:7], orientations_delta)

    # randomize sorting bowl
    rand_samples = math_utils.sample_uniform(
        ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=sorting_beaker.device
    )
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    positions_sorting_bowl = sorting_bowl_root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
    orientations_sorting_bowl = math_utils.quat_mul(sorting_bowl_root_states[:, 3:7], orientations_delta)

    # randomize scorting scale
    rand_samples = math_utils.sample_uniform(
        ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=sorting_beaker.device
    )
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    positions_sorting_scale = sorting_scale_root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
    orientations_sorting_scale = math_utils.quat_mul(sorting_scale_root_states[:, 3:7], orientations_delta)

    # set into the physics simulation
    sorting_beaker.write_root_pose_to_sim(
        torch.cat([positions_sorting_beaker, orientations_sorting_beaker], dim=-1), env_ids=env_ids
    )
    factory_nut.write_root_pose_to_sim(
        torch.cat([positions_factory_nut, orientations_factory_nut], dim=-1), env_ids=env_ids
    )
    sorting_bowl.write_root_pose_to_sim(
        torch.cat([positions_sorting_bowl, orientations_sorting_bowl], dim=-1), env_ids=env_ids
    )
    sorting_scale.write_root_pose_to_sim(
        torch.cat([positions_sorting_scale, orientations_sorting_scale], dim=-1), env_ids=env_ids
    )