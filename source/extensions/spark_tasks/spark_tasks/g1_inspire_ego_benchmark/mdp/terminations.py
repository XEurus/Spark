from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import isaaclab.utils.math as math_utils

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

def task_done_close_drawer(
    env,
    drawer_cfg: SceneEntityCfg = SceneEntityCfg("drawer"),
    drawer_bottom_joint_id: int = 0,
    close_ratio: float = 0.90,
    vel_threshold: float = 0.05,
):
    """MPD-style success condition for CLOSE DRAWER task."""

    drawer: Articulation = env.scene[drawer_cfg.name]

    # joint position
    joint_pos = drawer.data.joint_pos[:, drawer_bottom_joint_id]
    joint_upper = drawer.data.joint_pos_limits[:, drawer_bottom_joint_id, 1]

    # drawer close?
    close_mask = joint_pos > (joint_upper * close_ratio)

    # joint velocity
    joint_vel = torch.abs(drawer.data.joint_vel[:, drawer_bottom_joint_id])
    vel_mask = joint_vel < vel_threshold

    done = torch.logical_and(close_mask, vel_mask)
    return done

def task_done_close_drawer_hand_away(
    env,
    drawer_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    drawer_bottom_joint_id: int = 0,
    close_ratio: float = 0.90,
    vel_threshold: float = 0.05,
    hand_away_threshold: float = 0.18,
) -> torch.Tensor:
    """Determine if the drawer-closing task is complete.

    Success when:
    1. Drawer bottom joint has reached close_ratio * upper_limit.
    2. Drawer joint velocity is small enough.
    3. BOTH hands are at least hand_away_threshold away from the handle.
    """

    # Get drawer articulation
    drawer = env.scene[drawer_cfg.name]

    # -----------------------------
    # Drawer condition 1: joint closed
    # -----------------------------
    joint_pos = drawer.data.joint_pos[:, drawer_bottom_joint_id]
    joint_upper = drawer.data.joint_pos_limits[:, drawer_bottom_joint_id, 1]

    close_mask = joint_pos > (joint_upper * close_ratio)

    # -----------------------------
    # Drawer condition 2: joint velocity small
    # -----------------------------
    joint_vel = torch.abs(drawer.data.joint_vel[:, drawer_bottom_joint_id])
    vel_mask = joint_vel < vel_threshold

    # -----------------------------
    # Hand condition: both hands far away
    # -----------------------------
    # Handle world position = drawer root + fixed offset
    handle_offset = torch.tensor([-0.31, 0.30, -0.05], device=env.device)
    handle_pos_w = drawer.data.root_pos_w + handle_offset  # [n_env, 3]

    # Robot links
    robot = env.scene["robot"]
    robot_body_pos_w = robot.data.body_pos_w
    body_names = robot.data.body_names

    # Find indices of wrists
    left_idx = body_names.index("left_wrist_yaw_link")
    right_idx = body_names.index("right_wrist_yaw_link")

    # EEF positions
    left_pos = robot_body_pos_w[:, left_idx, :]
    right_pos = robot_body_pos_w[:, right_idx, :]

    # Distances to handle
    left_dist = torch.norm(left_pos - handle_pos_w, dim=-1)
    right_dist = torch.norm(right_pos - handle_pos_w, dim=-1)

    left_far = left_dist > hand_away_threshold
    right_far = right_dist > hand_away_threshold
    both_hands_far = torch.logical_and(left_far, right_far)

    # -----------------------------
    # Combine all success conditions
    # -----------------------------
    done = close_mask
    done = torch.logical_and(done, vel_mask)
    done = torch.logical_and(done, both_hands_far)

    return done

def task_done_open_drawer_hand_away(
    env,
    drawer_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    drawer_bottom_joint_id: int = 0,
    open_ratio: float = 0.90,
    vel_threshold: float = 0.05,
    hand_away_threshold: float = 0.18,
) -> torch.Tensor:
    """
    Determine if the drawer-opening task is complete.

    Success when:
    1. Drawer bottom joint has reached: lower_limit + open_ratio * (upper_limit - lower_limit)
       (Meaning: sufficiently OPEN)
    2. Drawer joint velocity is small.
    3. BOTH hands are at least hand_away_threshold away from the handle.
    """

    drawer = env.scene[drawer_cfg.name]

    # -----------------------------
    # Drawer condition 1: sufficiently OPEN
    # -----------------------------
    joint_pos = drawer.data.joint_pos[:, drawer_bottom_joint_id]
    joint_limits = drawer.data.joint_pos_limits[:, drawer_bottom_joint_id]

    lower = joint_limits[:, 0]   # fully OPEN
    upper = joint_limits[:, 1]   # fully CLOSED

    # open threshold = lower + ratio * (upper - lower)
    open_threshold = lower + open_ratio * (upper - lower)

    # A drawer is open when joint_pos is LOWER or equal to threshold
    open_mask = joint_pos < open_threshold

    # -----------------------------
    # Drawer condition 2: small velocity
    # -----------------------------
    joint_vel = torch.abs(drawer.data.joint_vel[:, drawer_bottom_joint_id])
    vel_mask = joint_vel < vel_threshold

    # -----------------------------
    # Hand condition: both hands far away
    # -----------------------------
    handle_offset = torch.tensor([-0.31, 0.30, -0.05], device=env.device)
    handle_pos_w = drawer.data.root_pos_w + handle_offset

    # robot
    robot = env.scene["robot"]
    pos_w = robot.data.body_pos_w
    names = robot.data.body_names

    left_idx = names.index("left_wrist_yaw_link")
    right_idx = names.index("right_wrist_yaw_link")

    left_pos = pos_w[:, left_idx]
    right_pos = pos_w[:, right_idx]

    left_far = torch.norm(left_pos - handle_pos_w, dim=-1) > hand_away_threshold
    right_far = torch.norm(right_pos - handle_pos_w, dim=-1) > hand_away_threshold

    both_far = torch.logical_and(left_far, right_far)

    # -----------------------------
    # Combine all conditions
    # -----------------------------
    done = torch.logical_and(open_mask, vel_mask)
    done = torch.logical_and(done, both_far)

    return done

def task_done_pick_place(
    env: ManagerBasedRLEnv,
    task_link_name: str = "",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    right_wrist_max_x: float = 0.26,
    min_x: float = 0.40,
    max_x: float = 0.85,
    min_y: float = 0.35,
    max_y: float = 0.60,
    max_height: float = 1.10,
    min_vel: float = 0.20,
) -> torch.Tensor:
    """Determine if the object placement task is complete.

    This function checks whether all success conditions for the task have been met:
    1. object is within the target x/y range
    2. object is below a minimum height
    3. object velocity is below threshold
    4. Right robot wrist is retracted back towards body (past a given x pos threshold)

    Args:
        env: The RL environment instance.
        object_cfg: Configuration for the object entity.
        right_wrist_max_x: Maximum x position of the right wrist for task completion.
        min_x: Minimum x position of the object for task completion.
        max_x: Maximum x position of the object for task completion.
        min_y: Minimum y position of the object for task completion.
        max_y: Maximum y position of the object for task completion.
        max_height: Maximum height (z position) of the object for task completion.
        min_vel: Minimum velocity magnitude of the object for task completion.

    Returns:
        Boolean tensor indicating which environments have completed the task.
    """
    if task_link_name == "":
        raise ValueError("task_link_name must be provided to task_done_pick_place")

    # Get object entity from the scene
    object: RigidObject = env.scene[object_cfg.name]

    # Extract wheel position relative to environment origin
    object_x = object.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    object_y = object.data.root_pos_w[:, 1] - env.scene.env_origins[:, 1]
    object_height = object.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    object_vel = torch.abs(object.data.root_vel_w)

    # Get right wrist position relative to environment origin
    robot_body_pos_w = env.scene["robot"].data.body_pos_w
    right_eef_idx = env.scene["robot"].data.body_names.index(task_link_name)
    right_wrist_x = robot_body_pos_w[:, right_eef_idx, 0] - env.scene.env_origins[:, 0]

    # Check all success conditions and combine with logical AND
    done = object_x < max_x
    done = torch.logical_and(done, object_x > min_x)
    done = torch.logical_and(done, object_y < max_y)
    done = torch.logical_and(done, object_y > min_y)
    done = torch.logical_and(done, object_height < max_height)
    done = torch.logical_and(done, right_wrist_x < right_wrist_max_x)
    done = torch.logical_and(done, object_vel[:, 0] < min_vel)
    done = torch.logical_and(done, object_vel[:, 1] < min_vel)
    done = torch.logical_and(done, object_vel[:, 2] < min_vel)

    return done

def task_done_nut_pour(
    env: ManagerBasedRLEnv,
    sorting_scale_cfg: SceneEntityCfg = SceneEntityCfg("sorting_scale"),
    sorting_bowl_cfg: SceneEntityCfg = SceneEntityCfg("sorting_bowl"),
    sorting_beaker_cfg: SceneEntityCfg = SceneEntityCfg("sorting_beaker"),
    factory_nut_cfg: SceneEntityCfg = SceneEntityCfg("factory_nut"),
    sorting_bin_cfg: SceneEntityCfg = SceneEntityCfg("black_sorting_bin"),
    max_bowl_to_scale_x: float = 0.055,
    max_bowl_to_scale_y: float = 0.055,
    max_bowl_to_scale_z: float = 0.025,
    max_nut_to_bowl_x: float = 0.050,
    max_nut_to_bowl_y: float = 0.050,
    max_nut_to_bowl_z: float = 0.019,
    max_beaker_to_bin_x: float = 0.08,
    max_beaker_to_bin_y: float = 0.12,
    max_beaker_to_bin_z: float = 0.07,
) -> torch.Tensor:
    """Determine if the nut pouring task is complete.

    This function checks whether all success conditions for the task have been met:
    1. The factory nut is in the sorting bowl
    2. The sorting beaker is in the sorting bin
    3. The sorting bowl is placed on the sorting scale

    Args:
        env: The RL environment instance.
        sorting_scale_cfg: Configuration for the sorting scale entity.
        sorting_bowl_cfg: Configuration for the sorting bowl entity.
        sorting_beaker_cfg: Configuration for the sorting beaker entity.
        factory_nut_cfg: Configuration for the factory nut entity.
        sorting_bin_cfg: Configuration for the sorting bin entity.
        max_bowl_to_scale_x: Maximum x position of the sorting bowl relative to the sorting scale for task completion.
        max_bowl_to_scale_y: Maximum y position of the sorting bowl relative to the sorting scale for task completion.
        max_bowl_to_scale_z: Maximum z position of the sorting bowl relative to the sorting scale for task completion.
        max_nut_to_bowl_x: Maximum x position of the factory nut relative to the sorting bowl for task completion.
        max_nut_to_bowl_y: Maximum y position of the factory nut relative to the sorting bowl for task completion.
        max_nut_to_bowl_z: Maximum z position of the factory nut relative to the sorting bowl for task completion.
        max_beaker_to_bin_x: Maximum x position of the sorting beaker relative to the sorting bin for task completion.
        max_beaker_to_bin_y: Maximum y position of the sorting beaker relative to the sorting bin for task completion.
        max_beaker_to_bin_z: Maximum z position of the sorting beaker relative to the sorting bin for task completion.

    Returns:
        Boolean tensor indicating which environments have completed the task.
    """
    # Get object entities from the scene
    sorting_scale: RigidObject = env.scene[sorting_scale_cfg.name]
    sorting_bowl: RigidObject = env.scene[sorting_bowl_cfg.name]
    factory_nut: RigidObject = env.scene[factory_nut_cfg.name]
    sorting_beaker: RigidObject = env.scene[sorting_beaker_cfg.name]
    sorting_bin: RigidObject = env.scene[sorting_bin_cfg.name]

    # Get positions relative to environment origin
    scale_pos = sorting_scale.data.root_pos_w - env.scene.env_origins
    bowl_pos = sorting_bowl.data.root_pos_w - env.scene.env_origins
    sorting_beaker_pos = sorting_beaker.data.root_pos_w - env.scene.env_origins
    nut_pos = factory_nut.data.root_pos_w - env.scene.env_origins
    bin_pos = sorting_bin.data.root_pos_w - env.scene.env_origins

    # nut to bowl
    nut_to_bowl_x = torch.abs(nut_pos[:, 0] - bowl_pos[:, 0])
    nut_to_bowl_y = torch.abs(nut_pos[:, 1] - bowl_pos[:, 1])
    nut_to_bowl_z = nut_pos[:, 2] - bowl_pos[:, 2]

    # bowl to scale
    bowl_to_scale_x = torch.abs(bowl_pos[:, 0] - scale_pos[:, 0])
    bowl_to_scale_y = torch.abs(bowl_pos[:, 1] - scale_pos[:, 1])
    bowl_to_scale_z = bowl_pos[:, 2] - scale_pos[:, 2]

    # beaker to bin
    beaker_to_bin_x = torch.abs(sorting_beaker_pos[:, 0] - bin_pos[:, 0])
    beaker_to_bin_y = torch.abs(sorting_beaker_pos[:, 1] - bin_pos[:, 1])
    beaker_to_bin_z = sorting_beaker_pos[:, 2] - bin_pos[:, 2]

    done = nut_to_bowl_x < max_nut_to_bowl_x
    done = torch.logical_and(done, nut_to_bowl_y < max_nut_to_bowl_y)
    done = torch.logical_and(done, nut_to_bowl_z < max_nut_to_bowl_z)
    done = torch.logical_and(done, bowl_to_scale_x < max_bowl_to_scale_x)
    done = torch.logical_and(done, bowl_to_scale_y < max_bowl_to_scale_y)
    done = torch.logical_and(done, bowl_to_scale_z < max_bowl_to_scale_z)
    done = torch.logical_and(done, beaker_to_bin_x < max_beaker_to_bin_x)
    done = torch.logical_and(done, beaker_to_bin_y < max_beaker_to_bin_y)
    done = torch.logical_and(done, beaker_to_bin_z < max_beaker_to_bin_z)

    return done

def task_done_exhaust_pipe(
    env: ManagerBasedRLEnv,
    blue_exhaust_pipe_cfg: SceneEntityCfg = SceneEntityCfg("blue_exhaust_pipe"),
    blue_sorting_bin_cfg: SceneEntityCfg = SceneEntityCfg("blue_sorting_bin"),
    max_blue_exhaust_to_bin_x: float = 0.085,
    max_blue_exhaust_to_bin_y: float = 0.200,
    min_blue_exhaust_to_bin_y: float = -0.090,
    max_blue_exhaust_to_bin_z: float = 0.070,
) -> torch.Tensor:
    """Determine if the exhaust pipe task is complete.

    This function checks whether all success conditions for the task have been met:
    1. The blue exhaust pipe is placed in the correct position

    Args:
        env: The RL environment instance.
        blue_exhaust_pipe_cfg: Configuration for the blue exhaust pipe entity.
        blue_sorting_bin_cfg: Configuration for the blue sorting bin entity.
        max_blue_exhaust_to_bin_x: Maximum x position of the blue exhaust pipe relative to the blue sorting bin for task completion.
        max_blue_exhaust_to_bin_y: Maximum y position of the blue exhaust pipe relative to the blue sorting bin for task completion.
        max_blue_exhaust_to_bin_z: Maximum z position of the blue exhaust pipe relative to the blue sorting bin for task completion.

    Returns:
        Boolean tensor indicating which environments have completed the task.
    """
    # Get object entities from the scene
    blue_exhaust_pipe: RigidObject = env.scene[blue_exhaust_pipe_cfg.name]
    blue_sorting_bin: RigidObject = env.scene[blue_sorting_bin_cfg.name]

    # Get positions relative to environment origin
    blue_exhaust_pipe_pos = blue_exhaust_pipe.data.root_pos_w - env.scene.env_origins
    blue_sorting_bin_pos = blue_sorting_bin.data.root_pos_w - env.scene.env_origins

    # blue exhaust to bin
    blue_exhaust_to_bin_x = torch.abs(blue_exhaust_pipe_pos[:, 0] - blue_sorting_bin_pos[:, 0])
    blue_exhaust_to_bin_y = blue_exhaust_pipe_pos[:, 1] - blue_sorting_bin_pos[:, 1]
    blue_exhaust_to_bin_z = blue_exhaust_pipe_pos[:, 2] - blue_sorting_bin_pos[:, 2]

    done = blue_exhaust_to_bin_x < max_blue_exhaust_to_bin_x
    done = torch.logical_and(done, blue_exhaust_to_bin_y < max_blue_exhaust_to_bin_y)
    done = torch.logical_and(done, blue_exhaust_to_bin_y > min_blue_exhaust_to_bin_y)
    done = torch.logical_and(done, blue_exhaust_to_bin_z < max_blue_exhaust_to_bin_z)

def task_done_unload_cans(
    env,
    can1_cfg: SceneEntityCfg = SceneEntityCfg("can_fanta_1"),
    can2_cfg: SceneEntityCfg = SceneEntityCfg("can_fanta_2"),
    container_cfg: SceneEntityCfg = SceneEntityCfg("container"),
    require_upright_and_height: bool = False,
    up_z_threshold: float = 0.5,
    min_height: float = 1.0,
    container_center_xy=(0.48, 0.0),
    container_half_extent_xy=(0.16, 0.10),
) -> torch.Tensor:
    """
    - First compute the container’s bounding box in the x–y plane using its env-local position.
    - The base footprint is defined by a nominal center ``[0.48, 0.0]`` and half extents ``[0.16, 0.10]``, which are then shifted by the container’s current position (via an inferred XY offset).
    - The episode is considered successful if and only if both cans lie outside this bounding box.
    """

    can1 = env.scene[can1_cfg.name]
    can2 = env.scene[can2_cfg.name]
    container = env.scene[container_cfg.name]

    # positions in env-local frame
    can1_pos = can1.data.root_pos_w - env.scene.env_origins
    can2_pos = can2.data.root_pos_w - env.scene.env_origins
    container_pos = container.data.root_pos_w - env.scene.env_origins

    device = container_pos.device

    # Base container footprint (in env frame) and its nominal center.
    container_center_ref = torch.tensor(container_center_xy, device=device)
    # half-width/half-height of the container footprint in XY
    container_half_extent = torch.tensor(container_half_extent_xy, device=device)
    container_xy_lower_ref = container_center_ref - container_half_extent
    container_xy_upper_ref = container_center_ref + container_half_extent

    # Infer per-env XY offset from current container position.
    offset_xy = container_pos[:, 0:2] - container_center_ref 
    # 随机偏移量offset_xy 通过容器位置传导。现在没有随机偏移，但是保存这一逻辑
    lower_xy = container_xy_lower_ref + offset_xy
    upper_xy = container_xy_upper_ref + offset_xy

    # Check whether cans are inside the container XY box.
    can1_in_x = torch.logical_and(can1_pos[:, 0] > lower_xy[:, 0], can1_pos[:, 0] < upper_xy[:, 0])
    can1_in_y = torch.logical_and(can1_pos[:, 1] > lower_xy[:, 1], can1_pos[:, 1] < upper_xy[:, 1])
    can2_in_x = torch.logical_and(can2_pos[:, 0] > lower_xy[:, 0], can2_pos[:, 0] < upper_xy[:, 0])
    can2_in_y = torch.logical_and(can2_pos[:, 1] > lower_xy[:, 1], can2_pos[:, 1] < upper_xy[:, 1])

    can1_in_container = torch.logical_and(can1_in_x, can1_in_y)
    can2_in_container = torch.logical_and(can2_in_x, can2_in_y)

    any_in_container = torch.logical_or(can1_in_container, can2_in_container)
    done = torch.logical_not(any_in_container) # 要没有一个罐子在容器内 not any_in_container

    if require_upright_and_height:
        # height in env-local frame
        can1_z = can1_pos[:, 2]
        can2_z = can2_pos[:, 2]

        # z-axis of each can in world frame
        z_axis = torch.zeros_like(can1_pos)
        z_axis[:, 2] = 1.0
        z_can1 = math_utils.quat_apply(can1.data.root_quat_w, z_axis)
        z_can2 = math_utils.quat_apply(can2.data.root_quat_w, z_axis)

        can1_up = z_can1[:, 2] > up_z_threshold
        can2_up = z_can2[:, 2] > up_z_threshold
        can1_high = can1_z > min_height
        can2_high = can2_z > min_height

        upright1 = torch.logical_and(can1_up, can1_high)
        upright2 = torch.logical_and(can2_up, can2_high)
        both_upright = torch.logical_and(upright1, upright2)

        done = torch.logical_and(done, both_upright)

    return done
# 子任务应该是还有一个卸载成功（要保持罐子竖起来）和拿起成功、成功到达。
# 目前ego内不要求罐子竖立，但是与ego代码保持一致，保留这一逻辑。

def task_done_sort_cans(
    env: "ManagerBasedRLEnv",
    can_sprite1_cfg: SceneEntityCfg = SceneEntityCfg("can_sprite_1"),
    can_sprite2_cfg: SceneEntityCfg = SceneEntityCfg("can_sprite_2"),
    can_fanta1_cfg: SceneEntityCfg = SceneEntityCfg("can_fanta_1"),
    can_fanta2_cfg: SceneEntityCfg = SceneEntityCfg("can_fanta_2"),
    container_left_center=(0.65, 0.085, 0.76),
    container_right_center=(0.65, -0.085, 0.76),
    up_half_extent=(0.12, 0.075, 0.05),
    down_half_extent=(0.13, 0.075, 0.08),
) -> torch.Tensor:
    """
    - Two red cans (Fanta) must lie *inside* the right container box.
    - Two orange cans (Sprite) must lie *inside* the left container box.
    - Coordinates are evaluated in env-local frame (subtract env origins).
    """
    # positions in env-local coordinates
    can_sprite_1 = env.scene[can_sprite1_cfg.name]
    can_sprite_2 = env.scene[can_sprite2_cfg.name]
    can_fanta_1 = env.scene[can_fanta1_cfg.name]
    can_fanta_2 = env.scene[can_fanta2_cfg.name]

    # positions in env-local coordinates
    sprite1_pos = can_sprite_1.data.root_pos_w - env.scene.env_origins
    sprite2_pos = can_sprite_2.data.root_pos_w - env.scene.env_origins
    fanta1_pos = can_fanta_1.data.root_pos_w - env.scene.env_origins
    fanta2_pos = can_fanta_2.data.root_pos_w - env.scene.env_origins

    device = sprite1_pos.device

    # Container centers in env-local frame (from inputs)
    # container_1: left container (y > 0), container_2: right container (y < 0)
    container_1 = torch.tensor(container_left_center, device=device)
    container_2 = torch.tensor(container_right_center, device=device)

    # Half extents of the container boxes
    up = torch.tensor(up_half_extent, device=device)  # 这边可能需要测试后调整
    down = torch.tensor(down_half_extent, device=device)

    right_center = container_2
    left_center = container_1

    # Container boxes (top-left / bottom-right) expressed as center +/- half extents
    right_top_left = right_center + up
    right_bot_right = right_center - down
    left_top_left = left_center + up
    left_bot_right = left_center - down

    def in_box(pos: torch.Tensor, bot_right: torch.Tensor, top_left: torch.Tensor) -> torch.Tensor:
        inside = torch.logical_and(pos > bot_right, pos < top_left)
        return inside.all(dim=-1)

    # Fanta cans should be in the right container box
    fanta1_in_right = in_box(fanta1_pos, right_bot_right, right_top_left)
    fanta2_in_right = in_box(fanta2_pos, right_bot_right, right_top_left)
    fanta_sorted = torch.logical_and(fanta1_in_right, fanta2_in_right)

    # Sprite cans should be in the left container box
    sprite1_in_left = in_box(sprite1_pos, left_bot_right, left_top_left)
    sprite2_in_left = in_box(sprite2_pos, left_bot_right, left_top_left)
    sprite_sorted = torch.logical_and(sprite1_in_left, sprite2_in_left)

    done = torch.logical_and(fanta_sorted, sprite_sorted)

    return done

def task_done_stack_can(
    env: "ManagerBasedRLEnv",
    can_cfg: SceneEntityCfg = SceneEntityCfg("can"),
    plate_cfg: SceneEntityCfg = SceneEntityCfg("plate"),
    plate_radius: float = 0.045, # 和柜子等比缩放了
    vertical_tolerance: float = 0.02,
) -> torch.Tensor:
    """
    - Compute horizontal (XY) distance between can and plate centers.
    - Compute vertical (Z) distance between can and plate.
    - Task is successful when horizontal distance is within ``plate_radius``
      and vertical distance is below ``vertical_tolerance``.
    """

    can = env.scene[can_cfg.name]
    plate = env.scene[plate_cfg.name]

    # positions relative to environment origin
    can_pos = can.data.root_pos_w - env.scene.env_origins
    plate_pos = plate.data.root_pos_w - env.scene.env_origins

    diff = can_pos - plate_pos
    dist_horizontal = torch.norm(diff[:, :2], dim=-1)
    dist_vertical = torch.abs(diff[:, 2])

    done = dist_horizontal < plate_radius
    done = torch.logical_and(done, dist_vertical < vertical_tolerance)

    return done

def task_done_stack_can_into_drawer(
    env: "ManagerBasedRLEnv",
    can_cfg: SceneEntityCfg = SceneEntityCfg("can"),
    plate_cfg: SceneEntityCfg = SceneEntityCfg("plate"),
    drawer_cfg: SceneEntityCfg = SceneEntityCfg("drawer"),
    drawer_bottom_joint_id: int = 1,
    close_ratio: float = 0.90,
    plate_radius: float = 0.045, # 和柜子等比缩放了
    vertical_tolerance: float = 0.02,
) -> torch.Tensor:
    """Success for stack_can_into_drawer task.

    Mirrors Ego's :meth:`StackCanIntoDrawerEnv._get_success` and
    ``_get_joints_data``:

    - Can must be horizontally within ``plate_radius`` of the plate.
    - Vertical distance between can and plate must be below
      ``vertical_tolerance``.
    - Drawer bottom joint must be sufficiently closed (>= close_ratio of
      its upper joint limit).
    """

    can = env.scene[can_cfg.name]
    plate = env.scene[plate_cfg.name]
    drawer = env.scene[drawer_cfg.name]

    # ---- can on plate (env-local) ----
    can_pos = can.data.root_pos_w - env.scene.env_origins
    plate_pos = plate.data.root_pos_w - env.scene.env_origins

    diff = can_pos - plate_pos
    dist_horizontal = torch.norm(diff[:, :2], dim=-1)
    dist_vertical = torch.abs(diff[:, 2])

    stacked = dist_horizontal < plate_radius
    stacked = torch.logical_and(stacked, dist_vertical < vertical_tolerance)

    # ---- drawer closed condition ----
    joint_pos = drawer.data.joint_pos[:, drawer_bottom_joint_id]
    joint_upper = drawer.data.joint_pos_limits[:, drawer_bottom_joint_id, 1]
    drawer_closed = joint_pos > (joint_upper * close_ratio)

    done = torch.logical_and(stacked, drawer_closed)

    return done