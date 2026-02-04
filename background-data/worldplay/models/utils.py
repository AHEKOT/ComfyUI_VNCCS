from typing import List
import numpy as np
import torch
import math
from einops import rearrange
from scipy.spatial.transform import Rotation as R_scipy
from ..utils.retrieval_context import calculate_fov_overlap_similarity

from typing import List, Dict, Tuple
# from distributed.parallel_state import get_sp_parallel_rank, get_sp_world_size
def get_sp_world_size(): return 1
def get_sp_parallel_rank(): return 0


def shard_latents_dim_across_sp(latents: torch.Tensor, total_dim) -> torch.Tensor:
    sp_world_size = get_sp_world_size()
    rank_in_sp_group = get_sp_parallel_rank()
    # latents = latents[:, :, :num_latent_t]
    if sp_world_size > 1:
        if total_dim == 5:
            latents = rearrange(
                latents, "b c (n s) h w -> b c n s h w", n=sp_world_size
            ).contiguous()
            latents = latents[:, :, rank_in_sp_group, :, :, :]
        elif total_dim == 4:
            latents = rearrange(
                latents, "b (n s) h w -> b n s h w", n=sp_world_size
            ).contiguous()
            latents = latents[:, rank_in_sp_group, :, :, :]
        elif total_dim == 3:
            latents = rearrange(
                latents, "b (n s) c -> b n s c", n=sp_world_size
            ).contiguous()
            latents = latents[:, rank_in_sp_group, :, :]
        elif total_dim == 2:
            latents = rearrange(
                latents, "b (n s)-> b n s", n=sp_world_size
            ).contiguous()
            latents = latents[:, rank_in_sp_group, :]
        elif total_dim == 1:
            latents = rearrange(latents, "(n s)-> n s", n=sp_world_size).contiguous()
            latents = latents[rank_in_sp_group, :]
        else:
            raise NotImplementedError(f"total_dim {total_dim} not supported")
    return latents


def select_mem_frames_wan(
    w2c_list: List[np.ndarray],
    current_frame_idx: int,
    memory_frames: int,
    temporal_context_size: int,
    pred_latent_size: int,
    pos_weight: float = 1.0,
    ang_weight: float = 1.0,
    device=None,
    points_local=None,
) -> List[int]:
    """
    为给定帧选择记忆帧和上下文帧，基于复杂的四帧片段距离计算。

    参数:
        w2c_list (List[np.ndarray]): 包含所有N个4x4外参矩阵的列表。
        current_frame_idx (int): 当前要处理的帧的索引。
        memory_frames (int): 需要选择的记忆帧总数。
        context_size (int): 需要选择的上下文帧总数。
        pos_weight (float): 空间距离的权重。
        ang_weight (float): 角度距离的权重。

    返回:
        List[int]: 包含选定记忆帧和上下文帧索引的列表。
    """
    num_total_frames = len(w2c_list)
    # 检查当前帧是否能构成一个完整的4帧片段
    if current_frame_idx >= num_total_frames or current_frame_idx < 3:
        raise ValueError("当前帧索引必须在 w2c_list 的有效范围内，且至少为3。")

    # 1. 选择上下文帧 (Context Frames)
    start_context_idx = max(0, current_frame_idx - temporal_context_size)
    context_frames_indices = list(range(start_context_idx, current_frame_idx))

    # 2. 计算记忆帧 (Memory Frames) 的候选池
    candidate_distances = []
    query_clip_indices = list(
        range(
            current_frame_idx,
            (
                current_frame_idx + pred_latent_size
                if current_frame_idx + pred_latent_size <= num_total_frames
                else num_total_frames
            ),
        )
    )

    historical_clip_indices = list(
        range(0, current_frame_idx - temporal_context_size, 4)
    )

    memory_frames_indices = []  # add the first latent frame as context
    memory_frames = memory_frames - temporal_context_size

    for hist_idx in historical_clip_indices:
        total_dist = 0
        hist_w2c_1 = w2c_list[hist_idx]
        hist_w2c_2 = w2c_list[hist_idx + 2]
        for query_idx in query_clip_indices:
            dist_1_for_query_idx = 1.0 - calculate_fov_overlap_similarity(
                w2c_list[query_idx],
                hist_w2c_1,
                fov_h_deg=60.0,
                fov_v_deg=35.0,
                device=device,
                points_local=points_local,
            )
            dist_2_for_query_idx = 1.0 - calculate_fov_overlap_similarity(
                w2c_list[query_idx],
                hist_w2c_2,
                fov_h_deg=60.0,
                fov_v_deg=35.0,
                device=device,
                points_local=points_local,
            )
            dist_for_query_idx = (dist_1_for_query_idx + dist_2_for_query_idx) / 2.0
            total_dist += dist_for_query_idx

        final_clip_distance = total_dist / len(query_clip_indices)
        candidate_distances.append((hist_idx, final_clip_distance))

    candidate_distances.sort(key=lambda x: x[1])

    # 遍历排序后的候选片段，直到收集到足够的记忆帧
    for start_idx, _ in candidate_distances:
        if start_idx not in memory_frames_indices:
            memory_frames_indices.extend(range(start_idx, start_idx + 4))

        # 检查是否已达到 memory_size 的要求
        if len(memory_frames_indices) >= memory_frames:
            break

    # 4. 组合并去重，以确保没有重复的帧
    selected_frames_set = set(context_frames_indices)
    selected_frames_set.update(memory_frames_indices)

    final_selected_frames = sorted(list(selected_frames_set))
    assert len(final_selected_frames) == memory_frames + temporal_context_size
    return final_selected_frames


# --- 核心函数重构 ---
# def select_memory_frames(w2c_list: List[np.ndarray], current_frame_idx: int, memory_size: int,
#                          context_size: int, pos_weight: float = 1.0, ang_weight: float = 1.0) -> List[int]:
#     """
#     为给定帧选择记忆帧和上下文帧，基于复杂的四帧片段距离计算。

#     参数:
#         w2c_list (List[np.ndarray]): 包含所有N个4x4外参矩阵的列表。
#         current_frame_idx (int): 当前要处理的帧的索引。
#         memory_size (int): 需要选择的记忆帧总数。
#         context_size (int): 需要选择的上下文帧总数。
#         pos_weight (float): 空间距离的权重。
#         ang_weight (float): 角度距离的权重。

#     返回:
#         List[int]: 包含选定记忆帧和上下文帧索引的列表。
#     """
#     num_total_frames = len(w2c_list)
#     # 检查当前帧是否能构成一个完整的4帧片段
#     if current_frame_idx >= num_total_frames or current_frame_idx < 3:
#         raise ValueError("当前帧索引必须在 w2c_list 的有效范围内，且至少为3。")

#     # 1. 选择上下文帧 (Context Frames)
#     start_context_idx = max(0, current_frame_idx - context_size)
#     context_frames_indices = list(range(start_context_idx, current_frame_idx))

#     # 2. 计算记忆帧 (Memory Frames) 的候选池
#     candidate_distances = []
#     query_clip_indices = list(range(current_frame_idx, current_frame_idx + 4))

#     # 遍历所有历史片段，将每个片段作为记忆帧候选
#     # 历史片段的起始索引必须是 4 的倍数，且不能与上下文帧重叠
#     for i in range(0, current_frame_idx, 4):
#         historical_clip_indices = list(range(i, i + 4))

#         # 排除上下文帧，如果历史片段与上下文帧有任何交集，就跳过
#         is_context_clip = any(idx in context_frames_indices for idx in historical_clip_indices)
#         if is_context_clip:
#             continue

#         avg_distance = calculate_complex_clip_distance(w2c_list, query_clip_indices, historical_clip_indices,
#                                                        pos_weight, ang_weight)

#         # 存储 (片段起始帧索引, 平均距离)
#         candidate_distances.append((i, avg_distance))

#     # 按平均距离从小到大排序
#     candidate_distances.sort(key=lambda x: x[1])

#     # 3. 选取最相似的 `memory_size` 个帧
#     memory_frames_indices = []

#     # 遍历排序后的候选片段，直到收集到足够的记忆帧
#     for start_idx, _ in candidate_distances:
#         # 将整个片段的帧都添加到记忆帧列表中
#         memory_frames_indices.extend(range(start_idx, start_idx + 4))

#         # 检查是否已达到 memory_size 的要求
#         if len(memory_frames_indices) >= memory_size:
#             break

#     # 4. 组合并去重，以确保没有重复的帧
#     selected_frames_set = set(context_frames_indices)
#     selected_frames_set.update(memory_frames_indices)

#     final_selected_frames = sorted(list(selected_frames_set))

#     return final_selected_frames


def calculate_complex_clip_distance(
    w2c_list: List[np.ndarray],
    query_clip_indices: List[int],
    historical_clip_indices: List[int],
    pos_weight: float = 1.0,
    ang_weight: float = 1.0,
) -> float:
    """
    计算查询片段与历史片段之间的复杂姿态距离。

    该距离是基于查询片段的第二帧和第四帧与历史片段的每一帧的平均距离。
    """
    if len(query_clip_indices) < 4 or len(historical_clip_indices) < 4:
        raise ValueError("片段索引列表必须包含4个元素。")

    # 提取查询片段的第二帧和第四帧的w2c矩阵
    query_2nd_w2c = w2c_list[query_clip_indices[1]]  # 第二帧
    query_4th_w2c = w2c_list[query_clip_indices[3]]  # 第四帧

    # 1. 计算查询片段第二帧与历史片段每帧的平均距离
    dists_from_2nd = []
    for hist_idx in historical_clip_indices:
        hist_w2c = w2c_list[hist_idx]
        dist = calculate_pose_distance_from_w2c(
            query_2nd_w2c, hist_w2c, pos_weight, ang_weight
        )
        dists_from_2nd.append(dist)
    avg_dist_from_2nd = np.mean(dists_from_2nd)

    # 2. 计算查询片段第四帧与历史片段每帧的平均距离
    dists_from_4th = []
    for hist_idx in historical_clip_indices:
        hist_w2c = w2c_list[hist_idx]
        dist = calculate_pose_distance_from_w2c(
            query_4th_w2c, hist_w2c, pos_weight, ang_weight
        )
        dists_from_4th.append(dist)
    avg_dist_from_4th = np.mean(dists_from_4th)

    # 3. 将两个平均值再次取平均，得到最终的片段距离
    final_clip_distance = (avg_dist_from_2nd + avg_dist_from_4th) / 2.0

    return final_clip_distance


def calculate_pose_distance_from_w2c(
    w2c_1: np.ndarray,
    w2c_2: np.ndarray,
    pos_weight: float = 1.0,
    ang_weight: float = 1.0,
) -> float:
    """
    根据两个 4x4 W2C (World-to-Camera) 矩阵计算它们之间的综合姿态距离。

    该距离量化了两个相机姿态的相似度，类似其 FOV 的重叠程度。

    参数:
        w2c_1 (np.ndarray): 第一个相机的 4x4 World-to-Camera 矩阵。
        w2c_2 (np.ndarray): 第二个相机的 4x4 World-to-Camera 矩阵。
        pos_weight (float): 空间距离的权重。
        ang_weight (float): 角度距离的权重。

    返回:
        float: 两个姿态之间的综合距离。
    """

    def w2c_to_6d_pose(w2c_matrix: np.ndarray) -> np.ndarray:
        """
        将 4x4 World-to-Camera (W2C) 矩阵转换为 6D 姿态。

        6D 姿态元组为 (x, y, z, pitch, yaw, roll)。
        """
        # 提取旋转矩阵 R 和平移向量 t
        R_cw = w2c_matrix[:3, :3]
        t_cw = w2c_matrix[:3, 3]

        # 计算相机在世界坐标系下的位置 C_world
        # C_world = -R_cw.T @ t_cw
        C_world = -np.dot(R_cw.T, t_cw)

        # 将旋转矩阵转换为欧拉角 (pitch, yaw, roll)
        # 注意: scipy 默认的欧拉角顺序是 ZYX，对应 yaw, pitch, roll
        # 为了与常见的 (pitch, yaw, roll) 顺序匹配，我们手动转换
        r = R_scipy.from_matrix(R_cw)
        pitch, yaw, roll = r.as_euler("yxz", degrees=True)

        return np.array([C_world[0], C_world[1], C_world[2], pitch, yaw, roll])

    # 1. 将两个 W2C 矩阵转换为 6D 姿态
    pose1_6d = w2c_to_6d_pose(w2c_1)
    pose2_6d = w2c_to_6d_pose(w2c_2)

    # 2. 计算空间距离 (欧几里得距离)
    pos1 = pose1_6d[:3]
    pos2 = pose2_6d[:3]
    spatial_distance = np.linalg.norm(pos1 - pos2)

    # 3. 计算角度距离 (考虑圆周特性)
    angles1 = pose1_6d[3:]
    angles2 = pose2_6d[3:]

    angle_diff = np.abs(angles1 - angles2)
    # 修正角度差，确保是最小的圆周距离
    angular_distance_vector = np.minimum(angle_diff, 360 - angle_diff)
    # 使用欧几里得范数作为综合角度距离
    angular_distance = np.linalg.norm(angular_distance_vector)

    # 4. 结合两种距离得到综合姿态距离
    total_distance = pos_weight * spatial_distance + ang_weight * angular_distance

    return total_distance
