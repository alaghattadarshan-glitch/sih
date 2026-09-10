"""Supervised Target Generation Module.

Generates ground-truth motion targets (displacement vector, final velocity, velocity change)
over corresponding window time intervals for model training and offline validation.

IMPORTANT:
- Ground-truth target generation is strictly for offline model training and evaluation.
- Targets must NEVER be passed as input features to real-time navigation components.
"""

import math
import numpy as np
from typing import List, Tuple, Dict, Any

from src.data.observations import GroundTruthObservation
from src.coordinate_transforms import LocalFrame


def build_window_targets(
    window_time_ranges: List[Tuple[float, float]],
    ground_truth_list: List[GroundTruthObservation],
    local_frame: LocalFrame,
    target_type: str = "displacement_enu",
) -> Tuple[np.ndarray, List[str]]:
    """Build supervised ground-truth targets corresponding to IMU window time ranges.

    Supported target types:
    - "displacement_enu": Position displacement Δp = p_gt(t_end) - p_gt(t_start) in ENU meters.
    - "velocity_end": True velocity vector v_gt(t_end) in ENU m/s.
    - "velocity_change": Velocity change Δv = v_gt(t_end) - v_gt(t_start) in ENU m/s.

    Args:
        window_time_ranges (List[Tuple[float, float]]): List of (t_start, t_end) timestamps per window.
        ground_truth_list (List[GroundTruthObservation]): Ground truth observations.
        local_frame (LocalFrame): Reference local ENU frame.
        target_type (str): Selected target category.

    Returns:
        Tuple[np.ndarray, List[str]]:
            - 2D numpy array of shape [N_windows, D_targets].
            - List of target feature column names.

    Raises:
        ValueError: If ground truth list is empty or target_type is unsupported.
    """
    if not ground_truth_list:
        raise ValueError("Cannot build targets from an empty ground truth list.")

    gt_times = np.array([gt.timestamp for gt in ground_truth_list], dtype=np.float64)
    gt_enu_positions = np.array(
        [local_frame.to_enu(gt.latitude, gt.longitude, gt.altitude) for gt in ground_truth_list],
        dtype=np.float64,
    )
    gt_velocities = np.array(
        [
            [
                gt.velocity_east if gt.velocity_east is not None else 0.0,
                gt.velocity_north if gt.velocity_north is not None else 0.0,
                gt.velocity_up if gt.velocity_up is not None else 0.0,
            ]
            for gt in ground_truth_list
        ],
        dtype=np.float64,
    )

    targets = []

    if target_type == "displacement_enu":
        target_names = ["delta_p_east", "delta_p_north", "delta_p_up"]
        for t_start, t_end in window_time_ranges:
            idx_start = int(np.argmin(np.abs(gt_times - t_start)))
            idx_end = int(np.argmin(np.abs(gt_times - t_end)))

            dp = gt_enu_positions[idx_end] - gt_enu_positions[idx_start]
            targets.append(dp)

    elif target_type == "velocity_end":
        target_names = ["v_end_east", "v_end_north", "v_end_up"]
        for _, t_end in window_time_ranges:
            idx_end = int(np.argmin(np.abs(gt_times - t_end)))
            targets.append(gt_velocities[idx_end])

    elif target_type == "velocity_change":
        target_names = ["delta_v_east", "delta_v_north", "delta_v_up"]
        for t_start, t_end in window_time_ranges:
            idx_start = int(np.argmin(np.abs(gt_times - t_start)))
            idx_end = int(np.argmin(np.abs(gt_times - t_end)))

            dv = gt_velocities[idx_end] - gt_velocities[idx_start]
            targets.append(dv)

    else:
        raise ValueError(f"Unsupported target_type: {target_type}")

    targets_array = np.array(targets, dtype=np.float32)
    return targets_array, target_names
