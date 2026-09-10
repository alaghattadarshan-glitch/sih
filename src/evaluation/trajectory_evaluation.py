"""Navigation Trajectory Evaluation Engine.

Computes quantitative error metrics (East, North, Up, Horizontal RMSE, Max Horizontal Error,
Final Position Error, Velocity RMS, Heading RMS) by comparing estimated navigation states
against reference ground-truth observations.
"""

import math
from typing import List, Dict, Any
import numpy as np

from src.navigation.state import NavigationState
from src.data.observations import GroundTruthObservation
from src.coordinate_transforms import LocalFrame


def evaluate_trajectory(
    estimated_states: List[NavigationState],
    ground_truth_list: List[GroundTruthObservation],
    local_frame: LocalFrame,
) -> Dict[str, Any]:
    """Evaluate estimated navigation states against reference ground truth trajectory.

    Args:
        estimated_states (List[NavigationState]): Sequence of estimated navigation states.
        ground_truth_list (List[GroundTruthObservation]): Sequence of ground-truth observations.
        local_frame (LocalFrame): Reference local ENU frame.

    Returns:
        Dict[str, Any]: Evaluation summary dictionary containing time arrays, position/velocity/heading
                        error arrays, and aggregate error metrics (RMSE, Max Error, Final Error).
    """
    if not estimated_states or not ground_truth_list:
        return {"status": "FAILED", "reason": "Empty inputs"}

    # Build ground truth timestamp to ENU position lookup array
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
    gt_headings = np.array(
        [gt.heading if gt.heading is not None else 0.0 for gt in ground_truth_list],
        dtype=np.float64,
    )

    timestamps = []
    east_errors = []
    north_errors = []
    up_errors = []
    horiz_errors = []
    pos_3d_errors = []
    vel_errors = []
    heading_errors = []

    for est in estimated_states:
        t = est.timestamp
        # Find matching ground truth index by time
        idx = int(np.argmin(np.abs(gt_times - t)))
        if abs(gt_times[idx] - t) > 0.5:
            continue  # Skip if no close ground truth match

        gt_pos = gt_enu_positions[idx]
        est_pos = est.position_enu

        e_err = est_pos[0] - gt_pos[0]
        n_err = est_pos[1] - gt_pos[1]
        u_err = est_pos[2] - gt_pos[2]
        h_err = math.sqrt(e_err**2 + n_err**2)
        err_3d = math.sqrt(e_err**2 + n_err**2 + u_err**2)

        timestamps.append(t)
        east_errors.append(e_err)
        north_errors.append(n_err)
        up_errors.append(u_err)
        horiz_errors.append(h_err)
        pos_3d_errors.append(err_3d)

        # Velocity error
        v_err = np.linalg.norm(est.velocity_enu - gt_velocities[idx])
        vel_errors.append(v_err)

        # Heading error (-180 to +180 deg wrapping)
        h_diff = (est.heading_deg() - gt_headings[idx] + 180.0) % 360.0 - 180.0
        heading_errors.append(abs(h_diff))

    if not timestamps:
        return {"status": "FAILED", "reason": "No matched timestamps"}

    times_arr = np.array(timestamps)
    e_arr = np.array(east_errors)
    n_arr = np.array(north_errors)
    u_arr = np.array(up_errors)
    h_arr = np.array(horiz_errors)
    v_arr = np.array(vel_errors)
    hdg_arr = np.array(heading_errors)

    horiz_rmse = float(np.sqrt(np.mean(h_arr**2)))
    max_horiz_err = float(np.max(h_arr))
    final_pos_err = float(h_arr[-1])
    vel_rmse = float(np.sqrt(np.mean(v_arr**2)))
    final_vel_err = float(v_arr[-1])
    hdg_rmse = float(np.sqrt(np.mean(hdg_arr**2)))
    final_hdg_err = float(hdg_arr[-1])

    return {
        "status": "PASS",
        "timestamps": times_arr,
        "east_errors": e_arr,
        "north_errors": n_arr,
        "up_errors": u_arr,
        "horizontal_errors": h_arr,
        "velocity_errors": v_arr,
        "heading_errors": hdg_arr,
        "horizontal_rmse": horiz_rmse,
        "max_horizontal_error": max_horiz_err,
        "final_position_error": final_pos_err,
        "velocity_rmse": vel_rmse,
        "final_velocity_error": final_vel_err,
        "heading_rmse": hdg_rmse,
        "final_heading_error": final_hdg_err,
    }
