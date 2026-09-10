"""Sensor Timestamp Quality Statistics & Multi-Sensor Time Synchronization.

Aligns high-rate IMU observations (100Hz+) with lower-rate GNSS observations (1Hz)
using nearest-neighbor matching within a specified timestamp tolerance threshold.
"""

import math
from typing import List, Tuple, Dict, Optional, Any
import numpy as np

from src.data.observations import IMUObservation, GNSSObservation, GroundTruthObservation


def estimate_sampling_rate(timestamps: List[float]) -> float:
    """Estimate sampling frequency in Hz from a timestamp sequence.

    Args:
        timestamps (List[float]): Sequence of timestamps in seconds.

    Returns:
        float: Estimated sampling frequency in Hz (or 0.0 if empty/single sample).
    """
    if len(timestamps) < 2:
        return 0.0
    dts = [t2 - t1 for t1, t2 in zip(timestamps[:-1], timestamps[1:]) if t2 > t1]
    if not dts:
        return 0.0
    median_dt = float(np.median(dts))
    return 1.0 / median_dt if median_dt > 0 else 0.0


def find_timestamp_gaps(
    timestamps: List[float], max_expected_dt: float
) -> List[Tuple[float, float, float]]:
    """Identify gaps in timestamps where dt exceeds max_expected_dt.

    Args:
        timestamps (List[float]): Sequence of timestamps in seconds.
        max_expected_dt (float): Threshold dt above which a gap is declared.

    Returns:
        List[Tuple[float, float, float]]: List of (start_time, end_time, gap_duration_sec).
    """
    gaps: List[Tuple[float, float, float]] = []
    for t1, t2 in zip(timestamps[:-1], timestamps[1:]):
        dt = t2 - t1
        if dt > max_expected_dt:
            gaps.append((t1, t2, dt))
    return gaps


def calculate_timestamp_statistics(timestamps: List[float]) -> Dict[str, Any]:
    """Compute comprehensive timestamp quality statistics.

    Args:
        timestamps (List[float]): Sequence of timestamps in seconds.

    Returns:
        Dict[str, Any]: Dictionary containing count, duration, sampling rate, jitter, gaps, duplicates.
    """
    if not timestamps:
        return {
            "count": 0,
            "duration_sec": 0.0,
            "sampling_rate_hz": 0.0,
            "mean_dt": 0.0,
            "std_dt": 0.0,
            "max_gap_sec": 0.0,
            "duplicate_count": 0,
        }

    duration = timestamps[-1] - timestamps[0]
    dts = [t2 - t1 for t1, t2 in zip(timestamps[:-1], timestamps[1:])]
    
    duplicates = sum(1 for dt in dts if dt == 0)
    valid_dts = [dt for dt in dts if dt > 0]

    mean_dt = float(np.mean(valid_dts)) if valid_dts else 0.0
    std_dt = float(np.std(valid_dts)) if valid_dts else 0.0
    max_gap = float(np.max(valid_dts)) if valid_dts else 0.0
    freq_hz = estimate_sampling_rate(timestamps)

    return {
        "count": len(timestamps),
        "duration_sec": float(duration),
        "sampling_rate_hz": float(freq_hz),
        "mean_dt": float(mean_dt),
        "std_dt": float(std_dt),
        "max_gap_sec": float(max_gap),
        "duplicate_count": duplicates,
    }


def synchronize_imu_gnss(
    imu_list: List[IMUObservation],
    gnss_list: List[GNSSObservation],
    max_tolerance_sec: float = 0.5,
) -> List[Tuple[int, Optional[int], float]]:
    """Align high-rate IMU observations with lower-rate GNSS observations.

    Uses nearest-neighbor timestamp matching within max_tolerance_sec.

    Args:
        imu_list (List[IMUObservation]): List of IMU observations.
        gnss_list (List[GNSSObservation]): List of GNSS observations.
        max_tolerance_sec (float): Maximum allowed time difference in seconds to match a fix.

    Returns:
        List[Tuple[int, Optional[int], float]]: List of (imu_index, matched_gnss_index, time_diff_sec).
    """
    if not imu_list:
        return []

    if not gnss_list:
        return [(i, None, float("inf")) for i in range(len(imu_list))]

    gnss_times = np.array([g.timestamp for g in gnss_list])
    sync_results: List[Tuple[int, Optional[int], float]] = []

    for imu_idx, imu_obs in enumerate(imu_list):
        t_imu = imu_obs.timestamp
        # Find nearest GNSS index
        diffs = np.abs(gnss_times - t_imu)
        min_idx = int(np.argmin(diffs))
        min_diff = float(diffs[min_idx])

        if min_diff <= max_tolerance_sec:
            sync_results.append((imu_idx, min_idx, min_diff))
        else:
            sync_results.append((imu_idx, None, min_diff))

    return sync_results
