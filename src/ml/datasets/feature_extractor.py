"""High-Rate IMU Window Generation & Feature Extraction.

Generates temporal sliding windows from raw IMU observation sequences and computes
derived physical features (acceleration magnitude, angular rate magnitude, relative time).
"""

import math
import numpy as np
from typing import List, Tuple, Optional

from src.data.observations import IMUObservation

DEFAULT_FEATURE_COLUMNS = [
    "accel_x",
    "accel_y",
    "accel_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "accel_norm",
    "gyro_norm",
    "rel_time",
]


def extract_imu_windows(
    imu_list: List[IMUObservation],
    window_size_samples: int = 100,
    stride_samples: int = 50,
    feature_columns: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[Tuple[float, float]], List[str]]:
    """Extract sliding windows of high-rate IMU features.

    Args:
        imu_list (List[IMUObservation]): Input sequence of IMU observations.
        window_size_samples (int): Number of samples per window (e.g. 100 = 1.0s at 100Hz).
        stride_samples (int): Window stride in samples (e.g. 50 = 0.5s stride).
        feature_columns (Optional[List[str]]): List of feature names to include.

    Returns:
        Tuple[np.ndarray, List[Tuple[float, float]], List[str]]:
            - 3D numpy array of shape [N_windows, window_size_samples, D_features].
            - List of (t_start, t_end) timestamps for each window.
            - List of feature column names.

    Raises:
        ValueError: If imu_list is empty, contains NaNs, or is shorter than window_size_samples.
    """
    if not imu_list:
        raise ValueError("Cannot extract windows from an empty IMU observation list.")

    if len(imu_list) < window_size_samples:
        raise ValueError(
            f"IMU observation list length ({len(imu_list)}) is shorter than window size ({window_size_samples})."
        )

    cols = feature_columns if feature_columns is not None else DEFAULT_FEATURE_COLUMNS

    # Build raw feature matrix [N_total, D_all]
    n_total = len(imu_list)
    raw_matrix = []
    t_start_base = imu_list[0].timestamp

    for i, obs in enumerate(imu_list):
        ax, ay, az = obs.accelerometer_x, obs.accelerometer_y, obs.accelerometer_z
        gx, gy, gz = obs.gyroscope_x, obs.gyroscope_y, obs.gyroscope_z

        if not (
            math.isfinite(ax)
            and math.isfinite(ay)
            and math.isfinite(az)
            and math.isfinite(gx)
            and math.isfinite(gy)
            and math.isfinite(gz)
        ):
            raise ValueError(f"Non-finite value detected in IMU observation at index {i}.")

        a_norm = math.sqrt(ax**2 + ay**2 + az**2)
        g_norm = math.sqrt(gx**2 + gy**2 + gz**2)
        rel_time = obs.timestamp - t_start_base

        row_dict = {
            "accel_x": ax,
            "accel_y": ay,
            "accel_z": az,
            "gyro_x": gx,
            "gyro_y": gy,
            "gyro_z": gz,
            "accel_norm": a_norm,
            "gyro_norm": g_norm,
            "rel_time": rel_time,
            "mag_x": obs.magnetometer_x if obs.magnetometer_x is not None else 0.0,
            "mag_y": obs.magnetometer_y if obs.magnetometer_y is not None else 0.0,
            "mag_z": obs.magnetometer_z if obs.magnetometer_z is not None else 0.0,
        }
        raw_matrix.append([row_dict[c] for c in cols])

    full_array = np.array(raw_matrix, dtype=np.float32)

    # Slice sliding windows
    windows = []
    time_ranges = []

    start_idx = 0
    while start_idx + window_size_samples <= n_total:
        end_idx = start_idx + window_size_samples
        window_data = full_array[start_idx:end_idx].copy()

        # Zero-center relative time per window
        if "rel_time" in cols:
            r_idx = cols.index("rel_time")
            window_data[:, r_idx] -= window_data[0, r_idx]

        t_start = imu_list[start_idx].timestamp
        t_end = imu_list[end_idx - 1].timestamp

        windows.append(window_data)
        time_ranges.append((t_start, t_end))

        start_idx += stride_samples

    windows_array = np.stack(windows, axis=0)  # [N_windows, L_window, D_features]

    return windows_array, time_ranges, cols
