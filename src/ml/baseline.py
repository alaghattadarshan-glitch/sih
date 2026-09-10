"""Classical Inertial Navigation Baseline Calculation.

Computes physical 3D window displacement using classical numerical integration
(gravity compensation + trapezoidal velocity and position integration) over raw IMU feature windows.
Used as a physical baseline to compare against AI model drift prediction.
"""

import numpy as np
from typing import List, Tuple


def compute_classical_imu_displacement(
    window_data: np.ndarray,
    dt: float = 0.01,
    gravity: float = 9.81,
) -> np.ndarray:
    """Compute 3D displacement vector for a single IMU feature window via trapezoidal numerical integration.

    Args:
        window_data (np.ndarray): Single window feature matrix [L_window, D_features].
                                   Expected columns 0,1,2 = accel_x, accel_y, accel_z.
        dt (float): IMU sampling interval in seconds (default 0.01s = 100Hz).
        gravity (float): Gravity magnitude in m/s^2 (default 9.81).

    Returns:
        np.ndarray: Estimated 3D displacement vector [delta_p_x, delta_p_y, delta_p_z] in meters.
    """
    if window_data.ndim != 2 or window_data.shape[1] < 3:
        raise ValueError(f"Expected 2D window matrix [L, D>=3], got shape {window_data.shape}")

    # Extract body acceleration columns
    accel = window_data[:, :3].copy()  # [L, 3]

    # Gravity compensation on Z-axis (assumes roughly level body orientation over 1.0s window)
    accel[:, 2] -= gravity

    # 1. Trapezoidal Velocity Integration
    # v[k+1] = v[k] + 0.5 * (a[k] + a[k+1]) * dt
    velocity = np.zeros_like(accel)
    for i in range(1, len(accel)):
        velocity[i] = velocity[i - 1] + 0.5 * (accel[i - 1] + accel[i]) * dt

    # 2. Trapezoidal Position Integration
    # p[k+1] = p[k] + 0.5 * (v[k] + v[k+1]) * dt
    position = np.zeros_like(velocity)
    for i in range(1, len(velocity)):
        position[i] = position[i - 1] + 0.5 * (velocity[i - 1] + velocity[i]) * dt

    # Total displacement over window is final position minus initial position
    displacement = position[-1] - position[0]
    return displacement.astype(np.float32)


def compute_dataset_classical_baseline(
    windows_features_raw: np.ndarray,
    dt: float = 0.01,
    gravity: float = 9.81,
) -> np.ndarray:
    """Compute classical IMU displacement predictions for all windows in a dataset.

    Args:
        windows_features_raw (np.ndarray): Unnormalized features array [N_windows, L_window, D_features].
        dt (float): Sampling interval in seconds.
        gravity (float): Gravity magnitude.

    Returns:
        np.ndarray: Array of shape [N_windows, 3] containing classical displacement predictions.
    """
    displacements = []
    for w in windows_features_raw:
        disp = compute_classical_imu_displacement(w, dt=dt, gravity=gravity)
        displacements.append(disp)
    return np.array(displacements, dtype=np.float32)
