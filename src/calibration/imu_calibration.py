"""IMU Sensor Static Calibration Module.

Estimates static accelerometer and gyroscope sensor biases from a stationary IMU log segment.

Assumptions:
1. The IMU is completely stationary during the provided calibration segment.
2. Gyroscope ideal output when stationary is 0.0 rad/s on all axes.
3. Accelerometer ideal output when stationary and level is +1g (9.80665 m/s^2) along Z-axis.
"""

from typing import List, Tuple
import numpy as np

from src.data.observations import IMUObservation


def estimate_static_bias(
    stationary_observations: List[IMUObservation],
    expected_gravity: float = 9.80665,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate static accelerometer and gyroscope biases from a stationary dataset.

    Args:
        stationary_observations (List[IMUObservation]): List of IMU observations collected while stationary.
        expected_gravity (float): Standard local gravitational acceleration in m/s^2.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Estimated (accel_bias, gyro_bias) vectors in m/s^2 and rad/s.

    Raises:
        ValueError: If stationary_observations list is empty.
    """
    if not stationary_observations:
        raise ValueError("Cannot estimate static bias from an empty observation list.")

    accel_data = np.array(
        [
            [obs.accelerometer_x, obs.accelerometer_y, obs.accelerometer_z]
            for obs in stationary_observations
        ],
        dtype=np.float64,
    )
    gyro_data = np.array(
        [
            [obs.gyroscope_x, obs.gyroscope_y, obs.gyroscope_z]
            for obs in stationary_observations
        ],
        dtype=np.float64,
    )

    # Gyroscope bias: Mean of stationary angular rates (ideal rate = 0 rad/s)
    gyro_bias = np.mean(gyro_data, axis=0)

    # Accelerometer bias: Mean of stationary acceleration minus expected gravity vector [0, 0, +g]
    mean_accel = np.mean(accel_data, axis=0)
    expected_stationary_accel = np.array([0.0, 0.0, expected_gravity], dtype=np.float64)
    accel_bias = mean_accel - expected_stationary_accel

    return accel_bias, gyro_bias


def apply_imu_calibration(
    obs: IMUObservation, accel_bias: np.ndarray, gyro_bias: np.ndarray
) -> IMUObservation:
    """Apply bias corrections to an IMU observation.

    Args:
        obs (IMUObservation): Raw input IMU observation.
        accel_bias (np.ndarray): 3-element accelerometer bias vector in m/s^2.
        gyro_bias (np.ndarray): 3-element gyroscope bias vector in rad/s.

    Returns:
        IMUObservation: Calibrated IMU observation instance.
    """
    b_a = np.asarray(accel_bias, dtype=np.float64)
    b_g = np.asarray(gyro_bias, dtype=np.float64)

    return IMUObservation(
        timestamp=obs.timestamp,
        accelerometer_x=obs.accelerometer_x - b_a[0],
        accelerometer_y=obs.accelerometer_y - b_a[1],
        accelerometer_z=obs.accelerometer_z - b_a[2],
        gyroscope_x=obs.gyroscope_x - b_g[0],
        gyroscope_y=obs.gyroscope_y - b_g[1],
        gyroscope_z=obs.gyroscope_z - b_g[2],
        magnetometer_x=obs.magnetometer_x,
        magnetometer_y=obs.magnetometer_y,
        magnetometer_z=obs.magnetometer_z,
    )
