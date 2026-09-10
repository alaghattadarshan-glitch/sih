"""IMU Signal Preprocessing and Low-Pass Filtering.

Applies digital low-pass filtering (Butterworth) to raw accelerometer and gyroscope
data sequences to suppress high-frequency noise while preserving true vehicle motion.
"""

import numpy as np
from scipy.signal import butter, filtfilt


def filter_signal_1d(data: np.ndarray, fs: float, cutoff_hz: float = 10.0, order: int = 2) -> np.ndarray:
    """Apply zero-phase Butterworth low-pass filter to a 1D signal.

    Args:
        data (np.ndarray): 1D signal array.
        fs (float): Sampling frequency in Hz.
        cutoff_hz (float): Low-pass cutoff frequency in Hz.
        order (int): Filter order.

    Returns:
        np.ndarray: Filtered 1D signal array.
    """
    if len(data) <= 3 * order:
        return data  # Return unfiltered if data sequence is too short for filtfilt

    nyquist = 0.5 * fs
    normal_cutoff = min(cutoff_hz / nyquist, 0.99)
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return filtfilt(b, a, data)


def filter_accelerometer(
    accel_data: np.ndarray, fs: float, cutoff_hz: float = 10.0, order: int = 2
) -> np.ndarray:
    """Filter 3-axis accelerometer data array [N x 3].

    Args:
        accel_data (np.ndarray): N x 3 array of accelerometer readings [ax, ay, az].
        fs (float): Sampling frequency in Hz.
        cutoff_hz (float): Cutoff frequency in Hz (default 10.0Hz).
        order (int): Filter order (default 2).

    Returns:
        np.ndarray: N x 3 filtered accelerometer array.
    """
    accel = np.asarray(accel_data, dtype=np.float64)
    filtered = np.zeros_like(accel)
    for col in range(3):
        filtered[:, col] = filter_signal_1d(accel[:, col], fs, cutoff_hz, order)
    return filtered


def filter_gyroscope(
    gyro_data: np.ndarray, fs: float, cutoff_hz: float = 10.0, order: int = 2
) -> np.ndarray:
    """Filter 3-axis gyroscope data array [N x 3].

    Args:
        gyro_data (np.ndarray): N x 3 array of gyroscope readings [gx, gy, gz].
        fs (float): Sampling frequency in Hz.
        cutoff_hz (float): Cutoff frequency in Hz (default 10.0Hz).
        order (int): Filter order (default 2).

    Returns:
        np.ndarray: N x 3 filtered gyroscope array.
    """
    gyro = np.asarray(gyro_data, dtype=np.float64)
    filtered = np.zeros_like(gyro)
    for col in range(3):
        filtered[:, col] = filter_signal_1d(gyro[:, col], fs, cutoff_hz, order)
    return filtered
