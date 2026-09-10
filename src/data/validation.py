"""Sensor Data Validation Utilities.

Provides functions to validate observation fields, check numerical limits (NaN, Inf),
flag suspicious values, and verify geographic bounds without silently discarding raw data.
"""

import math
from typing import List, Tuple
from src.data.observations import IMUObservation, GNSSObservation, GroundTruthObservation


class DataValidationError(ValueError):
    """Raised when an observation contains fatal invalid data (e.g. NaN timestamp, lat out of range)."""

    pass


def validate_imu_observation(
    obs: IMUObservation,
    max_accel_threshold: float = 200.0,
    max_gyro_threshold: float = 50.0,
) -> List[str]:
    """Validate an IMU observation.

    Args:
        obs (IMUObservation): Observation to validate.
        max_accel_threshold (float): Suspicious acceleration magnitude threshold in m/s^2.
        max_gyro_threshold (float): Suspicious angular velocity threshold in rad/s.

    Returns:
        List[str]: List of warning messages for flagged suspicious values.

    Raises:
        DataValidationError: If timestamp or required fields contain NaN/Inf or negative time.
    """
    warnings: List[str] = []

    # Fatal timestamp checks
    if not math.isfinite(obs.timestamp):
        raise DataValidationError("IMU timestamp must be a finite number.")
    if obs.timestamp < 0:
        raise DataValidationError(f"IMU timestamp cannot be negative: {obs.timestamp}")

    # Check finite numbers for required fields
    fields = [
        ("accelerometer_x", obs.accelerometer_x),
        ("accelerometer_y", obs.accelerometer_y),
        ("accelerometer_z", obs.accelerometer_z),
        ("gyroscope_x", obs.gyroscope_x),
        ("gyroscope_y", obs.gyroscope_y),
        ("gyroscope_z", obs.gyroscope_z),
    ]

    for name, val in fields:
        if not math.isfinite(val):
            raise DataValidationError(f"IMU field '{name}' is non-finite (NaN or Inf).")

    # Flag suspicious acceleration magnitude
    accel_mag = math.sqrt(
        obs.accelerometer_x**2 + obs.accelerometer_y**2 + obs.accelerometer_z**2
    )
    if accel_mag > max_accel_threshold:
        warnings.append(
            f"Suspiciously high IMU acceleration magnitude: {accel_mag:.2f} m/s^2 at t={obs.timestamp:.3f}s"
        )

    # Flag suspicious gyroscope magnitude
    gyro_mag = math.sqrt(obs.gyroscope_x**2 + obs.gyroscope_y**2 + obs.gyroscope_z**2)
    if gyro_mag > max_gyro_threshold:
        warnings.append(
            f"Suspiciously high IMU gyroscope rate: {gyro_mag:.2f} rad/s at t={obs.timestamp:.3f}s"
        )

    return warnings


def validate_gnss_observation(obs: GNSSObservation) -> List[str]:
    """Validate a GNSS observation.

    Args:
        obs (GNSSObservation): Observation to validate.

    Returns:
        List[str]: List of warning messages.

    Raises:
        DataValidationError: If coordinates or timestamp are invalid or out of bounds.
    """
    warnings: List[str] = []

    if not math.isfinite(obs.timestamp):
        raise DataValidationError("GNSS timestamp must be a finite number.")
    if obs.timestamp < 0:
        raise DataValidationError(f"GNSS timestamp cannot be negative: {obs.timestamp}")

    if not math.isfinite(obs.latitude) or not (-90.0 <= obs.latitude <= 90.0):
        raise DataValidationError(f"Invalid GNSS latitude: {obs.latitude}")

    if not math.isfinite(obs.longitude) or not (-180.0 <= obs.longitude <= 180.0):
        raise DataValidationError(f"Invalid GNSS longitude: {obs.longitude}")

    if not math.isfinite(obs.altitude):
        raise DataValidationError(f"Invalid GNSS altitude: {obs.altitude}")

    if obs.horizontal_accuracy is not None and obs.horizontal_accuracy > 100.0:
        warnings.append(
            f"Low GNSS horizontal accuracy: {obs.horizontal_accuracy:.1f}m at t={obs.timestamp:.3f}s"
        )

    return warnings


def validate_ground_truth_observation(obs: GroundTruthObservation) -> List[str]:
    """Validate a Ground Truth observation."""
    warnings: List[str] = []

    if not math.isfinite(obs.timestamp):
        raise DataValidationError("Ground Truth timestamp must be a finite number.")
    if obs.timestamp < 0:
        raise DataValidationError(
            f"Ground Truth timestamp cannot be negative: {obs.timestamp}"
        )

    if not math.isfinite(obs.latitude) or not (-90.0 <= obs.latitude <= 90.0):
        raise DataValidationError(f"Invalid Ground Truth latitude: {obs.latitude}")

    if not math.isfinite(obs.longitude) or not (-180.0 <= obs.longitude <= 180.0):
        raise DataValidationError(f"Invalid Ground Truth longitude: {obs.longitude}")

    if not math.isfinite(obs.altitude):
        raise DataValidationError(f"Invalid Ground Truth altitude: {obs.altitude}")

    return warnings


def validate_timestamp_sequence(timestamps: List[float]) -> Tuple[List[str], dict]:
    """Validate a sequence of timestamps for sorting, duplicates, gaps, and statistics.

    Args:
        timestamps (List[float]): Sequence of timestamps.

    Returns:
        Tuple[List[str], dict]: Warnings list and statistics dictionary.
    """
    warnings: List[str] = []
    stats = {
        "count": len(timestamps),
        "duplicate_count": 0,
        "unsorted_count": 0,
        "mean_dt": 0.0,
        "std_dt": 0.0,
        "estimated_freq_hz": 0.0,
        "max_gap_sec": 0.0,
    }

    if not timestamps:
        warnings.append("Timestamp sequence is empty.")
        return warnings, stats

    dts = []
    duplicates = 0
    unsorted = 0

    for i in range(1, len(timestamps)):
        dt = timestamps[i] - timestamps[i - 1]
        if dt == 0:
            duplicates += 1
        elif dt < 0:
            unsorted += 1
        else:
            dts.append(dt)

    stats["duplicate_count"] = duplicates
    stats["unsorted_count"] = unsorted

    if duplicates > 0:
        warnings.append(f"Found {duplicates} duplicate timestamps in sequence.")

    if unsorted > 0:
        warnings.append(f"Found {unsorted} non-monotonically increasing timestamps.")

    if dts:
        mean_dt = sum(dts) / len(dts)
        var_dt = sum((x - mean_dt) ** 2 for x in dts) / len(dts)
        std_dt = math.sqrt(var_dt)

        stats["mean_dt"] = mean_dt
        stats["std_dt"] = std_dt
        stats["estimated_freq_hz"] = 1.0 / mean_dt if mean_dt > 0 else 0.0
        stats["max_gap_sec"] = max(dts)

        if stats["max_gap_sec"] > 5.0 * mean_dt:
            warnings.append(
                f"Large timestamp gap detected: max gap = {stats['max_gap_sec']:.3f}s (mean dt = {mean_dt:.3f}s)"
            )

    return warnings, stats
