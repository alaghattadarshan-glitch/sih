"""Dataset Quality Audit and Stationarity Calibration Validation.

Analyzes real multi-sensor datasets for timestamp anomalies, sampling rates,
data gaps, numerical health, sensor statistics, spatial coverage, and stationary calibration segments.
"""

import math
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from src.data.observations import IMUObservation, GNSSObservation, GroundTruthObservation
from src.data.adapters.schema import StaticCalibrationConfig
from src.calibration.imu_calibration import estimate_static_bias


def validate_timestamps(
    timestamps: List[float], gap_threshold_factor: float = 3.0
) -> Dict[str, Any]:
    """Analyze timestamp sequence for monotonicity, duplicates, frequency, and gaps.

    Args:
        timestamps (List[float]): Sequence of epoch timestamps in seconds.
        gap_threshold_factor (float): Multiplier on median dt to classify a gap.

    Returns:
        Dict[str, Any]: Timestamp diagnostic metrics.
    """
    if not timestamps or len(timestamps) < 2:
        return {
            "sample_count": len(timestamps),
            "duration_sec": 0.0,
            "sampling_rate_hz": 0.0,
            "median_dt_sec": 0.0,
            "min_dt_sec": 0.0,
            "max_dt_sec": 0.0,
            "duplicate_count": 0,
            "non_monotonic_count": 0,
            "gaps": [],
            "status": "INSUFFICIENT_DATA",
        }

    t_arr = np.asarray(timestamps, dtype=np.float64)
    dts = np.diff(t_arr)

    duplicates = int(np.sum(dts == 0.0))
    non_monotonic = int(np.sum(dts < 0.0))

    valid_dts = dts[dts > 0.0]
    median_dt = float(np.median(valid_dts)) if len(valid_dts) > 0 else 0.0
    sampling_rate = (1.0 / median_dt) if median_dt > 1e-9 else 0.0

    min_dt = float(np.min(dts)) if len(dts) > 0 else 0.0
    max_dt = float(np.max(dts)) if len(dts) > 0 else 0.0
    duration = float(t_arr[-1] - t_arr[0])

    # Find large gaps
    gap_threshold = median_dt * gap_threshold_factor
    gap_indices = np.where(dts > gap_threshold)[0]
    gaps = [
        {
            "start_time": float(t_arr[idx]),
            "end_time": float(t_arr[idx + 1]),
            "gap_sec": float(dts[idx]),
            "gap_ratio": float(dts[idx] / median_dt) if median_dt > 0 else 0.0,
        }
        for idx in gap_indices[:50]  # Cap at first 50 gaps
    ]

    status = "PASS"
    if non_monotonic > 0 or duplicates > 0:
        status = "WARNING"
    if len(gaps) > 10 or duration <= 0:
        status = "WARNING"

    return {
        "sample_count": len(timestamps),
        "start_time": float(t_arr[0]),
        "end_time": float(t_arr[-1]),
        "duration_sec": duration,
        "sampling_rate_hz": float(sampling_rate),
        "median_dt_sec": median_dt,
        "min_dt_sec": min_dt,
        "max_dt_sec": max_dt,
        "duplicate_count": duplicates,
        "non_monotonic_count": non_monotonic,
        "gaps_count": len(gap_indices),
        "gaps": gaps,
        "status": status,
    }


def generate_dataset_quality_report(
    imu_list: List[IMUObservation],
    gnss_list: Optional[List[GNSSObservation]] = None,
    ground_truth_list: Optional[List[GroundTruthObservation]] = None,
    dataset_name: str = "generic_dataset",
) -> Dict[str, Any]:
    """Generate comprehensive diagnostic quality audit report for a multi-sensor dataset.

    Args:
        imu_list (List[IMUObservation]): Canonical IMU samples.
        gnss_list (Optional[List[GNSSObservation]]): Canonical GNSS samples.
        ground_truth_list (Optional[List[GroundTruthObservation]]): Reference ground truth samples.
        dataset_name (str): Identifier name for dataset.

    Returns:
        Dict[str, Any]: Structured machine-readable quality report.
    """
    imu_times = [obs.timestamp for obs in imu_list]
    imu_ts_stats = validate_timestamps(imu_times)

    # IMU Sensor statistics
    if imu_list:
        ax = np.array([obs.accelerometer_x for obs in imu_list])
        ay = np.array([obs.accelerometer_y for obs in imu_list])
        az = np.array([obs.accelerometer_z for obs in imu_list])
        gx = np.array([obs.gyroscope_x for obs in imu_list])
        gy = np.array([obs.gyroscope_y for obs in imu_list])
        gz = np.array([obs.gyroscope_z for obs in imu_list])

        a_norm = np.sqrt(ax**2 + ay**2 + az**2)
        g_norm = np.sqrt(gx**2 + gy**2 + gz**2)

        nan_inf_count = int(
            np.sum(np.isnan(ax) | np.isinf(ax))
            + np.sum(np.isnan(ay) | np.isinf(ay))
            + np.sum(np.isnan(az) | np.isinf(az))
            + np.sum(np.isnan(gx) | np.isinf(gx))
            + np.sum(np.isnan(gy) | np.isinf(gy))
            + np.sum(np.isnan(gz) | np.isinf(gz))
        )

        imu_sensor_stats = {
            "nan_inf_count": nan_inf_count,
            "accel_mean_mps2": [float(np.mean(ax)), float(np.mean(ay)), float(np.mean(az))],
            "accel_std_mps2": [float(np.std(ax)), float(np.std(ay)), float(np.std(az))],
            "accel_norm_mean_mps2": float(np.mean(a_norm)),
            "accel_norm_std_mps2": float(np.std(a_norm)),
            "gyro_mean_rads": [float(np.mean(gx)), float(np.mean(gy)), float(np.mean(gz))],
            "gyro_std_rads": [float(np.std(gx)), float(np.std(gy)), float(np.std(gz))],
            "gyro_norm_mean_rads": float(np.mean(g_norm)),
            "gyro_norm_std_rads": float(np.std(g_norm)),
        }
    else:
        imu_sensor_stats = {}

    # GNSS Diagnostics
    gnss_quality: Dict[str, Any] = {"available": False}
    if gnss_list and len(gnss_list) > 0:
        gnss_times = [obs.timestamp for obs in gnss_list]
        gnss_ts_stats = validate_timestamps(gnss_times)

        lats = np.array([obs.latitude for obs in gnss_list])
        lons = np.array([obs.longitude for obs in gnss_list])
        alts = np.array([obs.altitude for obs in gnss_list])

        # Approximate bounding box and horizontal displacement (haversine)
        d_lat = np.radians(np.max(lats) - np.min(lats))
        d_lon = np.radians(np.max(lons) - np.min(lons))
        mean_lat = np.radians(np.mean(lats))
        # Approximate metric span
        span_north_m = float(d_lat * 6378137.0)
        span_east_m = float(d_lon * 6378137.0 * math.cos(mean_lat))
        displacement_m = float(math.sqrt(span_north_m**2 + span_east_m**2))

        gnss_quality = {
            "available": True,
            "sample_count": len(gnss_list),
            "timestamp_stats": gnss_ts_stats,
            "bounding_box": {
                "min_latitude": float(np.min(lats)),
                "max_latitude": float(np.max(lats)),
                "min_longitude": float(np.min(lons)),
                "max_longitude": float(np.max(lons)),
                "min_altitude_m": float(np.min(alts)),
                "max_altitude_m": float(np.max(alts)),
            },
            "approx_bounding_span_north_m": span_north_m,
            "approx_bounding_span_east_m": span_east_m,
            "approx_horizontal_displacement_m": displacement_m,
        }

    # Ground Truth Diagnostics
    gt_quality: Dict[str, Any] = {"available": False}
    if ground_truth_list and len(ground_truth_list) > 0:
        gt_times = [obs.timestamp for obs in ground_truth_list]
        gt_ts_stats = validate_timestamps(gt_times)
        gt_quality = {
            "available": True,
            "sample_count": len(ground_truth_list),
            "timestamp_stats": gt_ts_stats,
        }

    # Overlapping time interval
    overlap_start = float(imu_times[0]) if imu_times else 0.0
    overlap_end = float(imu_times[-1]) if imu_times else 0.0
    if gnss_list and len(gnss_list) > 0:
        overlap_start = max(overlap_start, float(gnss_list[0].timestamp))
        overlap_end = min(overlap_end, float(gnss_list[-1].timestamp))

    usable_duration = max(0.0, overlap_end - overlap_start)

    overall_status = "PASS"
    if imu_ts_stats.get("status") == "WARNING" or (gnss_quality.get("available") and gnss_quality["timestamp_stats"].get("status") == "WARNING"):
        overall_status = "WARNING"
    if len(imu_list) == 0:
        overall_status = "FAIL"

    return {
        "dataset_name": dataset_name,
        "overall_status": overall_status,
        "usable_overlap_interval": {
            "start_time_sec": overlap_start,
            "end_time_sec": overlap_end,
            "duration_sec": usable_duration,
        },
        "imu_quality": {
            "timestamp_stats": imu_ts_stats,
            "sensor_stats": imu_sensor_stats,
        },
        "gnss_quality": gnss_quality,
        "ground_truth_quality": gt_quality,
    }


def validate_initial_calibration(
    imu_list: List[IMUObservation],
    calib_config: StaticCalibrationConfig,
    expected_gravity: float = 9.80665,
) -> Dict[str, Any]:
    """Validate initial stationary calibration window and calculate sensor biases.

    Args:
        imu_list (List[IMUObservation]): Full IMU observation list.
        calib_config (StaticCalibrationConfig): Calibration window settings.
        expected_gravity (float): Standard local gravity in m/s^2.

    Returns:
        Dict[str, Any]: Calibration validation report and estimated biases.
    """
    if not calib_config.enabled:
        return {
            "calibration_enabled": False,
            "is_stationary": False,
            "accel_bias_mps2": [0.0, 0.0, 0.0],
            "gyro_bias_rads": [0.0, 0.0, 0.0],
            "status": "DISABLED",
            "notice": "Static initial calibration is disabled in configuration.",
        }

    t0 = calib_config.start_time_sec
    t1 = t0 + calib_config.duration_sec

    # Extract calibration window observations
    calib_obs = [obs for obs in imu_list if t0 <= obs.timestamp <= t1]

    if len(calib_obs) < 10:
        return {
            "calibration_enabled": True,
            "is_stationary": False,
            "sample_count": len(calib_obs),
            "accel_bias_mps2": [0.0, 0.0, 0.0],
            "gyro_bias_rads": [0.0, 0.0, 0.0],
            "status": "WARNING",
            "notice": f"Insufficient stationary samples ({len(calib_obs)}) in window [{t0:.2f}s, {t1:.2f}s].",
        }

    # Stationarity variance checks
    ax = np.array([obs.accelerometer_x for obs in calib_obs])
    ay = np.array([obs.accelerometer_y for obs in calib_obs])
    az = np.array([obs.accelerometer_z for obs in calib_obs])
    gx = np.array([obs.gyroscope_x for obs in calib_obs])
    gy = np.array([obs.gyroscope_y for obs in calib_obs])
    gz = np.array([obs.gyroscope_z for obs in calib_obs])

    a_norm = np.sqrt(ax**2 + ay**2 + az**2)
    g_norm = np.sqrt(gx**2 + gy**2 + gz**2)

    accel_std = float(np.std(a_norm))
    gyro_std = float(np.std(g_norm))

    is_stationary = (
        accel_std <= calib_config.max_accel_std_mps2
        and gyro_std <= calib_config.max_gyro_std_rads
    )

    accel_bias, gyro_bias = estimate_static_bias(calib_obs, expected_gravity=expected_gravity)

    status = "VALID" if is_stationary else "WARNING"
    notice = (
        "Initial calibration window verified stationary."
        if is_stationary
        else f"Warning: Sensor motion detected in calibration window (accel std {accel_std:.3f} > {calib_config.max_accel_std_mps2:.3f} or gyro std {gyro_std:.3f} > {calib_config.max_gyro_std_rads:.3f}). Biases may be degraded."
    )

    return {
        "calibration_enabled": True,
        "window_start_sec": t0,
        "window_duration_sec": calib_config.duration_sec,
        "sample_count": len(calib_obs),
        "is_stationary": is_stationary,
        "accel_norm_std_mps2": accel_std,
        "gyro_norm_std_rads": gyro_std,
        "accel_bias_mps2": [float(b) for b in accel_bias],
        "gyro_bias_rads": [float(b) for b in gyro_bias],
        "status": status,
        "notice": notice,
    }
