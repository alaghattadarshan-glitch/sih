"""Deterministic Synthetic Sensor Trajectory Generator.

Generates realistic, deterministic vehicle trajectories (IMU 100Hz, GNSS 1Hz, Ground Truth 100Hz)
for software testing and pipeline verification.

IMPORTANT:
- Synthetic data is strictly for pipeline verification and software testing.
- Do NOT use synthetic data to claim real-world navigation accuracy.
"""

import math
import numpy as np
from typing import Tuple, List, Optional

from src.data.observations import IMUObservation, GNSSObservation, GroundTruthObservation
from src.coordinate_transforms import LocalFrame, WGS84Ellipsoid, WGS84


def generate_synthetic_trajectory(
    seed: int = 42,
    duration_sec: float = 80.0,
    imu_rate_hz: float = 100.0,
    gnss_rate_hz: float = 1.0,
    origin_lat: float = 12.9716,
    origin_lon: float = 77.5946,
    origin_alt: float = 920.0,
    accel_noise_std: float = 0.05,
    gyro_noise_std: float = 0.005,
    accel_bias: Tuple[float, float, float] = (0.02, -0.01, 0.03),
    gyro_bias: Tuple[float, float, float] = (0.001, -0.001, 0.002),
    gnss_noise_std: float = 1.5,
) -> Tuple[List[IMUObservation], List[GNSSObservation], List[GroundTruthObservation]]:
    """Generate a deterministic synthetic trajectory with IMU, GNSS, and Ground Truth observations.

    Motion Phases:
    - 0-10s: Acceleration forward (0 -> 10 m/s)
    - 10-40s: Constant velocity cruise (10 m/s)
    - 40-50s: Gradual right turn (yaw rate -0.05 rad/s)
    - 50-70s: Forward travel along turned heading
    - 70-80s: Deceleration to stop (10 -> 0 m/s)

    Args:
        seed (int): Fixed random seed for determinism.
        duration_sec (float): Trajectory duration in seconds (default 80.0s).
        imu_rate_hz (float): IMU sampling frequency in Hz (default 100.0Hz).
        gnss_rate_hz (float): GNSS sampling frequency in Hz (default 1.0Hz).
        origin_lat (float): Origin latitude in degrees.
        origin_lon (float): Origin longitude in degrees.
        origin_alt (float): Origin altitude in meters.
        accel_noise_std (float): Accelerometer Gaussian noise std dev in m/s^2.
        gyro_noise_std (float): Gyroscope Gaussian noise std dev in rad/s.
        accel_bias (Tuple[float, float, float]): Static accelerometer bias in m/s^2.
        gyro_bias (Tuple[float, float, float]): Static gyroscope bias in rad/s.
        gnss_noise_std (float): GNSS horizontal noise std dev in meters.

    Returns:
        Tuple[List[IMUObservation], List[GNSSObservation], List[GroundTruthObservation]]:
            Generated IMU, GNSS, and Ground Truth observation lists.
    """
    rng = np.random.default_rng(seed)
    local_frame = LocalFrame(origin_lat, origin_lon, origin_alt)

    dt_imu = 1.0 / imu_rate_hz
    num_imu_samples = int(duration_sec * imu_rate_hz) + 1
    imu_timestamps = np.linspace(0.0, duration_sec, num_imu_samples)

    dt_gnss = 1.0 / gnss_rate_hz
    num_gnss_samples = int(duration_sec * gnss_rate_hz) + 1
    gnss_timestamps = np.linspace(0.0, duration_sec, num_gnss_samples)

    # Initialize state vectors in local ENU frame
    east = 0.0
    north = 0.0
    up = 0.0
    heading_rad = 0.0  # 0 rad = East, pi/2 = North
    speed = 0.0

    imu_list: List[IMUObservation] = []
    ground_truth_list: List[GroundTruthObservation] = []

    # Gravity vector (Z-axis up)
    g = 9.80665

    # Simulate true state and high-rate IMU
    for idx, t in enumerate(imu_timestamps):
        # Determine acceleration and angular velocity commands based on time phase
        if 0.0 <= t < 10.0:
            accel_cmd = 1.0  # 1.0 m/s^2 forward acceleration
            yaw_rate_cmd = 0.0
        elif 10.0 <= t < 40.0:
            accel_cmd = 0.0
            yaw_rate_cmd = 0.0
        elif 40.0 <= t < 50.0:
            accel_cmd = 0.0
            yaw_rate_cmd = -0.05  # turn rate
        elif 50.0 <= t < 70.0:
            accel_cmd = 0.0
            yaw_rate_cmd = 0.0
        elif 70.0 <= t <= 80.0:
            accel_cmd = -1.0  # -1.0 m/s^2 deceleration
            yaw_rate_cmd = 0.0
        else:
            accel_cmd = 0.0
            yaw_rate_cmd = 0.0

        v_east = speed * math.cos(heading_rad)
        v_north = speed * math.sin(heading_rad)

        # True position to LLH
        lat, lon, alt = local_frame.from_enu(east, north, up)

        # Geographic azimuth (0 deg = North, 90 deg = East)
        geo_heading_deg = math.degrees(math.pi / 2.0 - heading_rad) % 360.0

        # Record Ground Truth observation at timestamp t
        gt_obs = GroundTruthObservation(
            timestamp=float(t),
            latitude=lat,
            longitude=lon,
            altitude=alt,
            velocity_east=v_east,
            velocity_north=v_north,
            velocity_up=0.0,
            speed=speed,
            heading=geo_heading_deg,
            roll=0.0,
            pitch=0.0,
            yaw=math.degrees(heading_rad),
        )
        ground_truth_list.append(gt_obs)

        # Specific force in body frame
        ax_true = accel_cmd
        ay_true = 0.0
        az_true = g  # Gravity +1g pointing up in body Z

        # Gyroscope rates
        gx_true = 0.0
        gy_true = 0.0
        gz_true = yaw_rate_cmd

        # Add bias and noise to simulate raw IMU sensor measurements
        ax_meas = ax_true + accel_bias[0] + rng.normal(0.0, accel_noise_std)
        ay_meas = ay_true + accel_bias[1] + rng.normal(0.0, accel_noise_std)
        az_meas = az_true + accel_bias[2] + rng.normal(0.0, accel_noise_std)

        gx_meas = gx_true + gyro_bias[0] + rng.normal(0.0, gyro_noise_std)
        gy_meas = gy_true + gyro_bias[1] + rng.normal(0.0, gyro_noise_std)
        gz_meas = gz_true + gyro_bias[2] + rng.normal(0.0, gyro_noise_std)

        imu_obs = IMUObservation(
            timestamp=float(t),
            accelerometer_x=ax_meas,
            accelerometer_y=ay_meas,
            accelerometer_z=az_meas,
            gyroscope_x=gx_meas,
            gyroscope_y=gy_meas,
            gyroscope_z=gz_meas,
        )
        imu_list.append(imu_obs)

        # Integrate kinematics for next timestep
        speed = max(0.0, speed + accel_cmd * dt_imu)
        heading_rad += yaw_rate_cmd * dt_imu
        east += v_east * dt_imu
        north += v_north * dt_imu

    # Generate lower-rate GNSS observations from ground truth with added noise
    gnss_list: List[GNSSObservation] = []
    for t_gnss in gnss_timestamps:
        # Find closest ground truth sample
        gt_idx = int(round(t_gnss * imu_rate_hz))
        gt_idx = min(gt_idx, len(ground_truth_list) - 1)
        gt = ground_truth_list[gt_idx]

        # Add horizontal ENU noise to ground truth position
        e_gt, n_gt, u_gt = local_frame.to_enu(gt.latitude, gt.longitude, gt.altitude)
        e_noisy = e_gt + rng.normal(0.0, gnss_noise_std)
        n_noisy = n_gt + rng.normal(0.0, gnss_noise_std)
        u_noisy = u_gt + rng.normal(0.0, gnss_noise_std * 1.5)

        gnss_lat, gnss_lon, gnss_alt = local_frame.from_enu(e_noisy, n_noisy, u_noisy)

        gnss_obs = GNSSObservation(
            timestamp=float(t_gnss),
            latitude=gnss_lat,
            longitude=gnss_lon,
            altitude=gnss_alt,
            velocity_east=gt.velocity_east,
            velocity_north=gt.velocity_north,
            velocity_up=gt.velocity_up,
            speed=gt.speed,
            heading=gt.heading,
            horizontal_accuracy=gnss_noise_std * 2.0,
            vertical_accuracy=gnss_noise_std * 3.0,
        )
        gnss_list.append(gnss_obs)

    return imu_list, gnss_list, ground_truth_list
