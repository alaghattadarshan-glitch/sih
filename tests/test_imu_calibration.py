"""Unit tests for static IMU bias estimation and calibration."""

import pytest
import numpy as np
from src.data.observations import IMUObservation
from src.calibration import estimate_static_bias, apply_imu_calibration


def test_static_bias_estimation():
    """Test estimation of static accelerometer and gyroscope biases."""
    g = 9.80665
    known_accel_bias = np.array([0.05, -0.02, 0.08])
    known_gyro_bias = np.array([0.002, -0.001, 0.003])

    obs_list = []
    for i in range(100):
        obs = IMUObservation(
            timestamp=i * 0.01,
            accelerometer_x=0.0 + known_accel_bias[0],
            accelerometer_y=0.0 + known_accel_bias[1],
            accelerometer_z=g + known_accel_bias[2],
            gyroscope_x=known_gyro_bias[0],
            gyroscope_y=known_gyro_bias[1],
            gyroscope_z=known_gyro_bias[2],
        )
        obs_list.append(obs)

    accel_bias_est, gyro_bias_est = estimate_static_bias(obs_list, expected_gravity=g)

    assert pytest.approx(accel_bias_est[0], abs=1e-5) == known_accel_bias[0]
    assert pytest.approx(accel_bias_est[1], abs=1e-5) == known_accel_bias[1]
    assert pytest.approx(accel_bias_est[2], abs=1e-5) == known_accel_bias[2]

    assert pytest.approx(gyro_bias_est[0], abs=1e-5) == known_gyro_bias[0]
    assert pytest.approx(gyro_bias_est[1], abs=1e-5) == known_gyro_bias[1]
    assert pytest.approx(gyro_bias_est[2], abs=1e-5) == known_gyro_bias[2]


def test_apply_imu_calibration():
    """Test applying bias corrections to an IMU observation."""
    obs = IMUObservation(
        timestamp=0.0,
        accelerometer_x=0.1,
        accelerometer_y=0.2,
        accelerometer_z=9.9,
        gyroscope_x=0.01,
        gyroscope_y=0.02,
        gyroscope_z=0.03,
    )
    a_bias = np.array([0.1, 0.2, 0.09335])
    g_bias = np.array([0.01, 0.02, 0.03])

    cal_obs = apply_imu_calibration(obs, a_bias, g_bias)

    assert pytest.approx(cal_obs.accelerometer_x, abs=1e-5) == 0.0
    assert pytest.approx(cal_obs.accelerometer_y, abs=1e-5) == 0.0
    assert pytest.approx(cal_obs.accelerometer_z, abs=1e-5) == 9.80665
    assert pytest.approx(cal_obs.gyroscope_x, abs=1e-5) == 0.0
    assert pytest.approx(cal_obs.gyroscope_y, abs=1e-5) == 0.0
    assert pytest.approx(cal_obs.gyroscope_z, abs=1e-5) == 0.0
