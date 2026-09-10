"""Unit tests for Strapdown INS mechanization engine."""

import pytest
import numpy as np
from src.data.observations import IMUObservation
from src.navigation.ins import StrapdownINS
from src.navigation.quaternion import euler_to_quaternion, heading_deg_to_yaw_rad


def test_stationary_imu_ins_behavior():
    """Test that stationary level IMU produces zero net acceleration and zero movement."""
    g = 9.80665
    ins = StrapdownINS(g_val=g)
    
    # Body X = East, Body Y = North, Body Z = Up
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    ins.initialize(
        initial_time=0.0,
        initial_llh=(12.9716, 77.5946, 920.0),
        initial_velocity_enu=np.zeros(3),
        initial_quaternion=q0,
    )

    # Process stationary level IMU readings (a_z = +g)
    for i in range(1, 101):
        obs = IMUObservation(
            timestamp=i * 0.01,
            accelerometer_x=0.0,
            accelerometer_y=0.0,
            accelerometer_z=g,
            gyroscope_x=0.0,
            gyroscope_y=0.0,
            gyroscope_z=0.0,
        )
        state = ins.update(obs)

    assert pytest.approx(np.linalg.norm(state.velocity_enu), abs=1e-5) == 0.0
    assert pytest.approx(np.linalg.norm(state.position_enu), abs=1e-5) == 0.0


def test_constant_acceleration_analytical():
    """Analytical verification: a = 1.0 m/s^2 forward for 10s starting from v0=0.

    Expected: v ~ 10.0 m/s, position ~ 50.0 m.
    """
    g = 9.80665
    ins = StrapdownINS(g_val=g)

    # Body X points East (Heading 90 deg)
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    ins.initialize(
        initial_time=0.0,
        initial_llh=(12.9716, 77.5946, 920.0),
        initial_velocity_enu=np.zeros(3),
        initial_quaternion=q0,
    )

    dt = 0.01
    for i in range(1, 1001):
        obs = IMUObservation(
            timestamp=i * dt,
            accelerometer_x=1.0,  # 1 m/s^2 forward acceleration along East
            accelerometer_y=0.0,
            accelerometer_z=g,
            gyroscope_x=0.0,
            gyroscope_y=0.0,
            gyroscope_z=0.0,
        )
        state = ins.update(obs)

    # Expected East velocity ~ 10.0 m/s, East position ~ 50.0 m
    assert pytest.approx(state.velocity_enu[0], abs=0.05) == 10.0
    assert pytest.approx(state.velocity_enu[1], abs=0.05) == 0.0
    assert pytest.approx(state.position_enu[0], abs=0.1) == 50.0
    assert pytest.approx(state.position_enu[1], abs=0.1) == 0.0


def test_constant_velocity_motion():
    """Test zero net acceleration with initial velocity preserves constant velocity and linear position."""
    g = 9.80665
    ins = StrapdownINS(g_val=g)

    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    v0 = np.array([5.0, 0.0, 0.0])  # 5 m/s East
    ins.initialize(
        initial_time=0.0,
        initial_llh=(12.9716, 77.5946, 920.0),
        initial_velocity_enu=v0,
        initial_quaternion=q0,
    )

    dt = 0.01
    for i in range(1, 1001):  # 10 seconds
        obs = IMUObservation(
            timestamp=i * dt,
            accelerometer_x=0.0,
            accelerometer_y=0.0,
            accelerometer_z=g,
            gyroscope_x=0.0,
            gyroscope_y=0.0,
            gyroscope_z=0.0,
        )
        state = ins.update(obs)

    # Expected velocity = 5 m/s, distance = 50 m
    assert pytest.approx(state.velocity_enu[0], abs=1e-3) == 5.0
    assert pytest.approx(state.position_enu[0], abs=1e-2) == 50.0


def test_invalid_dt_handling():
    """Test that non-positive dt values raise appropriate error."""
    ins = StrapdownINS()
    ins.initialize(0.0, (12.0, 77.0, 100.0), np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))

    # Negative dt
    obs_neg = IMUObservation(
        timestamp=-1.0, accelerometer_x=0.0, accelerometer_y=0.0, accelerometer_z=9.81, gyroscope_x=0.0, gyroscope_y=0.0, gyroscope_z=0.0
    )
    with pytest.raises(ValueError):
        ins.update(obs_neg)
