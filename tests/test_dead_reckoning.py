"""Unit tests for InertialDeadReckoning baseline system."""

import pytest
import numpy as np
from src.data.observations import IMUObservation
from src.navigation.dead_reckoning import InertialDeadReckoning


def test_dead_reckoning_initialization():
    """Test initialization of InertialDeadReckoning."""
    dr = InertialDeadReckoning()
    dr.initialize(
        initial_time=0.0,
        initial_llh=(12.9716, 77.5946, 920.0),
        initial_velocity_enu=np.array([1.0, 2.0, 0.0]),
        initial_quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
    )

    assert dr.elapsed_time == 0.0
    assert pytest.approx(dr.velocity_enu[0]) == 1.0
    assert pytest.approx(dr.velocity_enu[1]) == 2.0
    assert pytest.approx(dr.heading_deg) == 90.0


def test_process_imu_stream():
    """Test processing a stream of IMU observations through dead reckoning."""
    dr = InertialDeadReckoning()
    dr.initialize(
        initial_time=0.0,
        initial_llh=(12.9716, 77.5946, 920.0),
        initial_velocity_enu=np.zeros(3),
        initial_quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
    )

    imu_stream = [
        IMUObservation(
            timestamp=i * 0.01,
            accelerometer_x=0.0,
            accelerometer_y=0.0,
            accelerometer_z=9.80665,
            gyroscope_x=0.0,
            gyroscope_y=0.0,
            gyroscope_z=0.0,
        )
        for i in range(1, 101)
    ]

    states = dr.process_imu_stream(imu_stream)
    assert len(states) == 100
    assert pytest.approx(dr.elapsed_time, abs=1e-3) == 1.0
