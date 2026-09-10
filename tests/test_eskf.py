"""Unit tests for 15-State Error-State Kalman Filter (ESKF)."""

import pytest
import numpy as np

from src.data.observations import IMUObservation, GNSSObservation
from src.navigation.state import NavigationState
from src.navigation.quaternion import quaternion_to_heading_deg
from src.navigation.ekf import (
    ErrorStateKalmanFilter,
    ESKFStateConfig,
    STATE_DIM,
    skew_symmetric,
    build_continuous_error_matrix,
    build_continuous_noise_matrix,
    discretize_dynamics,
    build_gnss_position_measurement_matrix,
)


def test_state_dimension_and_covariance_shape():
    """Test 15-state dimensions and initial covariance shape and symmetry."""
    assert STATE_DIM == 15

    config = ESKFStateConfig()
    P0 = config.build_initial_covariance()

    assert P0.shape == (15, 15)
    # Symmetry check
    assert np.allclose(P0, P0.T)
    # Positive diagonal check
    assert np.all(np.diag(P0) > 0)


def test_skew_symmetric():
    """Test 3x3 skew-symmetric matrix properties."""
    v = np.array([1.0, 2.0, 3.0])
    S = skew_symmetric(v)

    assert S.shape == (3, 3)
    assert np.allclose(S, -S.T)
    # Cross product equivalence: S * u = v x u
    u = np.array([4.0, 5.0, 6.0])
    assert np.allclose(S @ u, np.cross(v, u))


def test_eskf_prediction_step():
    """Test ESKF prediction step covariance growth and state propagation."""
    eskf = ErrorStateKalmanFilter()
    eskf.initialize(
        initial_time=0.0,
        initial_llh=(12.9716, 77.5946, 920.0),
        initial_velocity_enu=np.zeros(3),
        initial_quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
    )

    P_init = eskf.get_covariance()

    obs = IMUObservation(
        timestamp=0.01,
        accelerometer_x=0.0,
        accelerometer_y=0.0,
        accelerometer_z=9.80665,
        gyroscope_x=0.0,
        gyroscope_y=0.0,
        gyroscope_z=0.0,
    )
    state = eskf.predict(obs)

    P_pred = eskf.get_covariance()

    assert state.timestamp == 0.01
    assert np.allclose(P_pred, P_pred.T)
    # Covariance trace should increase after prediction
    assert np.trace(P_pred) >= np.trace(P_init)


def test_eskf_zero_innovation_update():
    """Test GNSS update when measurement matches nominal position exactly (zero innovation)."""
    eskf = ErrorStateKalmanFilter()
    eskf.initialize(
        initial_time=0.0,
        initial_llh=(12.9716, 77.5946, 920.0),
        initial_velocity_enu=np.zeros(3),
        initial_quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
    )

    gnss_obs = GNSSObservation(
        timestamp=0.0,
        latitude=12.9716,
        longitude=77.5946,
        altitude=920.0,
    )

    P_before = eskf.get_covariance()
    state_after = eskf.update_gnss(gnss_obs)
    P_after = eskf.get_covariance()

    assert np.allclose(state_after.position_enu, np.zeros(3), atol=1e-5)
    # Measurement update reduces uncertainty trace
    assert np.trace(P_after) <= np.trace(P_before)


def test_eskf_known_position_correction():
    """Test position state correction when GNSS fix is offset from nominal INS."""
    eskf = ErrorStateKalmanFilter()
    eskf.initialize(
        initial_time=0.0,
        initial_llh=(12.9716, 77.5946, 920.0),
        initial_velocity_enu=np.zeros(3),
        initial_quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
    )

    # Convert offset position (10m East) to LLH for GNSS obs
    e_target, n_target, u_target = 10.0, 0.0, 0.0
    target_lat, target_lon, target_alt = eskf.ins.local_frame.from_enu(
        e_target, n_target, u_target
    )

    gnss_obs = GNSSObservation(
        timestamp=0.0,
        latitude=target_lat,
        longitude=target_lon,
        altitude=target_alt,
        horizontal_accuracy=0.5,
    )

    state_after = eskf.update_gnss(gnss_obs)

    # Nominal position should be pulled significantly toward 10m East
    assert state_after.position_enu[0] > 5.0


def test_eskf_gnss_outage_and_recovery():
    """Test filter behavior during GNSS outage and subsequent GNSS recovery."""
    eskf = ErrorStateKalmanFilter()
    eskf.initialize(
        initial_time=0.0,
        initial_llh=(12.9716, 77.5946, 920.0),
        initial_velocity_enu=np.zeros(3),
        initial_quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
    )

    # Predict during outage for 50 steps (0.5s) without GNSS
    for i in range(1, 51):
        obs = IMUObservation(
            timestamp=i * 0.01,
            accelerometer_x=0.0,
            accelerometer_y=0.0,
            accelerometer_z=9.80665,
            gyroscope_x=0.0,
            gyroscope_y=0.0,
            gyroscope_z=0.0,
        )
        state = eskf.predict(obs)

    P_outage = eskf.get_covariance()
    assert state.timestamp == 0.50

    # Apply GNSS recovery fix at t=0.5s
    gnss_obs = GNSSObservation(
        timestamp=0.50,
        latitude=12.9716,
        longitude=77.5946,
        altitude=920.0,
        horizontal_accuracy=0.5,
    )
    state_rec = eskf.update_gnss(gnss_obs)
    P_rec = eskf.get_covariance()

    assert state_rec is not None
    # Uncertainty trace drops upon GNSS recovery update
    assert np.trace(P_rec) < np.trace(P_outage)


def test_invalid_prediction_dt():
    """Test rejection of invalid non-positive timestamp increments."""
    eskf = ErrorStateKalmanFilter()
    eskf.initialize(
        initial_time=0.0,
        initial_llh=(12.9716, 77.5946, 920.0),
        initial_velocity_enu=np.zeros(3),
        initial_quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
    )

    obs_invalid = IMUObservation(
        timestamp=-1.0,
        accelerometer_x=0.0,
        accelerometer_y=0.0,
        accelerometer_z=9.81,
        gyroscope_x=0.0,
        gyroscope_y=0.0,
        gyroscope_z=0.0,
    )
    with pytest.raises(ValueError):
        eskf.predict(obs_invalid)
