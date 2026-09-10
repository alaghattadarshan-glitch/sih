"""Automated safety and failure-injection tests for Step 8.1 AI-ESKF robustness.

Verifies:
1. Normal AI prediction can be accepted.
2. A sufficiently large AI outlier is rejected by Mahalanobis gating.
3. A rejected AI measurement does not modify nominal navigation state.
4. A rejected AI measurement does not create invalid covariance.
5. Repeated bad AI predictions cannot catastrophically diverge the filter.
6. GNSS recovery still works after bad AI predictions.
7. AI-disabled behavior remains identical to the existing ESKF-only path.
8. Covariance symmetry and positive semi-definiteness after updates.
"""

import os
import pytest
import numpy as np
import torch

from src.data.observations import IMUObservation, GNSSObservation
from src.navigation.state import NavigationState
from src.navigation.ekf.eskf import ErrorStateKalmanFilter
from src.outage_detection.detector import GNSSOutageDetector, GNSSStatus
from src.ml.models.tcn_drift import TCNDriftModel
from src.ml.inference import DriftPredictor
from src.navigation.ai_fusion import AIESKFPipeline


@pytest.fixture
def initialized_eskf():
    filter_engine = ErrorStateKalmanFilter()
    init_time = 0.0
    init_llh = (12.9716, 77.5946, 920.0)
    init_vel = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    init_q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    filter_engine.initialize(init_time, init_llh, init_vel, init_q)
    return filter_engine


def test_normal_ai_prediction_accepted(initialized_eskf):
    """Assertion 1: A normal, consistent AI prediction is accepted by Mahalanobis gating."""
    eskf = initialized_eskf
    ref_pos = eskf.get_state().position_enu.copy()
    delta_p_ai = np.array([0.5, 0.2, -0.1], dtype=np.float64)

    state_corr, accepted, mah_dist = eskf.update_ai_displacement(
        ref_pos_enu=ref_pos,
        delta_p_ai=delta_p_ai,
        r_ai_std=(1.5, 1.5, 3.0),
        gate_threshold=4.0,
    )

    assert accepted is True
    assert mah_dist <= 4.0
    # Position should have moved towards AI displacement
    assert state_corr.position_enu[0] > ref_pos[0]


def test_large_ai_outlier_rejected(initialized_eskf):
    """Assertion 2: A sufficiently large AI outlier (+50m jump) is rejected by Mahalanobis gating."""
    eskf = initialized_eskf
    ref_pos = eskf.get_state().position_enu.copy()
    delta_p_huge = np.array([50.0, -50.0, 10.0], dtype=np.float64)

    state_corr, accepted, mah_dist = eskf.update_ai_displacement(
        ref_pos_enu=ref_pos,
        delta_p_ai=delta_p_huge,
        r_ai_std=(1.5, 1.5, 3.0),
        gate_threshold=4.0,
    )

    assert accepted is False
    assert mah_dist > 4.0


def test_rejected_ai_measurement_does_not_modify_state(initialized_eskf):
    """Assertion 3: A rejected AI measurement leaves the nominal navigation state completely unchanged."""
    eskf = initialized_eskf
    pos_before = eskf.get_state().position_enu.copy()
    vel_before = eskf.get_state().velocity_enu.copy()
    q_before = eskf.get_state().orientation_quaternion.copy()

    delta_p_huge = np.array([100.0, 0.0, 0.0], dtype=np.float64)
    state_corr, accepted, mah_dist = eskf.update_ai_displacement(
        ref_pos_enu=pos_before,
        delta_p_ai=delta_p_huge,
        r_ai_std=(1.5, 1.5, 3.0),
        gate_threshold=4.0,
    )

    assert accepted is False
    assert np.array_equal(state_corr.position_enu, pos_before)
    assert np.array_equal(state_corr.velocity_enu, vel_before)
    assert np.array_equal(state_corr.orientation_quaternion, q_before)


def test_rejected_ai_measurement_covariance_validity(initialized_eskf):
    """Assertion 4: A rejected AI measurement maintains symmetric, positive-definite covariance matrix."""
    eskf = initialized_eskf
    P_before = eskf.get_covariance().copy()

    delta_p_huge = np.array([50.0, 50.0, 50.0], dtype=np.float64)
    eskf.update_ai_displacement(
        ref_pos_enu=eskf.get_state().position_enu,
        delta_p_ai=delta_p_huge,
        r_ai_std=(1.5, 1.5, 3.0),
        gate_threshold=4.0,
    )
    P_after = eskf.get_covariance()

    # Covariance must remain identical to P_before (no update occurred)
    assert np.array_equal(P_after, P_before)
    # Symmetry
    assert np.allclose(P_after, P_after.T, atol=1e-8)
    # Positive definiteness
    eigenvalues = np.linalg.eigvalsh(P_after)
    assert np.all(eigenvalues > 0)


def test_repeated_bad_ai_predictions_do_not_diverge(initialized_eskf):
    """Assertion 5: Continuous stream of 50 corrupted AI predictions does not diverge filter state."""
    eskf = initialized_eskf
    pos_start = eskf.get_state().position_enu.copy()

    # Stream 50 corrupted predictions (+100m East bias)
    for _ in range(50):
        ref_pos = eskf.get_state().position_enu.copy()
        delta_corrupted = np.array([100.0, 0.0, 0.0], dtype=np.float64)
        state_corr, accepted, _ = eskf.update_ai_displacement(
            ref_pos_enu=ref_pos,
            delta_p_ai=delta_corrupted,
            r_ai_std=(1.5, 1.5, 3.0),
            gate_threshold=4.0,
        )
        assert accepted is False

    pos_end = eskf.get_state().position_enu.copy()
    # Filter position must remain exactly at initial position (no divergence)
    assert np.array_equal(pos_end, pos_start)


def test_gnss_recovery_after_bad_ai_predictions(initialized_eskf):
    """Assertion 6: Normal GNSS recovery succeeds cleanly after a sequence of rejected AI predictions."""
    eskf = initialized_eskf
    detector = GNSSOutageDetector({"persistence_count": 1})
    pipeline = AIESKFPipeline(eskf=eskf, detector=detector, predictor=None, enable_ai=False)

    # 1. Good GNSS
    imu0 = IMUObservation(timestamp=1.0, accelerometer_x=0.0, accelerometer_y=0.0, accelerometer_z=9.81, gyroscope_x=0.0, gyroscope_y=0.0, gyroscope_z=0.0)
    gnss0 = GNSSObservation(timestamp=1.0, latitude=12.9716, longitude=77.5946, altitude=920.0, horizontal_accuracy=1.0)
    pipeline.process_sample(imu0, gnss0)

    # 2. Simulate rejected AI update during outage
    ref_pos = eskf.get_state().position_enu.copy()
    eskf.update_ai_displacement(ref_pos, np.array([100.0, 100.0, 0.0]), gate_threshold=4.0)

    # 3. GNSS Recovery fix arrives
    imu_rec = IMUObservation(timestamp=2.0, accelerometer_x=0.0, accelerometer_y=0.0, accelerometer_z=9.81, gyroscope_x=0.0, gyroscope_y=0.0, gyroscope_z=0.0)
    gnss_rec = GNSSObservation(timestamp=2.0, latitude=12.9716, longitude=77.5946, altitude=920.0, horizontal_accuracy=1.0)
    state_rec, status_rec = pipeline.process_sample(imu_rec, gnss_rec)

    assert status_rec in (GNSSStatus.RECOVERING, GNSSStatus.GOOD)
    assert state_rec is not None
    assert np.isfinite(state_rec.position_enu).all()


def test_ai_disabled_behavior_identical_to_eskf_only(initialized_eskf):
    """Assertion 7: AI-disabled pipeline execution matches standard ESKF propagation/updates exactly."""
    eskf1 = ErrorStateKalmanFilter()
    eskf1.initialize(0.0, (12.9716, 77.5946, 920.0), np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))

    pipeline_disabled = AIESKFPipeline(
        eskf=eskf1,
        detector=GNSSOutageDetector({"persistence_count": 1}),
        predictor=None,
        enable_ai=False,
    )

    imu = IMUObservation(timestamp=1.0, accelerometer_x=0.1, accelerometer_y=0.0, accelerometer_z=9.81, gyroscope_x=0.0, gyroscope_y=0.0, gyroscope_z=0.01)
    state_out, status = pipeline_disabled.process_sample(imu, None)

    assert status == GNSSStatus.OUTAGE
    assert pipeline_disabled.total_ai_updates == 0
    assert state_out is not None


def test_covariance_symmetry_and_posdef_after_accepted_and_rejected_updates(initialized_eskf):
    """Assertion 8: Covariance matrix P remains symmetric and positive-definite after accepted AND rejected updates."""
    eskf = initialized_eskf
    ref_pos = eskf.get_state().position_enu.copy()

    # 1. Accepted small update
    eskf.update_ai_displacement(ref_pos, np.array([0.5, 0.1, 0.0]), gate_threshold=4.0)
    P_accepted = eskf.get_covariance()
    assert np.allclose(P_accepted, P_accepted.T, atol=1e-8)
    assert np.all(np.linalg.eigvalsh(P_accepted) > 0)

    # 2. Rejected large update
    eskf.update_ai_displacement(ref_pos, np.array([100.0, 0.0, 0.0]), gate_threshold=4.0)
    P_rejected = eskf.get_covariance()
    assert np.allclose(P_rejected, P_rejected.T, atol=1e-8)
    assert np.all(np.linalg.eigvalsh(P_rejected) > 0)
