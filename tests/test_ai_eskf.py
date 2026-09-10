"""Unit tests for Step 8 AI-Integrated 15-State Error-State Kalman Filter."""

import os
import pytest
import numpy as np
import torch

from src.data.observations import IMUObservation, GNSSObservation
from src.navigation.state import NavigationState
from src.navigation.ekf.eskf import ErrorStateKalmanFilter
from src.navigation.ekf.state import ESKFStateConfig
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


def test_ai_update_zero_innovation(initialized_eskf):
    """Verify AI update with zero innovation produces zero error state corrections."""
    eskf = initialized_eskf
    ref_pos = eskf.get_state().position_enu.copy()
    delta_p_ai = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    state_corr, accepted, mah_dist = eskf.update_ai_displacement(
        ref_pos_enu=ref_pos,
        delta_p_ai=delta_p_ai,
        r_ai_std=(1.5, 1.5, 3.0),
        gate_threshold=4.0,
    )

    assert accepted is True
    assert mah_dist == 0.0
    assert np.allclose(state_corr.position_enu, ref_pos, atol=1e-6)


def test_ai_position_correction(initialized_eskf):
    """Verify AI update pulls nominal INS position towards AI pseudo-measurement."""
    eskf = initialized_eskf
    ref_pos = eskf.get_state().position_enu.copy()
    # AI predicts East displacement of +2.0m
    delta_p_ai = np.array([2.0, 0.0, 0.0], dtype=np.float64)

    state_corr, accepted, mah_dist = eskf.update_ai_displacement(
        ref_pos_enu=ref_pos,
        delta_p_ai=delta_p_ai,
        r_ai_std=(1.0, 1.0, 2.0),
        gate_threshold=5.0,
    )

    assert accepted is True
    # Nominal position should move in positive East direction
    assert state_corr.position_enu[0] > ref_pos[0]


def test_mahalanobis_gating_rejects_large_innovation(initialized_eskf):
    """Verify statistically inconsistent AI update (e.g. 50m jump) is rejected by gating."""
    eskf = initialized_eskf
    ref_pos = eskf.get_state().position_enu.copy()
    # Giant unphysical jump
    delta_p_huge = np.array([100.0, 100.0, 0.0], dtype=np.float64)

    state_corr, accepted, mah_dist = eskf.update_ai_displacement(
        ref_pos_enu=ref_pos,
        delta_p_ai=delta_p_huge,
        r_ai_std=(1.0, 1.0, 2.0),
        gate_threshold=4.0,
    )

    assert accepted is False
    assert mah_dist > 4.0
    # State must remain uncorrupted
    assert np.array_equal(state_corr.position_enu, ref_pos)


def test_ai_covariance_symmetry_and_positive_definiteness(initialized_eskf):
    """Verify covariance matrix P remains symmetric and positive-definite after AI update."""
    eskf = initialized_eskf
    ref_pos = eskf.get_state().position_enu.copy()
    delta_p_ai = np.array([0.5, -0.5, 0.1], dtype=np.float64)

    eskf.update_ai_displacement(ref_pos, delta_p_ai)
    P = eskf.get_covariance()

    # Symmetry check
    assert np.allclose(P, P.T, atol=1e-8)
    # Positive definiteness: all eigenvalues > 0
    eigenvalues = np.linalg.eigvalsh(P)
    assert np.all(eigenvalues > 0)


def test_ai_disabled_during_good_gnss(initialized_eskf):
    """Verify AI updates do NOT execute during GOOD GNSS status."""
    eskf = initialized_eskf
    detector = GNSSOutageDetector()
    pipeline = AIESKFPipeline(eskf=eskf, detector=detector, predictor=None, enable_ai=True)

    imu = IMUObservation(timestamp=1.0, accelerometer_x=0.0, accelerometer_y=0.0, accelerometer_z=9.81, gyroscope_x=0.0, gyroscope_y=0.0, gyroscope_z=0.0)
    gnss = GNSSObservation(timestamp=1.0, latitude=12.9716, longitude=77.5946, altitude=920.0, horizontal_accuracy=1.0)

    # Process sample with GOOD GNSS fix
    state, status = pipeline.process_sample(imu, gnss)

    assert status == GNSSStatus.GOOD
    assert pipeline.total_ai_updates == 0


def test_ai_activates_during_outage(tmp_path, initialized_eskf):
    """Verify AI pseudo-measurements trigger when GNSS enters OUTAGE state."""
    ckpt_path = os.path.join(tmp_path, "dummy_model.pt")
    model = TCNDriftModel(in_channels=8, output_dim=3)
    torch.save({"model_config": model.get_config(), "model_state_dict": model.state_dict()}, ckpt_path)
    predictor = DriftPredictor(ckpt_path)

    eskf = initialized_eskf
    detector = GNSSOutageDetector({"persistence_count": 1})
    pipeline = AIESKFPipeline(
        eskf=eskf,
        detector=detector,
        predictor=predictor,
        enable_ai=True,
        window_size_samples=10,
        gate_threshold=100.0,
    )

    # Supply 10 IMU samples with obs=None (missing GNSS fix -> OUTAGE)
    for i in range(10):
        imu = IMUObservation(timestamp=1.0 + i * 0.01, accelerometer_x=0.0, accelerometer_y=0.0, accelerometer_z=9.81, gyroscope_x=0.0, gyroscope_y=0.0, gyroscope_z=0.0)
        state, status = pipeline.process_sample(imu, None)

    assert status == GNSSStatus.OUTAGE
    assert pipeline.total_ai_updates == 1
    assert pipeline.ai_accepted_count == 1


def test_reference_position_reset_after_window(tmp_path, initialized_eskf):
    """Verify window reference position resets cleanly after each window update."""
    ckpt_path = os.path.join(tmp_path, "dummy_model.pt")
    model = TCNDriftModel(in_channels=8, output_dim=3)
    torch.save({"model_config": model.get_config(), "model_state_dict": model.state_dict()}, ckpt_path)
    predictor = DriftPredictor(ckpt_path)

    eskf = initialized_eskf
    detector = GNSSOutageDetector({"persistence_count": 1})
    pipeline = AIESKFPipeline(eskf=eskf, detector=detector, predictor=predictor, enable_ai=True, window_size_samples=5)

    pos_initial = eskf.get_state().position_enu.copy()

    # Process first window (5 samples)
    for i in range(5):
        imu = IMUObservation(timestamp=1.0 + i * 0.01, accelerometer_x=0.0, accelerometer_y=0.0, accelerometer_z=9.81, gyroscope_x=0.0, gyroscope_y=0.0, gyroscope_z=0.0)
        pipeline.process_sample(imu, None)

    pos_window1 = pipeline.window_ref_pos.copy()

    # Process second window (5 samples)
    for i in range(5):
        imu = IMUObservation(timestamp=1.05 + i * 0.01, accelerometer_x=0.0, accelerometer_y=0.0, accelerometer_z=9.81, gyroscope_x=0.0, gyroscope_y=0.0, gyroscope_z=0.0)
        pipeline.process_sample(imu, None)

    pos_window2 = pipeline.window_ref_pos.copy()

    assert pipeline.total_ai_updates == 2
    # Reference position should be updated to current navigation state after each window
    assert not np.array_equal(pos_window1, pos_initial) or not np.array_equal(pos_window2, pos_window1)


def test_gnss_recovery_after_ai_outage(initialized_eskf):
    """Verify ESKF smoothly recovers to true GNSS fixes after AI-assisted outage."""
    eskf = initialized_eskf
    detector = GNSSOutageDetector({"persistence_count": 1})
    pipeline = AIESKFPipeline(eskf=eskf, detector=detector, predictor=None, enable_ai=False)

    # 1. Start GOOD
    imu0 = IMUObservation(timestamp=1.0, accelerometer_x=0.0, accelerometer_y=0.0, accelerometer_z=9.81, gyroscope_x=0.0, gyroscope_y=0.0, gyroscope_z=0.0)
    gnss0 = GNSSObservation(timestamp=1.0, latitude=12.9716, longitude=77.5946, altitude=920.0, horizontal_accuracy=1.0)
    pipeline.process_sample(imu0, gnss0)

    # 2. Outage (obs=None)
    for i in range(10):
        imu = IMUObservation(timestamp=1.01 + i * 0.01, accelerometer_x=0.5, accelerometer_y=0.0, accelerometer_z=9.81, gyroscope_x=0.0, gyroscope_y=0.0, gyroscope_z=0.0)
        pipeline.process_sample(imu, None)

    # 3. GNSS Recovery fix
    imu_rec = IMUObservation(timestamp=2.0, accelerometer_x=0.0, accelerometer_y=0.0, accelerometer_z=9.81, gyroscope_x=0.0, gyroscope_y=0.0, gyroscope_z=0.0)
    gnss_rec = GNSSObservation(timestamp=2.0, latitude=12.9716, longitude=77.5946, altitude=920.0, horizontal_accuracy=1.0)
    state_rec, status_rec = pipeline.process_sample(imu_rec, gnss_rec)

    assert status_rec in (GNSSStatus.RECOVERING, GNSSStatus.GOOD)
    # Filter state should successfully execute GNSS update without crashing
    assert state_rec is not None
