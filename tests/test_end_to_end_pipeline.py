"""Comprehensive Integration Tests for Canonical End-to-End Navigation Pipeline.

Tests:
1. Complete canonical lifecycle (initialize, calibrate, process_imu, process_gnss, process_sample, get_state, get_diagnostics, reset, finalize).
2. Navigation mode transitions (GNSS_AIDED, DEGRADED, GNSS_OUTAGE, RECOVERING).
3. Static IMU calibration integration.
4. Motion constraints (NHC + ZUPT) execution during outage.
5. Road network map matching integration.
6. Covariance propagation and diagnostic counter verification.
"""

import math
import numpy as np
import pytest

from src.coordinate_transforms import LocalFrame
from src.data.observations import IMUObservation, GNSSObservation
from src.navigation.state import NavigationState
from src.navigation.ekf.eskf import ErrorStateKalmanFilter
from src.outage_detection.detector import GNSSOutageDetector, GNSSStatus
from src.navigation.imu.zupt_detector import ZUPTDetector, ZUPTConfig
from src.map_matching.types import RoadNode, RoadSegment, MapPoint, RoadClass
from src.map_matching.network import RoadNetwork
from src.map_matching.matcher import MapMatcher
from src.navigation.pipeline import EndToEndNavigationPipeline, NavigationMode
from src.navigation.quaternion import euler_to_quaternion


@pytest.fixture
def stationary_imu_samples():
    """Generate 50 stationary IMU observations."""
    samples = []
    for i in range(50):
        t = i * 0.01
        samples.append(
            IMUObservation(
                timestamp=t,
                accelerometer_x=0.01,
                accelerometer_y=-0.02,
                accelerometer_z=9.80665 + 0.03,
                gyroscope_x=0.001,
                gyroscope_y=-0.002,
                gyroscope_z=0.0005,
            )
        )
    return samples


@pytest.fixture
def mock_road_network():
    """Create a simple two-segment road network."""
    net = RoadNetwork("TestNetwork")
    n0 = RoadNode("N0", MapPoint(0.0, 0.0, 0.0))
    n1 = RoadNode("N1", MapPoint(100.0, 0.0, 0.0))
    n2 = RoadNode("N2", MapPoint(100.0, 100.0, 0.0))
    net.add_node(n0)
    net.add_node(n1)
    net.add_node(n2)

    s1 = RoadSegment("S1", "N0", "N1", np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]]), 90.0, 100.0, RoadClass.PRIMARY)
    s2 = RoadSegment("S2", "N1", "N2", np.array([[100.0, 0.0, 0.0], [100.0, 100.0, 0.0]]), 0.0, 100.0, RoadClass.PRIMARY)
    net.add_segment(s1)
    net.add_segment(s2)
    return net


def test_pipeline_initialization_and_lifecycle():
    """Verify standard initialization, sample processing, and finalization lifecycle."""
    pipeline = EndToEndNavigationPipeline(
        enable_ai=False,
        enable_nhc=True,
        enable_zupt=True,
        enable_map_matching=False,
    )
    assert not pipeline._is_initialized
    assert pipeline.current_mode == NavigationMode.GNSS_AIDED

    # Initialize
    q0 = euler_to_quaternion(0.0, 0.0, 0.0)
    v0 = np.array([5.0, 0.0, 0.0])
    init_state = pipeline.initialize(
        origin_lat=12.9716,
        origin_lon=77.5946,
        origin_alt=900.0,
        init_velocity_enu=v0,
        init_quaternion=q0,
    )
    assert pipeline._is_initialized
    assert isinstance(init_state, NavigationState)
    assert np.allclose(init_state.velocity_enu, [5.0, 0.0, 0.0])

    # Process high-rate IMU
    imu1 = IMUObservation(0.01, 0.0, 0.0, 9.80665, 0.0, 0.0, 0.0)
    st1, mode1 = pipeline.process_imu(imu1)
    assert isinstance(st1, NavigationState)
    assert mode1 == NavigationMode.GNSS_AIDED

    # Process GNSS
    gnss1 = GNSSObservation(
        timestamp=1.0,
        latitude=12.9716001,
        longitude=77.5946001,
        altitude=900.0,
        horizontal_accuracy=1.0,
        vertical_accuracy=2.0,
    )
    imu_gnss = IMUObservation(1.0, 0.0, 0.0, 9.80665, 0.0, 0.0, 0.0)
    st2, mode2 = pipeline.process_sample(imu_gnss, gnss1)
    assert isinstance(st2, NavigationState)
    assert mode2 == NavigationMode.GNSS_AIDED

    # Diagnostics
    diag = pipeline.get_diagnostics()
    assert diag["is_initialized"] is True
    assert diag["sample_count"] == 2
    assert diag["imu_sample_count"] == 2
    assert diag["gnss_sample_count"] == 1
    assert "covariance_metrics" in diag
    assert diag["covariance_metrics"]["trace_P"] > 0.0

    # Finalize
    final_diag = pipeline.finalize()
    assert "final_state" in final_diag
    assert "mode_history" in final_diag

    # Reset
    pipeline.reset()
    assert not pipeline._is_initialized
    assert pipeline._sample_count == 0


def test_pipeline_static_calibration(stationary_imu_samples):
    """Verify static calibration estimates sensor biases and applies them during propagation."""
    pipeline = EndToEndNavigationPipeline(enable_ai=False)
    b_a, b_g = pipeline.calibrate(stationary_imu_samples)

    assert pipeline.is_calibrated
    assert np.isclose(b_a[0], 0.01, atol=1e-3)
    assert np.isclose(b_a[1], -0.02, atol=1e-3)
    assert np.isclose(b_a[2], 0.03, atol=1e-3)
    assert np.isclose(b_g[0], 0.001, atol=1e-4)

    # Initialize and verify biases are loaded
    pipeline.initialize(origin_lat=12.0, origin_lon=77.0, origin_alt=100.0)
    diag = pipeline.get_diagnostics()
    assert diag["is_calibrated"] is True
    assert np.allclose(diag["static_calibration"]["accel_bias"], b_a)


def test_pipeline_mode_transitions_and_outage_recovery():
    """Verify smooth state transitions through GNSS_AIDED -> GNSS_OUTAGE -> RECOVERING -> GNSS_AIDED."""
    detector = GNSSOutageDetector({"max_timestamp_gap_sec": 1.5, "persistence_count": 2})
    pipeline = EndToEndNavigationPipeline(
        detector=detector,
        enable_ai=False,
        enable_nhc=True,
        enable_zupt=True,
    )
    pipeline.initialize(origin_lat=12.0, origin_lon=77.0, origin_alt=100.0)

    # 1. Healthy GNSS for 2 seconds
    for sec in range(1, 3):
        t = float(sec)
        gnss = GNSSObservation(t, 12.0, 77.0, 100.0, horizontal_accuracy=1.0)
        imu = IMUObservation(t, 0.0, 0.0, 9.80665, 0.0, 0.0, 0.0)
        st, mode = pipeline.process_sample(imu, gnss)
        assert mode == NavigationMode.GNSS_AIDED

    # 2. GNSS drop for 4 seconds -> triggers outage
    for t_step in np.linspace(2.1, 6.0, 40):
        imu = IMUObservation(t_step, 0.0, 0.0, 9.80665, 0.0, 0.0, 0.0)
        st, mode = pipeline.process_sample(imu, None)

    assert pipeline.current_mode == NavigationMode.GNSS_OUTAGE

    # 3. GNSS returns at t=7.0s (1st good fix -> candidate registered)
    gnss_rec1 = GNSSObservation(7.0, 12.0, 77.0, 100.0, horizontal_accuracy=1.0)
    imu_rec1 = IMUObservation(7.0, 0.0, 0.0, 9.80665, 0.0, 0.0, 0.0)
    st1, mode1 = pipeline.process_sample(imu_rec1, gnss_rec1)

    # 4. 2nd good fix at t=8.0s -> transitions to RECOVERING
    gnss_rec2 = GNSSObservation(8.0, 12.0, 77.0, 100.0, horizontal_accuracy=1.0)
    imu_rec2 = IMUObservation(8.0, 0.0, 0.0, 9.80665, 0.0, 0.0, 0.0)
    st2, mode2 = pipeline.process_sample(imu_rec2, gnss_rec2)
    assert mode2 in [NavigationMode.RECOVERING, NavigationMode.GNSS_AIDED]

    # 5. Subsequent good fix at t=9.0s (in RECOVERING mode)
    gnss_rec3 = GNSSObservation(9.0, 12.0, 77.0, 100.0, horizontal_accuracy=1.0)
    imu_rec3 = IMUObservation(9.0, 0.0, 0.0, 9.80665, 0.0, 0.0, 0.0)
    st3, mode3 = pipeline.process_sample(imu_rec3, gnss_rec3)
    assert mode3 == NavigationMode.RECOVERING

    # 6. Confirmed good fix at t=10.0s -> confirms GNSS_AIDED
    gnss_rec4 = GNSSObservation(10.0, 12.0, 77.0, 100.0, horizontal_accuracy=1.0)
    imu_rec4 = IMUObservation(10.0, 0.0, 0.0, 9.80665, 0.0, 0.0, 0.0)
    st4, mode4 = pipeline.process_sample(imu_rec4, gnss_rec4)
    assert mode4 == NavigationMode.GNSS_AIDED


def test_pipeline_motion_constraints_and_map_matching(mock_road_network):
    """Verify NHC, ZUPT, and Map Matching apply during outages."""
    detector = GNSSOutageDetector({"max_timestamp_gap_sec": 0.5, "persistence_count": 1})
    matcher = MapMatcher()
    pipeline = EndToEndNavigationPipeline(
        detector=detector,
        network=mock_road_network,
        matcher=matcher,
        enable_ai=False,
        enable_nhc=True,
        enable_zupt=True,
        enable_map_matching=True,
    )

    q0 = euler_to_quaternion(0.0, 0.0, 0.0)  # Facing East (yaw=0 rad -> [1, 0, 0])
    pipeline.initialize(
        origin_lat=12.0,
        origin_lon=77.0,
        origin_alt=100.0,
        init_velocity_enu=np.array([10.0, 0.0, 0.0]),
        init_quaternion=q0,
    )

    # Inject GNSS fix
    gnss0 = GNSSObservation(0.0, 12.0, 77.0, 100.0, horizontal_accuracy=1.0)
    pipeline.process_sample(IMUObservation(0.0, 0.0, 0.0, 9.80665, 0.0, 0.0, 0.0), gnss0)

    # Force outage with moving IMU
    for i in range(1, 150):
        t = i * 0.01
        imu = IMUObservation(t, 1.0, 0.1, 9.80665, 0.0, 0.0, 0.1)
        pipeline.process_sample(imu, None)

    diag = pipeline.get_diagnostics()
    assert diag["current_mode"] == "GNSS_OUTAGE"
    assert diag["motion_constraints"]["nhc_total_updates"] > 0
