"""Unit and Integration Tests for Step 9 Map Matching and Road Network Constraints.

Verifies:
1. Point-to-road projection.
2. Candidate search radius.
3. Heading difference calculation.
4. Candidate scoring.
5. Temporal continuity.
6. Parallel-road disambiguation.
7. Crossing-road disambiguation.
8. Low-confidence candidate rejection.
9. Map innovation gating.
10. Map update covariance validity.
11. Map update state correction.
12. Map-disabled fallback.
13. GNSS recovery.
14. No catastrophic correction from incorrect map candidate.
"""

import math
import pytest
import numpy as np

from src.data.observations import IMUObservation, GNSSObservation
from src.navigation.state import NavigationState
from src.navigation.ekf.eskf import ErrorStateKalmanFilter
from src.outage_detection.detector import GNSSOutageDetector, GNSSStatus
from src.map_matching.types import RoadSegment, RoadNode, MapPoint, RoadClass, MapMatchResult
from src.map_matching.geometry import (
    wrap_angle_180,
    wrap_angle_360,
    calculate_segment_heading,
    point_to_line_segment_projection,
    point_to_polyline_projection,
)
from src.map_matching.network import RoadNetwork
from src.map_matching.matcher import MapMatcher, MapMatcherConfig
from src.map_matching.synthetic_network import build_synthetic_road_network
from src.navigation.ai_fusion import AIESKFPipeline


@pytest.fixture
def test_network() -> RoadNetwork:
    """Fixture providing the synthetic test road network."""
    return build_synthetic_road_network()


@pytest.fixture
def initialized_eskf() -> ErrorStateKalmanFilter:
    """Fixture providing initialized 15-state ESKF."""
    eskf = ErrorStateKalmanFilter()
    init_time = 0.0
    init_llh = (12.9716, 77.5946, 920.0)
    init_vel = np.array([10.0, 0.0, 0.0], dtype=np.float64)  # Traveling East
    # Quaternion for heading East (yaw = 0 rad, geographic azimuth = 90 deg)
    init_q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    eskf.initialize(init_time, init_llh, init_vel, init_q)
    return eskf


def test_point_to_road_projection():
    """Test 1: Verify 2D/3D point-to-line and point-to-polyline projections."""
    poly = np.array([
        [0.0, 0.0, 0.0],
        [100.0, 0.0, 0.0],
        [200.0, 0.0, 0.0],
    ], dtype=np.float64)

    # Point offset North by +15m at East=50m
    query_p = np.array([50.0, 15.0, 0.0], dtype=np.float64)
    proj_pt, dist, along_track, heading = point_to_polyline_projection(query_p, poly)

    assert np.allclose(proj_pt, [50.0, 0.0, 0.0], atol=1e-6)
    assert pytest.approx(dist, abs=1e-6) == 15.0
    assert pytest.approx(along_track, abs=1e-6) == 50.0
    assert pytest.approx(heading, abs=1e-4) == 90.0  # Heading East


def test_candidate_search_radius(test_network):
    """Test 2: Verify candidate query finds segments only within search radius."""
    # Point at (100, 5, 0)
    pos = np.array([100.0, 5.0, 0.0])
    cands_50m = test_network.query_candidate_segments(pos, search_radius_m=50.0)
    cands_10m = test_network.query_candidate_segments(pos, search_radius_m=10.0)
    cands_2m = test_network.query_candidate_segments(pos, search_radius_m=2.0)

    # 50m radius should include main road and parallel north road (offset +30m)
    assert len(cands_50m) >= 2
    # 10m radius should only include main road
    assert len(cands_10m) == 1
    assert cands_10m[0].segment_id == "seg_main_straight_1"
    # 2m radius should find nothing (point is 5m away)
    assert len(cands_2m) == 0


def test_heading_difference_calculation():
    """Test 3: Verify angular difference wrapping and segment azimuth calculation."""
    # Start (0,0) to End (100, 0) -> East -> 90 deg azimuth
    h_east = calculate_segment_heading(np.array([0.0, 0.0, 0.0]), np.array([100.0, 0.0, 0.0]))
    assert pytest.approx(h_east, abs=1e-4) == 90.0

    # Start (0,0) to End (0, 100) -> North -> 0 deg azimuth
    h_north = calculate_segment_heading(np.array([0.0, 0.0, 0.0]), np.array([0.0, 100.0, 0.0]))
    assert pytest.approx(h_north, abs=1e-4) == 0.0

    # Angular wrapping
    assert wrap_angle_180(350.0 - 10.0) == -20.0
    assert wrap_angle_180(10.0 - 350.0) == 20.0
    assert wrap_angle_180(180.0) == -180.0 or wrap_angle_180(180.0) == 180.0


def test_candidate_scoring(test_network):
    """Test 4: Verify candidate scoring weights and Gaussian likelihood decay."""
    matcher = MapMatcher(MapMatcherConfig(sigma_dist_m=10.0, sigma_heading_deg=25.0))
    # State heading East (90 deg), positioned 5m North of main road
    state = NavigationState(
        timestamp=1.0,
        position_enu=np.array([50.0, 5.0, 0.0]),
        velocity_enu=np.array([10.0, 0.0, 0.0]),
        orientation_quaternion=np.array([1.0, 0.0, 0.0, 0.0]),  # 90 deg azimuth
    )

    res = matcher.match(state, test_network)
    assert res.is_matched is True
    assert res.selected_candidate is not None
    assert res.selected_candidate.segment.segment_id == "seg_main_straight_1"
    assert res.confidence > 0.4
    assert res.selected_candidate.position_score > 0.8  # 5m distance with sigma=10m


def test_temporal_continuity(test_network):
    """Test 5: Verify temporal continuity favors previously matched and connected segments."""
    matcher = MapMatcher()
    matcher.last_matched_segment_id = "seg_main_straight_1"
    matcher.last_matched_pos_enu = np.array([195.0, 0.0, 0.0])

    # Position at boundary (East=205m) where seg_main_straight_2 begins
    state = NavigationState(
        timestamp=2.0,
        position_enu=np.array([205.0, 2.0, 0.0]),
        velocity_enu=np.array([10.0, 0.0, 0.0]),
        orientation_quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
    )

    res = matcher.match(state, test_network)
    assert res.is_matched is True
    # Connected segment should receive high transition score and be selected
    assert res.selected_candidate.segment.segment_id == "seg_main_straight_2"


def test_parallel_road_disambiguation(test_network):
    """Test 6: Disambiguate between parallel roads based on distance and continuity."""
    matcher = MapMatcher()
    # Vehicle is close to main road (offset +5m North), while parallel north road is at +30m (25m away)
    state = NavigationState(
        timestamp=5.0,
        position_enu=np.array([150.0, 5.0, 0.0]),
        velocity_enu=np.array([10.0, 0.0, 0.0]),
        orientation_quaternion=np.array([1.0, 0.0, 0.0, 0.0]),  # Heading East
    )

    res = matcher.match(state, test_network)
    assert res.is_matched is True
    assert res.selected_candidate.segment.segment_id == "seg_main_straight_1"
    # Main road score must strictly dominate parallel road score
    scores = {c.segment.segment_id: c.total_score for c in res.all_candidates}
    assert scores["seg_main_straight_1"] > scores["seg_parallel_north"]


def test_crossing_road_disambiguation(test_network):
    """Test 7: Verify crossing road (perpendicular North-South) is penalized by heading constraint."""
    matcher = MapMatcher()
    # Vehicle at intersection (East=200m, North=0m), traveling East (90 deg)
    state = NavigationState(
        timestamp=10.0,
        position_enu=np.array([200.0, 0.0, 0.0]),
        velocity_enu=np.array([10.0, 0.0, 0.0]),
        orientation_quaternion=np.array([1.0, 0.0, 0.0, 0.0]),  # Heading East
    )

    res = matcher.match(state, test_network)
    assert res.is_matched is True
    # Crossing road has 90 deg heading mismatch (> 60 deg threshold), so it must NOT be selected
    assert res.selected_candidate.segment.segment_id != "seg_crossing_north_south"


def test_low_confidence_candidate_rejection(test_network):
    """Test 8: Verify low confidence candidate is gated out and not applied."""
    # Set high min_confidence_threshold
    matcher = MapMatcher(MapMatcherConfig(min_confidence_threshold=0.95))
    state = NavigationState(
        timestamp=1.0,
        position_enu=np.array([100.0, 15.0, 0.0]),  # Equidistant between main and parallel road
        velocity_enu=np.array([10.0, 0.0, 0.0]),
        orientation_quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
    )

    res = matcher.match(state, test_network)
    assert res.is_matched is False
    assert res.gated_out is True
    assert "below threshold" in res.gating_reason


def test_map_innovation_gating(initialized_eskf):
    """Test 9: Verify map update with excessive innovation is rejected by Mahalanobis gating."""
    eskf = initialized_eskf
    pos_before = eskf.get_state().position_enu.copy()

    # Giant unphysical map projection (+100m jump)
    unphysical_proj = pos_before + np.array([100.0, 100.0, 0.0])
    state_out, accepted, mah_dist = eskf.update_map_constraint(
        projected_pos_enu=unphysical_proj,
        confidence=1.0,
        gate_threshold=4.0,
    )

    assert accepted is False
    assert mah_dist > 4.0
    # Nominal state must remain unmodified
    assert np.array_equal(state_out.position_enu, pos_before)


def test_map_update_covariance_validity(initialized_eskf):
    """Test 10: Verify covariance P remains symmetric and positive-definite after map update."""
    eskf = initialized_eskf
    pos_current = eskf.get_state().position_enu
    # Valid small road projection
    proj = pos_current + np.array([0.5, -0.2, 0.0])

    eskf.update_map_constraint(proj, confidence=0.8)
    P = eskf.get_covariance()

    # Symmetry
    assert np.allclose(P, P.T, atol=1e-8)
    # Positive definiteness
    eigenvalues = np.linalg.eigvalsh(P)
    assert np.all(eigenvalues > 0)


def test_map_update_state_correction(initialized_eskf):
    """Test 11: Verify soft map constraint pulls nominal position towards projected road point."""
    eskf = initialized_eskf
    pos_initial = eskf.get_state().position_enu.copy()

    # Map projection offset +2.0m North
    proj = pos_initial + np.array([0.0, 2.0, 0.0])
    state_corr, accepted, _ = eskf.update_map_constraint(proj, confidence=0.9, gate_threshold=5.0)

    assert accepted is True
    # Position must have moved in positive North direction towards road
    assert state_corr.position_enu[1] > pos_initial[1]
    assert state_corr.position_enu[1] < proj[1]  # Soft constraint, not hard snap


def test_map_disabled_fallback(initialized_eskf, test_network):
    """Test 12: Verify pipeline with map matching disabled operates without map updates."""
    detector = GNSSOutageDetector({"persistence_count": 1})
    pipeline = AIESKFPipeline(
        eskf=initialized_eskf,
        detector=detector,
        predictor=None,
        enable_ai=False,
        network=test_network,
        enable_map_matching=False,
    )

    imu = IMUObservation(timestamp=1.0, accelerometer_x=0.0, accelerometer_y=0.0, accelerometer_z=9.81, gyroscope_x=0.0, gyroscope_y=0.0, gyroscope_z=0.0)
    state, status = pipeline.process_sample(imu, None)

    assert status == GNSSStatus.OUTAGE
    assert pipeline.total_map_updates == 0


def test_gnss_recovery_after_map_assisted_outage(initialized_eskf, test_network):
    """Test 13: Verify clean filter recovery when GNSS fix returns after map-assisted outage."""
    detector = GNSSOutageDetector({"persistence_count": 1})
    pipeline = AIESKFPipeline(
        eskf=initialized_eskf,
        detector=detector,
        predictor=None,
        enable_ai=False,
        network=test_network,
        enable_map_matching=True,
    )

    # 1. Start GOOD
    imu0 = IMUObservation(timestamp=1.0, accelerometer_x=0.0, accelerometer_y=0.0, accelerometer_z=9.81, gyroscope_x=0.0, gyroscope_y=0.0, gyroscope_z=0.0)
    gnss0 = GNSSObservation(timestamp=1.0, latitude=12.9716, longitude=77.5946, altitude=920.0, horizontal_accuracy=1.0)
    pipeline.process_sample(imu0, gnss0)

    # 2. Outage with soft map constraint
    for i in range(10):
        imu = IMUObservation(timestamp=1.01 + i * 0.01, accelerometer_x=0.0, accelerometer_y=0.0, accelerometer_z=9.81, gyroscope_x=0.0, gyroscope_y=0.0, gyroscope_z=0.0)
        pipeline.process_sample(imu, None)

    # 3. GNSS recovery fix
    imu_rec = IMUObservation(timestamp=2.0, accelerometer_x=0.0, accelerometer_y=0.0, accelerometer_z=9.81, gyroscope_x=0.0, gyroscope_y=0.0, gyroscope_z=0.0)
    gnss_rec = GNSSObservation(timestamp=2.0, latitude=12.9716, longitude=77.5946, altitude=920.0, horizontal_accuracy=1.0)
    state_rec, status_rec = pipeline.process_sample(imu_rec, gnss_rec)

    assert status_rec in (GNSSStatus.RECOVERING, GNSSStatus.GOOD)
    assert state_rec is not None
    assert np.isfinite(state_rec.position_enu).all()


def test_no_catastrophic_correction_from_bad_candidate(initialized_eskf):
    """Test 14: Verify gating protects filter stability when an incorrect candidate is tested."""
    eskf = initialized_eskf
    pos_before = eskf.get_state().position_enu.copy()

    # Intentionally corrupt candidate (+80m East)
    bad_proj = pos_before + np.array([80.0, -80.0, 0.0])
    state_corr, accepted, mah_dist = eskf.update_map_constraint(bad_proj, confidence=0.5, gate_threshold=4.0)

    assert accepted is False
    assert mah_dist > 4.0
    assert np.array_equal(state_corr.position_enu, pos_before)
