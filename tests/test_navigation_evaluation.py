"""Unit tests for trajectory evaluation metrics engine."""

import pytest
import numpy as np
from src.navigation.state import NavigationState
from src.data.observations import GroundTruthObservation
from src.coordinate_transforms import LocalFrame
from src.evaluation import evaluate_trajectory


def test_trajectory_evaluation_zero_error():
    """Test trajectory evaluation metrics when estimated trajectory matches ground truth perfectly."""
    origin_lat, origin_lon, origin_alt = 12.9716, 77.5946, 920.0
    local_frame = LocalFrame(origin_lat, origin_lon, origin_alt)

    est_states = []
    gt_list = []

    for i in range(10):
        t = i * 1.0
        # Position 10m East
        e, n, u = 10.0 * i, 0.0, 0.0
        lat, lon, alt = local_frame.from_enu(e, n, u)

        est_state = NavigationState(
            timestamp=t,
            position_enu=np.array([e, n, u]),
            velocity_enu=np.array([10.0, 0.0, 0.0]),
            orientation_quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        )
        est_states.append(est_state)

        gt_obs = GroundTruthObservation(
            timestamp=t,
            latitude=lat,
            longitude=lon,
            altitude=alt,
            velocity_east=10.0,
            velocity_north=0.0,
            velocity_up=0.0,
            speed=10.0,
            heading=90.0,
        )
        gt_list.append(gt_obs)

    res = evaluate_trajectory(est_states, gt_list, local_frame)

    assert res["status"] == "PASS"
    assert pytest.approx(res["horizontal_rmse"], abs=1e-4) == 0.0
    assert pytest.approx(res["max_horizontal_error"], abs=1e-4) == 0.0
    assert pytest.approx(res["final_position_error"], abs=1e-4) == 0.0
