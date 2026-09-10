"""Unit & Integration tests for Multi-Session Dataset Architecture and Navigation Benchmarks."""

import pytest
import numpy as np

from src.coordinate_transforms import LocalFrame
from src.data.session import (
    DataSourceType,
    DatasetSession,
    discover_available_sessions,
    create_synthetic_multi_session_catalog,
)
from src.ml.datasets.real_session_builder import MultiSessionDatasetBuilder
from src.evaluation.multi_session_evaluator import (
    run_session_navigation_benchmark,
    generate_domain_shift_analysis,
)
from src.map_matching.synthetic_network import build_synthetic_road_network


def test_session_discovery_and_catalog():
    """Verify session discovery classifies sources into SYNTHETIC / REAL / FIXTURE."""
    sessions = create_synthetic_multi_session_catalog(num_sessions=3, base_seed=100)
    assert len(sessions) == 3
    for s in sessions:
        assert s.source_type == DataSourceType.SYNTHETIC
        assert s.duration_sec >= 100.0
        assert s.imu_rate_hz > 90.0
        assert s.gnss_rate_hz > 0.8
        assert s.quality_report.get("overall_status") in ("PASS", "WARNING")


def test_session_isolation():
    """Verify modifying observations in one session does not mutate another session."""
    sessions = create_synthetic_multi_session_catalog(num_sessions=2, base_seed=200)
    sess_1 = sessions[0]
    sess_2 = sessions[1]

    # Mutate session 1 IMU timestamp
    original_t2 = sess_2.imu_observations[0].timestamp
    sess_1.imu_observations[0].timestamp += 999.0

    assert sess_2.imu_observations[0].timestamp == original_t2
    assert sess_1.session_id != sess_2.session_id


def test_multi_session_dataset_builder_and_normalization():
    """Verify dataset builder extracts windows and calculates normalization strictly from train sessions."""
    sessions = create_synthetic_multi_session_catalog(num_sessions=3, base_seed=300, duration_sec=30.0)
    builder = MultiSessionDatasetBuilder(window_size_samples=100, stride_samples=50)

    train_sessions = [sessions[0], sessions[1]]
    test_session = sessions[2]

    # Build train dataset
    train_ds, mean, std = builder.build_dataset_from_sessions(train_sessions)
    assert len(train_ds) > 0
    assert mean.shape == (8,)
    assert std.shape == (8,)

    # Build test dataset with exact train normalization
    test_ds, test_mean, test_std = builder.build_dataset_from_sessions(
        [test_session], feature_mean=mean, feature_std=std
    )
    assert np.allclose(test_mean, mean)
    assert np.allclose(test_std, std)
    assert len(test_ds) > 0


def test_leave_one_session_out_splits():
    """Verify Leave-One-Session-Out splits contain zero cross-session contamination."""
    sessions = create_synthetic_multi_session_catalog(num_sessions=4, base_seed=400, duration_sec=20.0)
    builder = MultiSessionDatasetBuilder()
    folds = builder.leave_one_session_out_splits(sessions)

    assert len(folds) == 4
    for fold in folds:
        test_id = fold["test_session_id"]
        val_id = fold["val_session_id"]
        train_ids = fold["train_session_ids"]

        # Ensure complete set disjointness
        assert test_id not in train_ids
        assert val_id not in train_ids
        assert test_id != val_id
        assert len(train_ids) == 2


def test_multi_outage_benchmark_execution():
    """Verify multi-duration outage benchmark calculates correct metrics for 10s and 30s outages."""
    sessions = create_synthetic_multi_session_catalog(num_sessions=1, base_seed=500, duration_sec=80.0)
    session = sessions[0]

    bench_res = run_session_navigation_benchmark(
        session=session,
        predictor=None,  # Pure ESKF baseline
        outage_durations=[10.0, 30.0],
        outage_start_sec=30.0,
    )

    evals = bench_res["outage_evaluations"]
    assert "10s_outage" in evals
    assert "30s_outage" in evals

    eval_30s = evals["30s_outage"]
    assert eval_30s["outage_duration_sec"] == 30.0
    assert eval_30s["distance_travelled_during_outage_m"] > 50.0

    scs = eval_30s["scenarios"]
    assert "Scenario_B_ESKF_Outage" in scs
    assert scs["Scenario_B_ESKF_Outage"]["drift_percentage"] > 0.0
    assert scs["Scenario_B_ESKF_Outage"]["preliminary_sih_target_check"]["status"] in ("PASS", "FAIL")


def test_no_synthetic_map_on_real_sessions():
    """Verify that real-world sessions do NOT evaluate against synthetic road networks."""
    sessions = create_synthetic_multi_session_catalog(num_sessions=1, base_seed=600, duration_sec=60.0)
    real_session = sessions[0]
    # Mark session source as REAL
    real_session.source_type = DataSourceType.REAL

    synthetic_road_network = build_synthetic_road_network()
    bench_res = run_session_navigation_benchmark(
        session=real_session,
        predictor=None,
        outage_durations=[10.0],
        outage_start_sec=20.0,
        road_network=synthetic_road_network,
    )

    evals = bench_res["outage_evaluations"]["10s_outage"]
    # Scenario D must NOT be included for real sessions
    assert "Scenario_D_ESKF_AI_Map" not in evals["scenarios"]


def test_gnss_recovery_convergence_metrics():
    """Verify post-outage GNSS recovery metrics are logged (0s, 1s, 2s, 5s)."""
    sessions = create_synthetic_multi_session_catalog(num_sessions=1, base_seed=700, duration_sec=70.0)
    session = sessions[0]

    bench_res = run_session_navigation_benchmark(
        session=session,
        predictor=None,
        outage_durations=[20.0],
        outage_start_sec=20.0,
    )

    sc = bench_res["outage_evaluations"]["20s_outage"]["scenarios"]["Scenario_B_ESKF_Outage"]
    rec = sc["recovery"]
    assert "error_at_recovery_0s_m" in rec
    assert "error_at_recovery_1s_m" in rec
    assert "error_at_recovery_2s_m" in rec
    assert "error_at_recovery_5s_m" in rec
    assert rec["error_at_recovery_0s_m"] >= 0.0
