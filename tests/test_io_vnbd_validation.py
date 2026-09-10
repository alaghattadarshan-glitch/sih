"""Unit & Integration tests for Step 12 IO-VNBD Acquisition Gate & Algorithm Validation."""

import pytest
import numpy as np
from pathlib import Path

from src.data.session import create_synthetic_multi_session_catalog, DataSourceType
from src.evaluation.error_budget import analyze_error_budget
from src.evaluation.outage_envelope import evaluate_outage_operating_envelope
from scripts.validate_io_vnbd import check_io_vnbd_availability


def test_io_vnbd_availability_gate(tmp_path):
    """Verify IO-VNBD availability gate detects missing vs present files accurately."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)
    
    # Empty directory -> False
    avail, files = check_io_vnbd_availability(raw_dir)
    assert avail is False
    assert len(files) == 0

    # Populated directory -> True
    vnbd_dir = raw_dir / "io_vnbd"
    vnbd_dir.mkdir()
    (vnbd_dir / "session_01_imu.csv").write_text("timestamp,ax,ay,az\n0,0,0,9.81\n")
    
    avail, files = check_io_vnbd_availability(raw_dir)
    assert avail is True
    assert len(files) == 1


def test_error_budget_sensitivity_analysis():
    """Verify error budget analysis calculates baseline and perturbation deltas."""
    sessions = create_synthetic_multi_session_catalog(num_sessions=1, base_seed=1000, duration_sec=50.0)
    session = sessions[0]

    budget_res = analyze_error_budget(
        session=session,
        predictor=None,
        outage_start_sec=20.0,
        outage_duration_sec=20.0,
    )

    assert budget_res["status"] == "SUCCESS"
    assert "baseline_outage_rmse_m" in budget_res
    assert budget_res["baseline_outage_rmse_m"] > 0.0
    
    sens = budget_res["sensitivity_analysis"]
    assert "accelerometer_bias" in sens
    assert "gyroscope_bias" in sens
    assert "initial_attitude_heading_error" in sens
    assert "sensor_noise_increase" in sens

    assert "dominant_error_source" in budget_res
    assert len(budget_res["ranked_error_contributors"]) == 4


def test_outage_operating_envelope_spectrum():
    """Verify outage operating envelope tests multiple durations (5s, 10s, 20s) and returns valid metrics."""
    sessions = create_synthetic_multi_session_catalog(num_sessions=1, base_seed=1100, duration_sec=60.0)
    session = sessions[0]

    envelope_res = evaluate_outage_operating_envelope(
        session=session,
        predictor=None,
        candidate_durations=[5.0, 10.0, 20.0],
        outage_start_sec=20.0,
    )

    assert envelope_res["session_id"] == session.session_id
    evals = envelope_res["outage_duration_evaluations"]
    assert "5s_outage" in evals
    assert "10s_outage" in evals
    assert "20s_outage" in evals

    assert evals["5s_outage"]["outage_duration_sec"] == 5.0
    assert evals["20s_outage"]["outage_duration_sec"] == 20.0
    assert evals["5s_outage"]["distance_travelled_during_outage_m"] > 0.0


def test_ai_gating_and_recovery_metrics_in_envelope():
    """Verify AI gating statistics and post-outage recovery metrics (0s, 1s, 2s, 5s, 10s) are recorded."""
    sessions = create_synthetic_multi_session_catalog(num_sessions=1, base_seed=1200, duration_sec=60.0)
    session = sessions[0]

    envelope_res = evaluate_outage_operating_envelope(
        session=session,
        predictor=None,
        candidate_durations=[15.0],
        outage_start_sec=20.0,
    )

    eval_15s = envelope_res["outage_duration_evaluations"]["15s_outage"]
    assert "ai_gating" in eval_15s
    gating = eval_15s["ai_gating"]
    assert "total_ai_updates" in gating
    assert "accepted_ai_updates" in gating
    assert "acceptance_rate_pct" in gating

    assert "recovery" in eval_15s
    rec = eval_15s["recovery"]
    assert "recovery_0s_error_m" in rec
    assert "recovery_1s_error_m" in rec
    assert "recovery_2s_error_m" in rec
    assert "recovery_5s_error_m" in rec
    assert "recovery_10s_error_m" in rec


def test_system_classification_and_disclaimer():
    """Verify that when real IO-VNBD data is absent, system reports YELLOW classification and honest disclaimer."""
    sessions = create_synthetic_multi_session_catalog(num_sessions=1, base_seed=1300, duration_sec=50.0)
    session = sessions[0]

    envelope_res = evaluate_outage_operating_envelope(
        session=session,
        predictor=None,
        candidate_durations=[10.0],
        outage_start_sec=20.0,
    )

    sih_check = envelope_res["outage_duration_evaluations"]["10s_outage"]["sih_target_check"]
    assert "Preliminary target check on this recording/session." in sih_check["disclaimer"]
