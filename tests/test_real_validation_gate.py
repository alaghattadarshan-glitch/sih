"""Automated Tests for Step 15 Real-World Android Validation Gate.

Tests:
- Android real session adapter and parser
- Sensor rate and timestamp jitter analysis
- Data quality gate auditing and PASS/WARN/FAIL assignment
- Stationary calibration and motion rejection
- Multi-duration outage evaluation and drift percentage calculation
- Post-outage GNSS recovery curve and stability
- SIH Evidence Table generation with strict source tags
"""

import json
import math
import os
from pathlib import Path
import numpy as np
import pytest

from src.coordinate_transforms import LocalFrame
from src.data.observations import IMUObservation, GNSSObservation, GroundTruthObservation
from src.data.adapters.android_real_adapter import AndroidRealDataAdapter, AndroidRealSession
from src.data.loaders.synthetic import generate_synthetic_trajectory
from src.evaluation.sensor_rate_analyzer import SensorRateAnalyzer
from src.evaluation.data_quality_gate import DataQualityGate, DataQualityReport
from src.evaluation.real_validation_runner import (
    run_calibration_test,
    run_controlled_outage_experiment,
    run_recovery_experiment,
    generate_sih_evidence_table,
    run_full_validation_pipeline,
)


@pytest.fixture
def sample_session_data():
    """Fixture providing synthetic trajectory for validation testing."""
    imu_list, gnss_list, gt_list = generate_synthetic_trajectory(seed=42, duration_sec=30.0)
    local_frame = LocalFrame(
        latitude=gt_list[0].latitude,
        longitude=gt_list[0].longitude,
        height=gt_list[0].altitude,
    )
    return imu_list, gnss_list, gt_list, local_frame


def test_android_real_adapter_loading():
    """Verify AndroidRealDataAdapter correctly parses a recorded session directory."""
    session_dir = "data/raw/android_real/session_001"
    if os.path.exists(session_dir):
        session = AndroidRealDataAdapter.load_session(session_dir)
        assert isinstance(session, AndroidRealSession)
        assert session.session_id == "android_real_session_001"
        assert session.imu_count > 0
        assert session.gnss_count > 0
        assert session.duration_sec > 0.0


def test_sensor_rate_analyzer_nominal():
    """Verify SensorRateAnalyzer calculates nominal 100 Hz rate and low jitter."""
    t_nominal = [i * 0.01 for i in range(1001)]  # Exact 100 Hz for 10s
    stats = SensorRateAnalyzer.analyze_timestamps(t_nominal, nominal_hz=100.0)

    assert stats["sample_count"] == 1001
    assert stats["duration_sec"] == pytest.approx(10.0, rel=1e-3)
    assert stats["effective_hz"] == pytest.approx(100.0, rel=1e-3)
    assert stats["mean_dt_ms"] == pytest.approx(10.0, rel=1e-3)
    assert stats["timestamp_gaps_count"] == 0
    assert stats["duplicate_timestamps_count"] == 0


def test_sensor_rate_analyzer_with_jitter_and_gaps():
    """Verify SensorRateAnalyzer detects timestamp jitter, gaps, and duplicates."""
    np.random.seed(42)
    # 100 samples with noise and one large gap
    t = [0.0]
    for i in range(1, 100):
        if i == 50:
            t.append(t[-1] + 0.15)  # 150ms gap (>50ms threshold)
        elif i == 70:
            t.append(t[-1])  # Duplicate
        else:
            t.append(t[-1] + 0.01 + float(np.random.normal(0, 0.001)))

    stats = SensorRateAnalyzer.analyze_timestamps(t, nominal_hz=100.0, gap_threshold_sec=0.05)
    assert stats["timestamp_gaps_count"] >= 1
    assert stats["duplicate_timestamps_count"] >= 1


def test_data_quality_gate_clean_data(sample_session_data):
    """Verify DataQualityGate assigns PASS to healthy trajectory."""
    imu_list, gnss_list, _, _ = sample_session_data
    gate = DataQualityGate()
    report = gate.evaluate(
        imu_list=imu_list,
        gnss_list=gnss_list,
        dataset_name="clean_test_session",
        simulated_outage_duration=10.0,
    )

    assert report.overall_status in ["PASS", "WARN"]
    assert report.criteria["imu_monotonicity"].status == "PASS"
    assert report.criteria["imu_nan_inf"].status == "PASS"
    assert report.criteria["accel_validity"].status == "PASS"
    assert report.criteria["gyro_validity"].status == "PASS"


def test_data_quality_gate_detects_corruptions():
    """Verify DataQualityGate flags NaNs, negative intervals, and invalid coordinates."""
    corrupt_imu = [
        IMUObservation(timestamp=0.0, accelerometer_x=0.0, accelerometer_y=0.0, accelerometer_z=9.81, gyroscope_x=0.0, gyroscope_y=0.0, gyroscope_z=0.0),
        IMUObservation(timestamp=-1.0, accelerometer_x=float("nan"), accelerometer_y=0.0, accelerometer_z=9.81, gyroscope_x=0.0, gyroscope_y=0.0, gyroscope_z=0.0),
        IMUObservation(timestamp=0.02, accelerometer_x=100.0, accelerometer_y=0.0, accelerometer_z=9.81, gyroscope_x=0.0, gyroscope_y=0.0, gyroscope_z=50.0),
    ]
    corrupt_gnss = [
        GNSSObservation(timestamp=0.0, latitude=150.0, longitude=300.0, altitude=-500.0, horizontal_accuracy=-5.0)
    ]

    gate = DataQualityGate()
    report = gate.evaluate(
        imu_list=corrupt_imu,
        gnss_list=corrupt_gnss,
        dataset_name="corrupt_test_session",
    )

    assert report.overall_status == "FAIL"
    assert report.criteria["imu_monotonicity"].status == "FAIL"
    assert report.criteria["imu_nan_inf"].status == "FAIL"
    assert report.criteria["gnss_validity"].status == "FAIL"


def test_stationary_calibration_and_motion_rejection(sample_session_data):
    """Verify stationary calibration succeeds on static data and rejects dynamic motion."""
    imu_list, _, _, _ = sample_session_data
    stationary_segment = imu_list[:300]
    moving_segment = [
        IMUObservation(
            timestamp=obs.timestamp,
            accelerometer_x=obs.accelerometer_x + (1.5 * (i % 5)),
            accelerometer_y=obs.accelerometer_y + (1.0 * (i % 3)),
            accelerometer_z=obs.accelerometer_z,
            gyroscope_x=obs.gyroscope_x + 0.5,
            gyroscope_y=obs.gyroscope_y - 0.5,
            gyroscope_z=obs.gyroscope_z + 0.2,
        )
        for i, obs in enumerate(stationary_segment)
    ]

    report = run_calibration_test(stationary_segment, moving_segment)

    assert report["stationary_calibration_success"] is True
    assert report["motion_rejection_test"]["motion_rejected_correctly"] is True


def test_controlled_outage_experiment(sample_session_data):
    """Verify controlled outage experiment evaluates modes and computes drift percentage."""
    imu_list, gnss_list, gt_list, local_frame = sample_session_data

    outage_report = run_controlled_outage_experiment(
        imu_list=imu_list,
        gnss_list=gnss_list,
        gt_list=gt_list,
        local_frame=local_frame,
        outage_durations=[5.0, 10.0],
    )

    assert "5s" in outage_report
    assert "10s" in outage_report
    assert "ESKF_ONLY" in outage_report["5s"]["modes"]
    assert "ESKF_PLUS_AI" in outage_report["5s"]["modes"]

    drift_5s = outage_report["5s"]["modes"]["ESKF_PLUS_AI"]["drift_percentage"]
    assert drift_5s >= 0.0


def test_recovery_experiment(sample_session_data):
    """Verify post-outage recovery curve calculates errors at defined checkpoints."""
    imu_list, gnss_list, gt_list, local_frame = sample_session_data

    recovery_report = run_recovery_experiment(
        imu_list=imu_list,
        gnss_list=gnss_list,
        gt_list=gt_list,
        local_frame=local_frame,
        outage_start=5.0,
        outage_end=15.0,
    )

    cps = recovery_report["error_at_recovery_checkpoints_m"]
    assert "0.0s" in cps
    assert "0.5s" in cps
    assert "1.0s" in cps
    assert "2.0s" in cps
    assert "5.0s" in cps
    assert "10.0s" in cps
    assert recovery_report["has_discontinuous_jump"] is False


def test_sih_evidence_table_sources():
    """Verify SIH Evidence Table enforces strict provenance labeling."""
    sensor_report = {"sensors": {"imu": {"effective_hz": 100.0, "mean_dt_sec": 0.01}}}
    calib_report = {"stationary_calibration_success": True, "motion_rejection_test": {"motion_rejected_correctly": True}}
    outage_report = {"60s": {"modes": {"ESKF_PLUS_AI": {"drift_percentage": 3.2}}, "ai_drift_reduction_percent": 35.0}}
    recovery_report = {"error_at_recovery_checkpoints_m": {"2.0s": 0.45}}

    table = generate_sih_evidence_table(
        apk_build_success=False,
        physical_device_connected=False,
        device_info=None,
        sensor_rate_report=sensor_report,
        calibration_report=calib_report,
        outage_report=outage_report,
        recovery_report=recovery_report,
        native_parity_rmse=0.0001,
        io_vnbd_status="IO-VNBD raw data unavailable — adapter validated, benchmark execution pending.",
    )

    items = table["evidence_items"]
    assert len(items) >= 10
    valid_sources = {"PHYSICAL_DEVICE", "SOFTWARE_HARNESS", "SYNTHETIC_DATA", "REAL_DATASET", "SOFTWARE_VALIDATION"}
    for it in items:
        assert it["source"] in valid_sources
        assert "criterion" in it
        assert "status" in it
