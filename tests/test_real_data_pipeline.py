"""Integration tests for Real Dataset Ingestion, GNSS Outage Simulation, and Metrics Evaluation."""

import os
import pytest
import numpy as np

from src.coordinate_transforms import LocalFrame
from src.data.adapters.schema import (
    DatasetConfig,
    IMUColumnMapping,
    GNSSColumnMapping,
    GroundTruthColumnMapping,
    StaticCalibrationConfig,
    OutageSimulationConfig,
    TimestampUnit,
    AccelUnit,
    GyroUnit,
)
from src.data.adapters.generic_csv import GenericCSVAdapter
from src.evaluation.dataset_quality import (
    generate_dataset_quality_report,
    validate_initial_calibration,
)
from src.navigation.ekf.eskf import ErrorStateKalmanFilter
from src.outage_detection.detector import GNSSOutageDetector
from src.navigation.ai_fusion import AIESKFPipeline


@pytest.fixture
def sample_dataset_config() -> DatasetConfig:
    """Fixture providing configuration pointing to sample CSV recordings."""
    return DatasetConfig(
        dataset_name="test_pipeline",
        imu_file_path="data/sample/imu_sample.csv",
        gnss_file_path="data/sample/gnss_sample.csv",
        ground_truth_file_path="data/sample/ground_truth_sample.csv",
        imu_mapping=IMUColumnMapping(
            timestamp="timestamp",
            accel_x="accel_x",
            accel_y="accel_y",
            accel_z="accel_z",
            gyro_x="gyro_x",
            gyro_y="gyro_y",
            gyro_z="gyro_z",
        ),
        gnss_mapping=GNSSColumnMapping(
            timestamp="timestamp",
            latitude="latitude",
            longitude="longitude",
            altitude="altitude",
            horizontal_accuracy="horizontal_accuracy",
        ),
        ground_truth_mapping=GroundTruthColumnMapping(
            timestamp="timestamp",
            latitude="latitude",
            longitude="longitude",
            altitude="altitude",
            heading="heading",
        ),
        calibration=StaticCalibrationConfig(
            enabled=True,
            start_time_sec=0.0,
            duration_sec=5.0,
            max_accel_std_mps2=0.4,
            max_gyro_std_rads=0.05,
        ),
        outage=OutageSimulationConfig(
            enabled=True,
            start_time_sec=30.0,
            duration_sec=30.0,
        ),
    )


def test_real_dataset_ingestion_and_quality_audit(sample_dataset_config):
    """Verify loading, synchronization, and dataset quality reporting."""
    adapter = GenericCSVAdapter(sample_dataset_config)
    imu_list = adapter.load_imu()
    gnss_list = adapter.load_gnss()
    gt_list = adapter.load_ground_truth()

    assert len(imu_list) > 1000
    assert len(gnss_list) > 50
    assert gt_list is not None and len(gt_list) > 1000

    report = generate_dataset_quality_report(imu_list, gnss_list, gt_list, "test_audit")
    assert report["overall_status"] in ("PASS", "WARNING")
    assert report["usable_overlap_interval"]["duration_sec"] > 70.0


def test_stationary_calibration_check(sample_dataset_config):
    """Verify initial stationary calibration window validation and bias estimation."""
    adapter = GenericCSVAdapter(sample_dataset_config)
    imu_list = adapter.load_imu()

    calib_report = validate_initial_calibration(imu_list, sample_dataset_config.calibration)
    assert calib_report["calibration_enabled"] is True
    assert calib_report["is_stationary"] is True
    assert len(calib_report["accel_bias_mps2"]) == 3
    assert len(calib_report["gyro_bias_rads"]) == 3


def test_stationarity_motion_warning(sample_dataset_config):
    """Verify calibration detects sensor motion and issues a warning."""
    adapter = GenericCSVAdapter(sample_dataset_config)
    imu_list = adapter.load_imu()

    # Highly dynamic motion window (10s to 15s where vehicle is driving) with small max std threshold
    calib_cfg = StaticCalibrationConfig(
        enabled=True,
        start_time_sec=10.0,
        duration_sec=5.0,
        max_accel_std_mps2=0.01,
        max_gyro_std_rads=0.001,
    )
    calib_report = validate_initial_calibration(imu_list, calib_cfg)

    assert calib_report["status"] == "WARNING"
    assert "Sensor motion detected" in calib_report["notice"]


def test_artificial_gnss_outage_removal(sample_dataset_config):
    """Verify GNSS fixes during outage interval [30s, 60s] are completely withheld from estimator."""
    adapter = GenericCSVAdapter(sample_dataset_config)
    imu_list = adapter.load_imu()
    gnss_list = adapter.load_gnss()

    eskf = ErrorStateKalmanFilter()
    eskf.initialize(0.0, (12.9716, 77.5946, 920.0), np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
    pipeline = AIESKFPipeline(eskf=eskf, detector=GNSSOutageDetector({"persistence_count": 1}), enable_ai=False)

    gnss_by_time = {round(g.timestamp, 2): g for g in gnss_list}

    # Simulate 30s-60s outage
    for imu in imu_list:
        t = imu.timestamp
        g_fix = gnss_by_time.get(round(t, 2), None)
        if 30.0 <= t < 60.0:
            g_fix = None  # Withhold GNSS fix

        state, status = pipeline.process_sample(imu, g_fix)

        if 32.0 <= t <= 58.0:
            # Must remain in OUTAGE state
            assert status.value == "OUTAGE"


def test_drift_percentage_and_sih_target_calculation():
    """Verify drift percentage equation and preliminary SIH target check (< 10% pass)."""
    # 500m distance travelled during outage, 40m end-of-outage error -> 8.0% drift (PASS)
    dist_travelled = 500.0
    err_end_pass = 40.0
    drift_pct_pass = (err_end_pass / dist_travelled) * 100.0
    assert drift_pct_pass == 8.0
    assert (drift_pct_pass < 10.0) is True

    # 500m distance travelled, 65m error -> 13.0% drift (FAIL)
    err_end_fail = 65.0
    drift_pct_fail = (err_end_fail / dist_travelled) * 100.0
    assert drift_pct_fail == 13.0
    assert (drift_pct_fail < 10.0) is False


def test_pipeline_operation_without_ground_truth(sample_dataset_config):
    """Verify pipeline runs smoothly when ground truth is completely unavailable."""
    sample_dataset_config.ground_truth_file_path = None
    adapter = GenericCSVAdapter(sample_dataset_config)
    imu_list = adapter.load_imu()
    gnss_list = adapter.load_gnss()
    gt_list = adapter.load_ground_truth()

    assert gt_list is None

    # Pipeline initialization without ground truth
    eskf = ErrorStateKalmanFilter()
    eskf.initialize(0.0, (gnss_list[0].latitude, gnss_list[0].longitude, gnss_list[0].altitude), np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
    pipeline = AIESKFPipeline(eskf=eskf, detector=GNSSOutageDetector(), enable_ai=False)

    for imu in imu_list[:100]:
        state, status = pipeline.process_sample(imu, None)
        assert state is not None
        assert np.isfinite(state.position_enu).all()


def test_pipeline_operation_without_map_and_without_ai(sample_dataset_config):
    """Verify pipeline operates cleanly in classical INS/ESKF mode without AI or map matching."""
    adapter = GenericCSVAdapter(sample_dataset_config)
    imu_list = adapter.load_imu()
    gnss_list = adapter.load_gnss()

    eskf = ErrorStateKalmanFilter()
    eskf.initialize(0.0, (12.9716, 77.5946, 920.0), np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
    pipeline = AIESKFPipeline(
        eskf=eskf,
        detector=GNSSOutageDetector(),
        predictor=None,
        enable_ai=False,
        network=None,
        matcher=None,
        enable_map_matching=False,
    )

    for imu in imu_list[:200]:
        state, status = pipeline.process_sample(imu, None)

    assert pipeline.total_ai_updates == 0
    assert pipeline.total_map_updates == 0
    assert state is not None
