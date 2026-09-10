"""Unit tests for CSV loaders and DataFrame conversion utilities."""

import pytest
from pathlib import Path
from src.data.loaders import load_imu_csv, load_gnss_csv, load_ground_truth_csv
from src.data.conversions import (
    imu_to_dataframe,
    gnss_to_dataframe,
    dataframe_to_imu,
    dataframe_to_gnss,
)


@pytest.fixture
def sample_dir():
    return Path(__file__).resolve().parent.parent / "data" / "sample"


def test_csv_loading(sample_dir):
    """Test loading IMU, GNSS, and Ground Truth observations from sample CSV files."""
    imu_path = sample_dir / "imu_sample.csv"
    gnss_path = sample_dir / "gnss_sample.csv"
    gt_path = sample_dir / "ground_truth_sample.csv"

    assert imu_path.exists(), "Sample IMU CSV file missing."
    assert gnss_path.exists(), "Sample GNSS CSV file missing."

    imu_list = load_imu_csv(imu_path)
    gnss_list = load_gnss_csv(gnss_path)

    assert len(imu_list) > 0
    assert len(gnss_list) > 0
    assert imu_list[0].timestamp == 0.0
    assert gnss_list[0].timestamp == 0.0

    if gt_path.exists():
        gt_list = load_ground_truth_csv(gt_path)
        assert len(gt_list) > 0


def test_dataframe_conversions(sample_dir):
    """Test roundtrip conversion between observation lists and Pandas DataFrames."""
    imu_list = load_imu_csv(sample_dir / "imu_sample.csv")
    df = imu_to_dataframe(imu_list)

    assert "timestamp" in df.columns
    assert "accel_x" in df.columns
    assert "gyro_z" in df.columns
    assert len(df) == len(imu_list)

    rec_imu = dataframe_to_imu(df)
    assert len(rec_imu) == len(imu_list)
    assert rec_imu[0].timestamp == imu_list[0].timestamp
    assert rec_imu[0].accelerometer_z == imu_list[0].accelerometer_z


def test_deg_to_rad_gyro_conversion(tmp_path):
    """Test gyroscope unit conversion from deg/s to rad/s during CSV loading."""
    csv_file = tmp_path / "imu_deg.csv"
    csv_file.write_text(
        "timestamp,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z\n"
        "0.0,0.0,0.0,9.81,180.0,0.0,0.0\n"
    )

    imu = load_imu_csv(csv_file, gyro_unit="deg/s")
    assert pytest.approx(imu[0].gyroscope_x, abs=1e-5) == 3.1415926535  # pi rad/s
