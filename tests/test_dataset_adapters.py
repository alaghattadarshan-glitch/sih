"""Unit tests for Dataset Adapters, Schema, Unit Normalization, and Axis Transformations."""

import os
import math
import pytest
import numpy as np

from src.data.adapters.schema import (
    TimestampUnit,
    AccelUnit,
    GyroUnit,
    AxisTransformConfig,
    IMUColumnMapping,
    GNSSColumnMapping,
    DatasetConfig,
)
from src.data.adapters.base import DatasetAdapter
from src.data.adapters.generic_csv import GenericCSVAdapter
from src.data.adapters.io_vnbd import IOVNBDAdapter
from src.evaluation.dataset_quality import validate_timestamps


def test_timestamp_unit_conversions():
    """Verify timestamp normalization across seconds, milliseconds, microseconds, nanoseconds."""
    assert pytest.approx(DatasetAdapter.normalize_timestamp(1.5, TimestampUnit.SECONDS), abs=1e-9) == 1.5
    assert pytest.approx(DatasetAdapter.normalize_timestamp(1500.0, TimestampUnit.MILLISECONDS), abs=1e-9) == 1.5
    assert pytest.approx(DatasetAdapter.normalize_timestamp(1500000.0, TimestampUnit.MICROSECONDS), abs=1e-9) == 1.5
    assert pytest.approx(DatasetAdapter.normalize_timestamp(1500000000.0, TimestampUnit.NANOSECONDS), abs=1e-9) == 1.5


def test_accel_unit_conversions():
    """Verify accelerometer normalization across m/s^2 and g."""
    assert pytest.approx(DatasetAdapter.normalize_accel(9.80665, AccelUnit.MPS2), abs=1e-6) == 9.80665
    assert pytest.approx(DatasetAdapter.normalize_accel(1.0, AccelUnit.G), abs=1e-5) == 9.80665
    assert pytest.approx(DatasetAdapter.normalize_accel(2.5, AccelUnit.G), abs=1e-4) == 24.516625


def test_gyro_unit_conversions():
    """Verify gyroscope normalization across rad/s and deg/s."""
    assert pytest.approx(DatasetAdapter.normalize_gyro(1.0, GyroUnit.RAD_PER_SEC), abs=1e-9) == 1.0
    assert pytest.approx(DatasetAdapter.normalize_gyro(180.0, GyroUnit.DEG_PER_SEC), abs=1e-6) == math.pi
    assert pytest.approx(DatasetAdapter.normalize_gyro(57.2957795, GyroUnit.DEG_PER_SEC), abs=1e-5) == 1.0


def test_axis_transform_valid_matrix():
    """Verify valid orthogonal signed permutation axis transforms."""
    # Identity transform
    cfg_ident = AxisTransformConfig(x_map="+x", y_map="+y", z_map="+z")
    R_ident = cfg_ident.build_matrix()
    assert np.allclose(R_ident, np.eye(3))

    # Permutation: body X = +Y, body Y = -X, body Z = +Z
    cfg_swap = AxisTransformConfig(x_map="+y", y_map="-x", z_map="+z")
    R_swap = cfg_swap.build_matrix()
    assert np.allclose(R_swap @ R_swap.T, np.eye(3))
    assert np.allclose(R_swap @ np.array([1.0, 2.0, 3.0]), np.array([2.0, -1.0, 3.0]))


def test_invalid_axis_transform_rejection():
    """Verify non-orthogonal or duplicate axis mappings are rejected with ValueError."""
    # Duplicate source axis
    with pytest.raises(ValueError, match="Duplicate source axis"):
        AxisTransformConfig(x_map="+x", y_map="+x", z_map="+z").build_matrix()

    # Invalid specifier
    with pytest.raises(ValueError, match="Invalid axis map"):
        AxisTransformConfig(x_map="invalid", y_map="+y", z_map="+z").build_matrix()


def test_generic_csv_unit_normalization():
    """Verify GenericCSVAdapter loads, converts units, and scales values correctly."""
    fixture_path = os.path.join(os.path.dirname(__file__), "fixtures", "test_imu_units.csv")
    cfg = DatasetConfig(
        dataset_name="test_units",
        imu_file_path=fixture_path,
        imu_mapping=IMUColumnMapping(
            timestamp="timestamp_ms",
            accel_x="ax_g",
            accel_y="ay_g",
            accel_z="az_g",
            gyro_x="gx_dps",
            gyro_y="gy_dps",
            gyro_z="gz_dps",
            timestamp_unit=TimestampUnit.MILLISECONDS,
            accel_unit=AccelUnit.G,
            gyro_unit=GyroUnit.DEG_PER_SEC,
            axis_transform=AxisTransformConfig(x_map="+x", y_map="+y", z_map="+z"),
        ),
    )

    adapter = GenericCSVAdapter(cfg)
    imu_list = adapter.load_imu()

    assert len(imu_list) == 5
    # Timestamp converted from 10ms to 0.01s
    assert pytest.approx(imu_list[1].timestamp, abs=1e-6) == 0.01
    # Accel Z converted from 1.0g to 9.80665 m/s^2
    assert pytest.approx(imu_list[0].accelerometer_z, abs=1e-4) == 9.80665
    # Gyro Z converted from 5.729578 deg/s to 0.1 rad/s
    assert pytest.approx(imu_list[1].gyroscope_z, abs=1e-4) == 0.1


def test_generic_csv_missing_column_rejection():
    """Verify adapter raises descriptive ValueError if required column is absent."""
    fixture_path = os.path.join(os.path.dirname(__file__), "fixtures", "test_imu_units.csv")
    cfg = DatasetConfig(
        dataset_name="test_missing",
        imu_file_path=fixture_path,
        imu_mapping=IMUColumnMapping(
            timestamp="non_existent_timestamp",
            accel_x="ax_g",
            accel_y="ay_g",
            accel_z="az_g",
            gyro_x="gx_dps",
            gyro_y="gy_dps",
            gyro_z="gz_dps",
        ),
    )

    adapter = GenericCSVAdapter(cfg)
    with pytest.raises(ValueError, match="Missing required IMU columns"):
        adapter.load_imu()


def test_timestamp_anomalies_validation():
    """Verify validation detects duplicate, non-monotonic timestamps, and gaps."""
    fixture_path = os.path.join(os.path.dirname(__file__), "fixtures", "test_imu_anomalies.csv")
    cfg = DatasetConfig(dataset_name="test_anomalies", imu_file_path=fixture_path)
    adapter = GenericCSVAdapter(cfg)
    imu_list = adapter.load_imu()

    times = [obs.timestamp for obs in imu_list]
    stats = validate_timestamps(times)

    assert stats["duplicate_count"] == 1
    assert stats["non_monotonic_count"] == 1
    assert stats["status"] == "WARNING"


def test_io_vnbd_adapter_standby_and_template():
    """Verify IOVNBDAdapter initializes schema template and handles standby mode gracefully."""
    cfg = IOVNBDAdapter.create_default_config(base_dir="data/raw/io_vnbd", sequence_name="seq_01")
    adapter = IOVNBDAdapter(cfg)

    # When files are not present locally
    assert adapter.is_dataset_available() is False
    meta = adapter.get_metadata()
    assert meta["dataset_available"] is False
    assert "pending dataset archive" in meta["availability_notice"]
