"""Unit tests for sensor observation dataclasses and validation logic."""

import pytest
import math
from src.data.observations import IMUObservation, GNSSObservation, GroundTruthObservation
from src.data.validation import (
    validate_imu_observation,
    validate_gnss_observation,
    validate_ground_truth_observation,
    validate_timestamp_sequence,
    DataValidationError,
)


def test_valid_imu_creation():
    """Test valid IMUObservation creation and validation."""
    obs = IMUObservation(
        timestamp=10.0,
        accelerometer_x=0.1,
        accelerometer_y=0.2,
        accelerometer_z=9.81,
        gyroscope_x=0.01,
        gyroscope_y=-0.02,
        gyroscope_z=0.005,
    )
    warnings = validate_imu_observation(obs)
    assert len(warnings) == 0


def test_invalid_imu_rejection():
    """Test rejection of non-finite numbers and negative timestamps in IMU data."""
    # Negative timestamp
    obs_neg = IMUObservation(
        timestamp=-1.0,
        accelerometer_x=0.0,
        accelerometer_y=0.0,
        accelerometer_z=9.81,
        gyroscope_x=0.0,
        gyroscope_y=0.0,
        gyroscope_z=0.0,
    )
    with pytest.raises(DataValidationError):
        validate_imu_observation(obs_neg)

    # NaN accelerometer
    obs_nan = IMUObservation(
        timestamp=1.0,
        accelerometer_x=float("nan"),
        accelerometer_y=0.0,
        accelerometer_z=9.81,
        gyroscope_x=0.0,
        gyroscope_y=0.0,
        gyroscope_z=0.0,
    )
    with pytest.raises(DataValidationError):
        validate_imu_observation(obs_nan)


def test_valid_gnss_creation():
    """Test valid GNSSObservation creation."""
    obs = GNSSObservation(
        timestamp=1.0,
        latitude=12.9716,
        longitude=77.5946,
        altitude=920.0,
    )
    warnings = validate_gnss_observation(obs)
    assert len(warnings) == 0


def test_invalid_latitude_longitude_rejection():
    """Test rejection of out-of-range geographic coordinates."""
    # Latitude > 90
    obs_lat = GNSSObservation(timestamp=1.0, latitude=95.0, longitude=77.0, altitude=100.0)
    with pytest.raises(DataValidationError):
        validate_gnss_observation(obs_lat)

    # Longitude < -180
    obs_lon = GNSSObservation(timestamp=1.0, latitude=12.0, longitude=-185.0, altitude=100.0)
    with pytest.raises(DataValidationError):
        validate_gnss_observation(obs_lon)


def test_timestamp_sequence_checks():
    """Test timestamp sequence statistics, duplicate detection, and sorting checks."""
    timestamps = [0.0, 1.0, 2.0, 2.0, 3.0, 2.5]
    warnings, stats = validate_timestamp_sequence(timestamps)

    assert stats["duplicate_count"] == 1
    assert stats["unsorted_count"] == 1
    assert len(warnings) >= 2
