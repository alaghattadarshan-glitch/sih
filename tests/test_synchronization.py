"""Unit tests for sensor timestamp quality calculations and time synchronization."""

import pytest
from src.data.observations import IMUObservation, GNSSObservation
from src.data.synchronization import (
    estimate_sampling_rate,
    find_timestamp_gaps,
    calculate_timestamp_statistics,
    synchronize_imu_gnss,
    SynchronizedDataset,
)


def test_sampling_rate_calculation():
    """Test estimation of sampling frequency."""
    # 100 Hz signal (dt = 0.01s)
    timestamps = [i * 0.01 for i in range(101)]
    freq = estimate_sampling_rate(timestamps)
    assert pytest.approx(freq, abs=0.1) == 100.0


def test_find_timestamp_gaps():
    """Test gap detection in timestamp sequences."""
    timestamps = [0.0, 0.01, 0.02, 1.0, 1.01, 1.02]
    gaps = find_timestamp_gaps(timestamps, max_expected_dt=0.05)
    assert len(gaps) == 1
    assert gaps[0][0] == 0.02
    assert gaps[0][1] == 1.0
    assert pytest.approx(gaps[0][2], abs=1e-5) == 0.98


def test_calculate_timestamp_statistics():
    """Test calculation of detailed timestamp statistics."""
    timestamps = [0.0, 0.1, 0.2, 0.3, 0.4]
    stats = calculate_timestamp_statistics(timestamps)

    assert stats["count"] == 5
    assert pytest.approx(stats["duration_sec"]) == 0.4
    assert pytest.approx(stats["sampling_rate_hz"]) == 10.0
    assert stats["duplicate_count"] == 0


def test_imu_gnss_synchronization():
    """Test alignment of high-rate IMU and low-rate GNSS observations."""
    imu_list = [
        IMUObservation(timestamp=i * 0.1, accelerometer_x=0.0, accelerometer_y=0.0, accelerometer_z=9.81, gyroscope_x=0.0, gyroscope_y=0.0, gyroscope_z=0.0)
        for i in range(10)
    ]
    gnss_list = [
        GNSSObservation(timestamp=0.201, latitude=12.0, longitude=77.0, altitude=100.0),
        GNSSObservation(timestamp=0.702, latitude=12.001, longitude=77.001, altitude=100.0),
    ]

    sync_results = synchronize_imu_gnss(imu_list, gnss_list, max_tolerance_sec=0.05)
    
    # t_imu = 0.2 should match gnss[0] (t_gnss = 0.201, diff = 0.001s)
    imu_02_match = sync_results[2]
    assert imu_02_match[1] == 0
    assert pytest.approx(imu_02_match[2], abs=1e-4) == 0.001

    # t_imu = 0.7 should match gnss[1] (t_gnss = 0.702, diff = 0.002s)
    imu_07_match = sync_results[7]
    assert imu_07_match[1] == 1


def test_synchronized_dataset_creation():
    """Test SynchronizedDataset object creation and metadata initialization."""
    imu_list = [
        IMUObservation(timestamp=i * 0.01, accelerometer_x=0.0, accelerometer_y=0.0, accelerometer_z=9.81, gyroscope_x=0.0, gyroscope_y=0.0, gyroscope_z=0.0)
        for i in range(101)
    ]
    gnss_list = [
        GNSSObservation(timestamp=0.5, latitude=12.0, longitude=77.0, altitude=100.0)
    ]

    synced = SynchronizedDataset.create(imu_list, gnss_list)
    assert len(synced.imu) == 101
    assert len(synced.gnss) == 1
    assert synced.metadata["status"] == "PASS"
    assert synced.metadata["sync_matched_count"] > 0
