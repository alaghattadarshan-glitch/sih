"""Unit tests for deterministic synthetic trajectory generator."""

import pytest
from src.data.loaders.synthetic import generate_synthetic_trajectory


def test_synthetic_trajectory_generation():
    """Test deterministic generation of synthetic sensor data."""
    imu_1, gnss_1, gt_1 = generate_synthetic_trajectory(seed=42, duration_sec=10.0)
    imu_2, gnss_2, gt_2 = generate_synthetic_trajectory(seed=42, duration_sec=10.0)

    # 10s at 100Hz = 1001 samples
    assert len(imu_1) == 1001
    # 10s at 1Hz = 11 samples
    assert len(gnss_1) == 11
    assert len(gt_1) == 1001

    # Check strict determinism with fixed seed
    assert imu_1[10].accelerometer_x == imu_2[10].accelerometer_x
    assert gnss_1[5].latitude == gnss_2[5].latitude
    assert gt_1[100].speed == gt_2[100].speed


def test_synthetic_trajectory_motion_phases():
    """Test that synthetic motion phases produce expected physical acceleration/velocities."""
    imu, gnss, gt = generate_synthetic_trajectory(seed=42, duration_sec=80.0)

    # Phase 1: 0-10s -> Acceleration phase (speed increases from 0 to 10 m/s)
    assert gt[0].speed == 0.0
    assert gt[1000].speed > 9.0  # At t=10s, speed ~ 10 m/s

    # Phase 2: 10-40s -> Straight cruise (~10 m/s)
    assert pytest.approx(gt[2000].speed, abs=0.5) == 10.0

    # Phase 5: 70-80s -> Deceleration phase (speed decreases back to 0)
    assert pytest.approx(gt[8000].speed, abs=1e-9) == 0.0
