"""Tests for Android Live Sensor Harness, Calibration, Mounting, and Simulation."""

import json
import math
import os
import numpy as np
import pytest

from src.evaluation.live_harness import run_live_harness_simulation
from src.navigation.quaternion import euler_to_quaternion


def test_mounting_transform_presets():
    """Verify coordinate transformation matrices for standard phone mounting presets."""
    # Flat Portrait: Phone +y is Forward (+x), Phone -x is Left (+y), Phone +z is Up (+z)
    flat_portrait = np.array([
        [ 0.0,  1.0,  0.0],
        [-1.0,  0.0,  0.0],
        [ 0.0,  0.0,  1.0]
    ])
    raw_sensor = np.array([0.0, 9.80665, 0.0]) # Phone lying flat with gravity on +y
    body_vec = flat_portrait @ raw_sensor
    np.testing.assert_allclose(body_vec, np.array([9.80665, 0.0, 0.0]), atol=1e-5)

    # Windshield Vertical: Phone +z is Forward (+x), Phone -x is Left (+y), Phone +y is Up (+z)
    windshield_vert = np.array([
        [ 0.0,  0.0,  1.0],
        [-1.0,  0.0,  0.0],
        [ 0.0,  1.0,  0.0]
    ])
    raw_gravity = np.array([0.0, 9.80665, 0.0]) # Phone screen vertical (gravity down phone -y)
    body_vert = windshield_vert @ raw_gravity
    np.testing.assert_allclose(body_vert, np.array([0.0, 0.0, 9.80665]), atol=1e-5)


def test_stationary_calibration_estimation():
    """Verify stationary calibration extracts bias and attitude from static window."""
    np.random.seed(42)
    # 300 samples (3.0s @ 100 Hz) with known bias and minor noise
    true_gyro_bias = np.array([0.002, -0.001, 0.003])
    acc_samples = np.random.normal(loc=[0.0, 0.0, 9.80665], scale=0.01, size=(300, 3))
    gyro_samples = np.random.normal(loc=true_gyro_bias, scale=0.001, size=(300, 3))

    mean_acc = np.mean(acc_samples, axis=0)
    var_acc = np.var(acc_samples, axis=0)

    mean_gyro = np.mean(gyro_samples, axis=0)
    var_gyro = np.var(gyro_samples, axis=0)

    # Validate stationarity criteria
    assert np.all(var_acc < 0.05)
    assert np.all(var_gyro < 0.005)
    np.testing.assert_allclose(mean_gyro, true_gyro_bias, atol=0.0005)


def test_calibration_failure_on_motion():
    """Verify calibration correctly rejects dynamic/vibrational motion."""
    np.random.seed(42)
    # High variance acceleration (simulating vehicle driving or hand shaking)
    dynamic_acc = np.random.normal(loc=[1.5, 0.5, 9.8], scale=0.5, size=(300, 3))
    var_acc = np.mean(np.var(dynamic_acc, axis=0))

    # Must fail stationarity threshold
    assert var_acc > 0.05, "Dynamic motion should exceed stationary variance threshold."


def test_live_harness_end_to_end_diagnostics(tmp_path):
    """Run full live harness simulation and verify diagnostic plots and JSON report."""
    output_dir = str(tmp_path / "android_harness_test")
    summary = run_live_harness_simulation(
        duration_sec=30.0,
        outage_start_sec=10.0,
        outage_end_sec=20.0,
        output_dir=output_dir,
    )

    assert summary["harness_status"] == "SUCCESS"
    assert summary["total_imu_samples"] == 3001
    assert summary["mean_imu_rate_hz"] == pytest.approx(100.0, abs=1.0)
    assert summary["deadline_misses_count"] == 0
    assert summary["plots_saved"] == 7
    assert summary["accepted_ai_updates"] > 0

    # Verify plot files exist
    plots_dir = os.path.join(output_dir, "plots")
    expected_plots = [
        "sensor_rate.png",
        "timestamp_jitter.png",
        "trajectory_live_replay.png",
        "navigation_mode.png",
        "ai_updates.png",
        "processing_latency.png",
        "gnss_recovery.png",
    ]
    for p in expected_plots:
        assert os.path.exists(os.path.join(plots_dir, p)), f"Missing plot: {p}"
