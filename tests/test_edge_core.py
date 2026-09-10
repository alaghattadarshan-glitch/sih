"""Tests for C++ Native Navigation Core via ctypes."""

import os
import time
import numpy as np
import pytest

from src.evaluation.edge_parity import CTypesNavCore


@pytest.fixture(scope="module")
def native_core():
    """Build library if necessary and return CTypesNavCore instance."""
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    lib_path = os.path.join(root, "android/native/build/libnav_core.dylib")
    if not os.path.exists(lib_path):
        import subprocess
        subprocess.run(["bash", os.path.join(root, "android/native/build_lib.sh")], check=True)

    return CTypesNavCore(lib_path)


def test_native_core_lifecycle_and_reset(native_core):
    """Test native initialization, state querying, and reset."""
    t0 = 1000.0
    lat0, lon0, alt0 = 12.9716, 77.5946, 920.0
    v0 = np.array([5.0, 10.0, 0.0])
    q0 = np.array([1.0, 0.0, 0.0, 0.0])

    native_core.initialize(t0, lat0, lon0, alt0, v0, q0)
    state = native_core.get_state()

    assert state["timestamp"] == t0
    np.testing.assert_allclose(state["pos_enu"], np.zeros(3), atol=1e-5)
    np.testing.assert_allclose(state["vel_enu"], v0, atol=1e-5)
    assert state["status"] == 0  # NAV_STATUS_GOOD

    native_core.reset()


def test_native_imu_propagation(native_core):
    """Test 100 Hz IMU propagation steps."""
    t0 = 0.0
    lat0, lon0, alt0 = 12.9716, 77.5946, 920.0
    v0 = np.array([0.0, 0.0, 0.0])
    q0 = np.array([1.0, 0.0, 0.0, 0.0])

    native_core.initialize(t0, lat0, lon0, alt0, v0, q0)

    # 1.0 second of stationary 100 Hz IMU (accel_z = 9.80665 m/s^2)
    for step in range(1, 101):
        t = step * 0.01
        native_core.process_imu(t, 0.0, 0.0, 9.80665, 0.0, 0.0, 0.0)

    state = native_core.get_state()
    assert state["timestamp"] == pytest.approx(1.0, abs=1e-6)
    # Stationary: position and velocity should remain approximately zero
    np.testing.assert_allclose(state["vel_enu"], np.zeros(3), atol=0.01)
    np.testing.assert_allclose(state["pos_enu"], np.zeros(3), atol=0.01)


def test_native_gnss_update_correction(native_core):
    """Test GNSS position measurement correction in C++ ESKF."""
    t0 = 0.0
    lat0, lon0, alt0 = 12.9716, 77.5946, 920.0
    v0 = np.array([10.0, 0.0, 0.0])
    q0 = np.array([1.0, 0.0, 0.0, 0.0])

    native_core.initialize(t0, lat0, lon0, alt0, v0, q0)

    # Propagate 1 second with minor bias
    for step in range(1, 101):
        t = step * 0.01
        native_core.process_imu(t, 0.1, 0.0, 9.80665, 0.0, 0.0, 0.0)

    # Inject GNSS fix at t = 1.0s with known coordinates
    native_core.process_gnss(1.0, lat0, lon0, alt0, h_acc=1.0)
    state = native_core.get_state()

    # Position should be corrected toward origin
    assert np.linalg.norm(state["pos_enu"]) < 10.0


def test_native_ai_displacement_gating(native_core):
    """Test AI pseudo-measurement update and Mahalanobis gating in C++."""
    t0 = 0.0
    lat0, lon0, alt0 = 12.9716, 77.5946, 920.0
    v0 = np.array([0.0, 0.0, 0.0])
    q0 = np.array([1.0, 0.0, 0.0, 0.0])

    native_core.initialize(t0, lat0, lon0, alt0, v0, q0)

    # 1. Consistent AI displacement update (should be ACCEPTED)
    ref_pos = np.array([0.0, 0.0, 0.0])
    delta_ai_valid = np.array([0.5, 0.2, 0.0])
    accepted, mahalanobis = native_core.process_ai_displacement(ref_pos, delta_ai_valid)

    assert accepted is True
    assert mahalanobis < 4.0

    # 2. Extreme Outlier AI displacement (should be REJECTED by gating)
    delta_ai_outlier = np.array([500.0, 500.0, 100.0])
    accepted_outlier, mahalanobis_outlier = native_core.process_ai_displacement(ref_pos, delta_ai_outlier)

    assert accepted_outlier is False
    assert mahalanobis_outlier > 4.0


def test_native_outage_state_machine(native_core):
    """Test GNSS outage and recovery state transitions in C++ core."""
    t0 = 0.0
    lat0, lon0, alt0 = 12.9716, 77.5946, 920.0
    v0 = np.array([0.0, 0.0, 0.0])
    q0 = np.array([1.0, 0.0, 0.0, 0.0])

    native_core.initialize(t0, lat0, lon0, alt0, v0, q0)
    assert native_core.get_state()["status"] == 0  # GOOD

    # Simulate GPS gap > 2.0s
    for step in range(1, 301):
        t = step * 0.01
        native_core.process_imu(t, 0.0, 0.0, 9.80665, 0.0, 0.0, 0.0)

    # Missing fix -> OUTAGE (2)
    native_core.process_gnss(3.0, 0.0, 0.0, 0.0)
    native_core.process_gnss(3.1, 0.0, 0.0, 0.0)
    assert native_core.get_state()["status"] == 2  # OUTAGE

    # Recovery: 2 consecutive valid fixes trigger transition OUTAGE -> RECOVERING (3)
    native_core.process_gnss(4.0, lat0, lon0, alt0, h_acc=2.0)
    native_core.process_gnss(4.1, lat0, lon0, alt0, h_acc=2.0)
    assert native_core.get_state()["status"] == 3  # RECOVERING

    # Confirmed Recovery: 2 consecutive valid fixes while in RECOVERING -> GOOD (0)
    native_core.process_gnss(4.2, lat0, lon0, alt0, h_acc=2.0)
    native_core.process_gnss(4.3, lat0, lon0, alt0, h_acc=2.0)
    assert native_core.get_state()["status"] == 0  # GOOD


def test_native_imu_step_latency_benchmark(native_core):
    """Benchmark C++ 100 Hz step latency: must be well below 10 ms budget."""
    t0 = 0.0
    lat0, lon0, alt0 = 12.9716, 77.5946, 920.0
    v0 = np.array([15.0, 0.0, 0.0])
    q0 = np.array([1.0, 0.0, 0.0, 0.0])

    native_core.initialize(t0, lat0, lon0, alt0, v0, q0)

    num_samples = 1000
    times_us = []

    for i in range(num_samples):
        t = (i + 1) * 0.01
        t_start = time.perf_counter_ns()
        native_core.process_imu(t, 0.05, -0.02, 9.80665, 0.001, -0.001, 0.002)
        t_end = time.perf_counter_ns()
        times_us.append((t_end - t_start) / 1000.0)

    mean_us = np.mean(times_us)
    p95_us = np.percentile(times_us, 95)
    max_us = np.max(times_us)

    # 10 ms budget = 10,000 microseconds
    assert mean_us < 1000.0, f"Mean latency {mean_us:.2f} us exceeded 1ms target."
    assert p95_us < 2000.0, f"P95 latency {p95_us:.2f} us exceeded 2ms target."
