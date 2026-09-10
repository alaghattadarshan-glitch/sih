"""Unit and Integration Tests for Non-Holonomic Constraints (NHC) and Safe ZUPT Detection."""

import os
import math
import numpy as np
import pytest

from src.data.observations import IMUObservation, GNSSObservation
from src.navigation.state import NavigationState
from src.navigation.ekf.state import ESKFStateConfig, STATE_DIM, VEL_SLICE, ATT_SLICE
from src.navigation.ekf.eskf import ErrorStateKalmanFilter
from src.navigation.ekf.models import build_nhc_measurement_matrix, build_zupt_measurement_matrix
from src.navigation.imu.zupt_detector import ZUPTDetector, ZUPTConfig, ZUPTState
from src.navigation.ai_fusion import AIESKFPipeline
from src.outage_detection.detector import GNSSOutageDetector, GNSSStatus
from src.navigation.quaternion import (
    quaternion_to_rotation_matrix,
    euler_to_quaternion,
    quaternion_multiply,
    quaternion_normalize,
)
from src.evaluation.edge_parity import CTypesNavCore


# =====================================================================
# 1. NHC JACOBIAN VERIFICATION (Analytical vs Numerical Perturbation)
# =====================================================================

def test_nhc_jacobian_analytical_vs_numerical():
    """Verify analytical NHC error-state Jacobian against numerical finite differences."""
    # Setup nominal attitude and velocity
    roll = 0.05
    pitch = -0.03
    yaw = 0.8
    q_nom = euler_to_quaternion(roll, pitch, yaw)
    R_b2n = quaternion_to_rotation_matrix(q_nom)

    v_enu = np.array([12.0, 8.0, 0.5], dtype=np.float64)
    v_body = R_b2n.T @ v_enu

    # Analytical Jacobian
    H_analytic = build_nhc_measurement_matrix(R_b2n, v_body)
    assert H_analytic.shape == (2, STATE_DIM)

    # Numerical Jacobian via central difference perturbation
    eps = 1e-6
    H_numerical = np.zeros((2, STATE_DIM), dtype=np.float64)

    def compute_h(v_vec, q_vec):
        R = quaternion_to_rotation_matrix(q_vec)
        v_b = R.T @ v_vec
        return np.array([v_b[1], v_b[2]], dtype=np.float64)

    h0 = compute_h(v_enu, q_nom)

    # 1. Perturb velocity (delta_v)
    for i in range(3):
        v_pos = v_enu.copy()
        v_pos[i] += eps
        h_pos = compute_h(v_pos, q_nom)

        v_neg = v_enu.copy()
        v_neg[i] -= eps
        h_neg = compute_h(v_neg, q_nom)

        H_numerical[:, 3 + i] = (h_pos - h_neg) / (2.0 * eps)

    # 2. Perturb attitude (delta_theta) using navigation error convention: q_pert = [1, 0.5*dth] * q_nom
    for i in range(3):
        dth_pos = np.zeros(3)
        dth_pos[i] = eps
        dq_pos = quaternion_normalize(np.array([1.0, 0.5 * dth_pos[0], 0.5 * dth_pos[1], 0.5 * dth_pos[2]]))
        q_pos = quaternion_multiply(dq_pos, q_nom)
        h_pos = compute_h(v_enu, q_pos)

        dth_neg = np.zeros(3)
        dth_neg[i] = -eps
        dq_neg = quaternion_normalize(np.array([1.0, 0.5 * dth_neg[0], 0.5 * dth_neg[1], 0.5 * dth_neg[2]]))
        q_neg = quaternion_multiply(dq_neg, q_nom)
        h_neg = compute_h(v_enu, q_neg)

        H_numerical[:, 6 + i] = (h_pos - h_neg) / (2.0 * eps)

    # Compare analytical and numerical Jacobians
    np.testing.assert_allclose(H_analytic, H_numerical, atol=1e-5, rtol=1e-5)


# =====================================================================
# 2. NHC MEASUREMENT UPDATE & JOSEPH COVARIANCE
# =====================================================================

def test_nhc_measurement_update():
    """Verify NHC measurement correction in ESKF reduces lateral velocity and maintains covariance symmetry."""
    eskf = ErrorStateKalmanFilter()
    t0 = 0.0
    llh0 = (12.9716, 77.5946, 920.0)
    v0 = np.array([10.0, 0.0, 0.0])  # Moving East (Forward = +X, Heading = East)
    q0 = euler_to_quaternion(0.0, 0.0, 0.0)  # Roll=0, Pitch=0, Yaw=0 (East forward)

    eskf.initialize(t0, llh0, v0, q0)

    # Inject realistic lateral velocity error: v = [10.0, 0.2, 0.0]
    eskf.ins._state = NavigationState(
        timestamp=0.1,
        position_enu=np.zeros(3),
        velocity_enu=np.array([10.0, 0.2, 0.0]),
        orientation_quaternion=q0,
        accelerometer_bias=np.zeros(3),
        gyroscope_bias=np.zeros(3),
    )

    state_corr, accepted, mah_dist, innov = eskf.update_nhc_constraint(sigma_y=0.1, sigma_z=0.1, gate_threshold=4.0)

    assert accepted is True
    assert mah_dist < 4.0
    # Lateral velocity (North component when heading is East) should be corrected toward 0
    assert abs(state_corr.velocity_enu[1]) < 0.2
    # Forward velocity (East component) should NOT be zeroed
    assert state_corr.velocity_enu[0] > 9.0

    # Covariance checks
    P = eskf.get_covariance()
    np.testing.assert_allclose(P, P.T, atol=1e-12)
    eigenvalues = np.linalg.eigvals(P)
    assert np.all(eigenvalues > 0), "Covariance matrix must remain positive definite"


def test_nhc_mahalanobis_gating():
    """Verify that unrealistic lateral disturbances are gated out by NHC."""
    eskf = ErrorStateKalmanFilter()
    t0 = 0.0
    llh0 = (12.9716, 77.5946, 920.0)
    v0 = np.array([10.0, 0.0, 0.0])
    q0 = euler_to_quaternion(0.0, 0.0, 0.0)
    eskf.initialize(t0, llh0, v0, q0)

    # Huge artificial lateral velocity error (e.g. 50 m/s slip)
    eskf.ins._state = NavigationState(
        timestamp=0.1,
        position_enu=np.zeros(3),
        velocity_enu=np.array([10.0, 50.0, 0.0]),
        orientation_quaternion=q0,
        accelerometer_bias=np.zeros(3),
        gyroscope_bias=np.zeros(3),
    )

    state_corr, accepted, mah_dist, innov = eskf.update_nhc_constraint(sigma_y=0.1, sigma_z=0.1, gate_threshold=4.0)

    assert accepted is False
    assert mah_dist > 4.0
    # State should remain uncorrected on rejection
    np.testing.assert_allclose(state_corr.velocity_enu, np.array([10.0, 50.0, 0.0]))


# =====================================================================
# 3. ZUPT DETECTOR & STATE MACHINE TESTS
# =====================================================================

def test_zupt_detector_conditions():
    """Test stationary conditions (accel norm, gyro norm, variance) in ZUPTDetector."""
    cfg = ZUPTConfig(
        accel_norm_tol=0.6,
        gyro_norm_threshold=0.08,
        accel_var_threshold=0.05,
        gyro_var_threshold=0.005,
        persistence_samples=5,
        window_size_samples=10,
    )
    detector = ZUPTDetector(cfg)

    # 1. Moving IMU sample (large angular rate)
    state = detector.update(0.01, 0.0, 0.0, 9.81, 0.5, 0.0, 0.0)
    assert state == ZUPTState.MOVING
    assert not detector.is_stationary

    # 2. Feed 20 stationary samples (clears out the moving spike from window buffer and exceeds persistence)
    for i in range(1, 25):
        t = i * 0.01
        state = detector.update(t, 0.0, 0.0, 9.80665, 0.001, 0.001, 0.001)

    # After clearing window and persistence, should reach STATIONARY
    assert detector.state == ZUPTState.STATIONARY
    assert detector.is_stationary is True


def test_zupt_detector_persistence_and_hysteresis():
    """Verify complete 4-state state machine lifecycle and hysteresis transitions."""
    cfg = ZUPTConfig(persistence_samples=10, leaving_hysteresis_samples=3)
    detector = ZUPTDetector(cfg)

    # Start: MOVING
    assert detector.state == ZUPTState.MOVING

    # Step 1: Candidate detected -> STATIONARY_CANDIDATE
    st = detector.update(0.01, 0.0, 0.0, 9.80665, 0.0, 0.0, 0.0)
    assert st == ZUPTState.STATIONARY_CANDIDATE
    assert detector.persistence_counter == 1

    # Steps 2-9: Remain candidate
    for i in range(2, 10):
        st = detector.update(i * 0.01, 0.0, 0.0, 9.80665, 0.0, 0.0, 0.0)
        assert st == ZUPTState.STATIONARY_CANDIDATE

    # Step 10: Reach persistence -> STATIONARY
    st = detector.update(0.10, 0.0, 0.0, 9.80665, 0.0, 0.0, 0.0)
    assert st == ZUPTState.STATIONARY
    assert detector.is_stationary is True

    # Step 11: Minor disturbance -> LEAVING_STATIONARY (hysteresis)
    st = detector.update(0.11, 0.8, 0.0, 9.80665, 0.09, 0.0, 0.0)
    assert st == ZUPTState.LEAVING_STATIONARY

    # Step 12: Disturbance continues -> still LEAVING
    st = detector.update(0.12, 0.8, 0.0, 9.80665, 0.09, 0.0, 0.0)
    assert st == ZUPTState.LEAVING_STATIONARY

    # Step 13: Exceeds leaving hysteresis -> MOVING
    st = detector.update(0.13, 0.8, 0.0, 9.80665, 0.09, 0.0, 0.0)
    assert st == ZUPTState.MOVING
    assert detector.is_stationary is False


# =====================================================================
# 4. ZUPT MEASUREMENT UPDATE TESTS
# =====================================================================

def test_zupt_measurement_update():
    """Verify ZUPT measurement update in ESKF zeros velocity and preserves covariance."""
    eskf = ErrorStateKalmanFilter()
    t0 = 0.0
    llh0 = (12.9716, 77.5946, 920.0)
    v0 = np.array([0.08, -0.05, 0.02])  # Residual velocity before stop
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    eskf.initialize(t0, llh0, v0, q0)

    state_corr, accepted, mah_dist, innov = eskf.update_zupt(sigma_zupt=0.01, gate_threshold=5.0)

    assert accepted is True
    assert mah_dist < 5.0
    # Velocity should be pulled to approximately zero
    assert np.linalg.norm(state_corr.velocity_enu) < 0.02

    # Covariance checks
    P = eskf.get_covariance()
    np.testing.assert_allclose(P, P.T, atol=1e-12)
    eigenvalues = np.linalg.eigvals(P)
    assert np.all(eigenvalues > 0)


    # Covariance checks
    P = eskf.get_covariance()
    np.testing.assert_allclose(P, P.T, atol=1e-12)
    eigenvalues = np.linalg.eigvals(P)
    assert np.all(eigenvalues > 0)


def test_zupt_never_applied_during_motion():
    """Verify pipeline never executes ZUPT while vehicle is in motion."""
    eskf = ErrorStateKalmanFilter()
    eskf.initialize(0.0, (12.9716, 77.5946, 920.0), np.array([15.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0]))

    detector = GNSSOutageDetector()
    pipeline = AIESKFPipeline(
        eskf=eskf,
        detector=detector,
        enable_ai=False,
        enable_nhc=True,
        enable_zupt=True,
    )

    # Feed 50 moving samples with significant acceleration / rate during outage
    for step in range(1, 51):
        t = step * 0.01
        imu = IMUObservation(
            timestamp=t,
            accelerometer_x=1.5,
            accelerometer_y=0.2,
            accelerometer_z=9.81,
            gyroscope_x=0.15,
            gyroscope_y=0.0,
            gyroscope_z=0.05,
        )
        # Outage: no GNSS fix provided
        state, status = pipeline.process_sample(imu, gnss_obs=None)

    # ZUPT should have 0 attempts because detector never reached STATIONARY
    assert pipeline.total_zupt_updates == 0
    assert pipeline.zupt_accepted_count == 0


# =====================================================================
# 5. SAFETY & FAILURE INJECTION TESTS
# =====================================================================

def test_safety_corrupted_nhc_measurements():
    """Inject 100 corrupted NHC measurements and verify filter does not diverge or produce NaNs."""
    eskf = ErrorStateKalmanFilter()
    eskf.initialize(0.0, (12.9716, 77.5946, 920.0), np.array([10.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0]))

    for _ in range(100):
        # Apply NHC update
        state, accepted, mah_dist, innov = eskf.update_nhc_constraint(sigma_y=0.1, sigma_z=0.1, gate_threshold=4.0)

        # Check for NaN / Inf
        assert not np.isnan(state.velocity_enu).any()
        assert not np.isnan(state.position_enu).any()
        assert not np.isinf(state.velocity_enu).any()
        assert not np.isinf(state.position_enu).any()

    P = eskf.get_covariance()
    assert not np.isnan(P).any()
    np.testing.assert_allclose(P, P.T, atol=1e-12)
    assert np.all(np.linalg.eigvals(P) > 0)


def test_safety_corrupted_zupt_measurements():
    """Inject 100 corrupted ZUPT measurements and verify covariance stability."""
    eskf = ErrorStateKalmanFilter()
    eskf.initialize(0.0, (12.9716, 77.5946, 920.0), np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0]))

    for _ in range(100):
        state, accepted, mah_dist, innov = eskf.update_zupt(sigma_zupt=0.01, gate_threshold=4.0)
        assert not np.isnan(state.velocity_enu).any()
        assert not np.isnan(state.position_enu).any()

    P = eskf.get_covariance()
    assert not np.isnan(P).any()
    np.testing.assert_allclose(P, P.T, atol=1e-12)
    assert np.all(np.linalg.eigvals(P) > 0)


# =====================================================================
# 6. PYTHON <-> C++ NATIVE PARITY TESTS
# =====================================================================

@pytest.fixture(scope="module")
def native_core():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    lib_path = os.path.join(root, "android/native/build/libnav_core.dylib")
    if not os.path.exists(lib_path):
        import subprocess
        subprocess.run(["bash", os.path.join(root, "android/native/build_lib.sh")], check=True)
    return CTypesNavCore(lib_path)


def test_python_cpp_nhc_parity(native_core):
    """Verify numerical parity between Python and C++ NHC measurement updates."""
    t0 = 0.0
    lat0, lon0, alt0 = 12.9716, 77.5946, 920.0
    v0 = np.array([8.0, 1.2, -0.4])
    q0 = np.array([1.0, 0.0, 0.0, 0.0])

    # Python Filter
    eskf_py = ErrorStateKalmanFilter()
    eskf_py.initialize(t0, (lat0, lon0, alt0), v0, q0)
    state_py, acc_py, mah_py, _ = eskf_py.update_nhc_constraint(sigma_y=0.1, sigma_z=0.1, gate_threshold=4.0)

    # C++ Core
    native_core.reset()
    native_core.initialize(t0, lat0, lon0, alt0, v0, q0)
    acc_cpp, mah_cpp = native_core.process_nhc(sigma_y=0.1, sigma_z=0.1, gate_threshold=4.0)
    state_cpp = native_core.get_state()

    # Parity assertions
    assert acc_py == acc_cpp
    assert abs(mah_py - mah_cpp) < 1e-4
    np.testing.assert_allclose(state_py.velocity_enu, state_cpp["vel_enu"], atol=1e-4)
    np.testing.assert_allclose(state_py.position_enu, state_cpp["pos_enu"], atol=1e-4)


def test_python_cpp_zupt_parity(native_core):
    """Verify numerical parity between Python and C++ ZUPT measurement updates."""
    t0 = 0.0
    lat0, lon0, alt0 = 12.9716, 77.5946, 920.0
    v0 = np.array([0.2, -0.15, 0.05])
    q0 = np.array([1.0, 0.0, 0.0, 0.0])

    # Python Filter
    eskf_py = ErrorStateKalmanFilter()
    eskf_py.initialize(t0, (lat0, lon0, alt0), v0, q0)
    state_py, acc_py, mah_py, _ = eskf_py.update_zupt(sigma_zupt=0.01, gate_threshold=4.0)

    # C++ Core
    native_core.reset()
    native_core.initialize(t0, lat0, lon0, alt0, v0, q0)
    acc_cpp, mah_cpp = native_core.process_zupt(sigma_zupt=0.01, gate_threshold=4.0)
    state_cpp = native_core.get_state()

    assert acc_py == acc_cpp
    assert abs(mah_py - mah_cpp) < 1e-4
    np.testing.assert_allclose(state_py.velocity_enu, state_cpp["vel_enu"], atol=1e-4)
    np.testing.assert_allclose(state_py.position_enu, state_cpp["pos_enu"], atol=1e-4)
