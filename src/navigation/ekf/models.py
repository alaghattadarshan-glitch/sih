"""ESKF Dynamics, Transition, and Measurement Models.

Implements the continuous and discrete 15-state error dynamics matrices (F, G, Φ, Qd)
and the GNSS position measurement matrix (H).
"""

import numpy as np
from src.navigation.ekf.state import STATE_DIM


def skew_symmetric(v: np.ndarray) -> np.ndarray:
    """Compute 3x3 skew-symmetric matrix [v ×] from a 3-element vector v."""
    vx, vy, vz = v
    return np.array(
        [[0.0, -vz, vy], [vz, 0.0, -vx], [-vy, vx, 0.0]], dtype=np.float64
    )


def build_continuous_error_matrix(
    R_b2n: np.ndarray, f_body: np.ndarray
) -> np.ndarray:
    """Build continuous 15x15 system matrix F for error-state dynamics.

    Args:
        R_b2n (np.ndarray): 3x3 rotation matrix from Body to ENU navigation frame.
        f_body (np.ndarray): 3-element body specific force measurement vector [ax, ay, az].

    Returns:
        np.ndarray: 15x15 continuous error transition matrix F.
    """
    F = np.zeros((STATE_DIM, STATE_DIM), dtype=np.float64)

    # Specific force rotated to ENU frame
    f_enu = R_b2n @ f_body

    # δp_dot = δv
    F[0:3, 3:6] = np.eye(3)

    # δv_dot = -[f_enu ×] * δθ + R_b2n * δb_a
    F[3:6, 6:9] = -skew_symmetric(f_enu)
    F[3:6, 9:12] = R_b2n

    # δθ_dot = -R_b2n * δb_g
    F[6:9, 12:15] = -R_b2n

    return F


def build_continuous_noise_matrix(R_b2n: np.ndarray) -> np.ndarray:
    """Build continuous 15x12 noise coupling matrix G.

    Args:
        R_b2n (np.ndarray): 3x3 rotation matrix from Body to ENU navigation frame.

    Returns:
        np.ndarray: 15x12 continuous noise matrix G.
    """
    G = np.zeros((STATE_DIM, 12), dtype=np.float64)

    # Accel noise -> velocity error
    G[3:6, 0:3] = R_b2n

    # Gyro noise -> attitude error
    G[6:9, 3:6] = -R_b2n

    # Accel bias random walk
    G[9:12, 6:9] = np.eye(3)

    # Gyro bias random walk
    G[12:15, 9:12] = np.eye(3)

    return G


def discretize_dynamics(
    F: np.ndarray, G: np.ndarray, Qc: np.ndarray, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """Compute discrete transition matrix Φ and discrete process noise covariance Qd.

    Using second-order Taylor series approximation:
    Φ ≈ I + F*dt + 0.5*(F*dt)^2
    Qd ≈ G * Qc * G^T * dt

    Args:
        F (np.ndarray): 15x15 continuous system matrix.
        G (np.ndarray): 15x12 noise input matrix.
        Qc (np.ndarray): 12x12 process noise spectral density matrix.
        dt (float): Discretization time interval in seconds.

    Returns:
        tuple[np.ndarray, np.ndarray]: (Phi, Qd) discrete matrices.
    """
    I = np.eye(STATE_DIM, dtype=np.float64)
    F_dt = F * dt
    Phi = I + F_dt + 0.5 * (F_dt @ F_dt)

    GQcGt = G @ Qc @ G.T
    Qd = GQcGt * dt + 0.5 * (F @ GQcGt + GQcGt @ F.T) * (dt**2)
    Qd = 0.5 * (Qd + Qd.T)  # Force symmetry

    return Phi, Qd


def build_gnss_position_measurement_matrix() -> np.ndarray:
    """Build 3x15 measurement matrix H for GNSS ENU position measurements.

    z = p_GNSS_ENU = H * δx + v
    H = [I_3 | 0_3x12]
    """
    H = np.zeros((3, STATE_DIM), dtype=np.float64)
    H[0:3, 0:3] = np.eye(3)
    return H


def build_nhc_measurement_matrix(
    R_b2n: np.ndarray, v_body: np.ndarray
) -> np.ndarray:
    """Build 2x15 measurement matrix H_nhc for Non-Holonomic Constraints (NHC).

    Constrains vehicle body lateral (y) and vertical (z) velocities:
    z_nhc = [0, 0]^T
    h(x) = [v_body_y, v_body_z]^T

    Linearized error model:
    δv_body = R_b2n^T * δv + [v_body ×] * δθ

    Args:
        R_b2n (np.ndarray): 3x3 Body to ENU rotation matrix.
        v_body (np.ndarray): 3-element body velocity vector [vx_b, vy_b, vz_b].

    Returns:
        np.ndarray: 2x15 measurement matrix H_nhc.
    """
    H = np.zeros((2, STATE_DIM), dtype=np.float64)
    vx_b, vy_b, vz_b = v_body

    # Row 0: Body Y (Lateral) Velocity Constraint
    # d(vy_b)/d(delta_v) = R_b2n[:, 1]^T (second column of R_b2n)
    H[0, 3:6] = R_b2n[:, 1]
    # d(vy_b)/d(delta_theta_nav) = [vz_b, 0, -vx_b] @ R_b2n.T
    H[0, 6:9] = np.array([vz_b, 0.0, -vx_b], dtype=np.float64) @ R_b2n.T

    # Row 1: Body Z (Vertical) Velocity Constraint
    # d(vz_b)/d(delta_v) = R_b2n[:, 2]^T (third column of R_b2n)
    H[1, 3:6] = R_b2n[:, 2]
    # d(vz_b)/d(delta_theta_nav) = [-vy_b, vx_b, 0] @ R_b2n.T
    H[1, 6:9] = np.array([-vy_b, vx_b, 0.0], dtype=np.float64) @ R_b2n.T

    return H


def build_zupt_measurement_matrix() -> np.ndarray:
    """Build 3x15 measurement matrix H_zupt for Zero-Velocity Update (ZUPT).

    Constrains 3D ENU velocity to zero:
    z_zupt = [0, 0, 0]^T
    h(x) = v_enu

    Returns:
        np.ndarray: 3x15 measurement matrix H_zupt.
    """
    H = np.zeros((3, STATE_DIM), dtype=np.float64)
    H[0:3, 3:6] = np.eye(3)
    return H

