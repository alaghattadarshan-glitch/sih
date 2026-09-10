"""Quaternion Mathematics & Attitude Kinematics Module.

Implements unit quaternion algebra, orientation propagation from gyroscope angular rates,
body-to-ENU coordinate rotations, and Euler angle conversions.

Convention:
- Quaternion Format: [q_w, q_x, q_y, q_z] (Scalar first, unit magnitude).
- Frame Rotation Convention: Body frame -> Local ENU Navigation frame (R_b2n).
- Vector Transformation: v_enu = R_b2n * v_body.
- Heading Definition: 0° = North, 90° = East, 180° = South, 270° = West [0°, 360°).
"""

import math
import numpy as np
from typing import Tuple


def quaternion_normalize(q: np.ndarray) -> np.ndarray:
    """Normalize quaternion [q_w, q_x, q_y, q_z] to unit magnitude."""
    q = np.asarray(q, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / norm


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Compute quaternion multiplication q1 * q2."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return quaternion_normalize(np.array([w, x, y, z], dtype=np.float64))


def quaternion_conjugate(q: np.ndarray) -> np.ndarray:
    """Compute quaternion conjugate q* = [q_w, -q_x, -q_y, -q_z]."""
    w, x, y, z = q
    return np.array([w, -x, -y, -z], dtype=np.float64)


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [q_w, q_x, q_y, q_z] to 3x3 Body->ENU rotation matrix R_b2n."""
    qw, qx, qy, qz = quaternion_normalize(q)
    return np.array(
        [
            [
                1.0 - 2.0 * (qy**2 + qz**2),
                2.0 * (qx * qy - qz * qw),
                2.0 * (qx * qz + qy * qw),
            ],
            [
                2.0 * (qx * qy + qz * qw),
                1.0 - 2.0 * (qx**2 + qz**2),
                2.0 * (qy * qz - qx * qw),
            ],
            [
                2.0 * (qx * qz - qy * qw),
                2.0 * (qy * qz + qx * qw),
                1.0 - 2.0 * (qx**2 + qy**2),
            ],
        ],
        dtype=np.float64,
    )


def euler_to_quaternion(roll_rad: float, pitch_rad: float, yaw_rad: float) -> np.ndarray:
    """Convert Euler angles (roll phi, pitch theta, yaw psi in radians) to quaternion.

    Z-Y-X Tait-Bryan convention: Yaw around Z, Pitch around Y, Roll around X.
    Yaw = 0 rad -> East, pi/2 rad -> North in ENU math convention.
    """
    cr = math.cos(roll_rad * 0.5)
    sr = math.sin(roll_rad * 0.5)
    cp = math.cos(pitch_rad * 0.5)
    sp = math.sin(pitch_rad * 0.5)
    cy = math.cos(yaw_rad * 0.5)
    sy = math.sin(yaw_rad * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return quaternion_normalize(np.array([qw, qx, qy, qz], dtype=np.float64))


def quaternion_to_euler(q: np.ndarray) -> Tuple[float, float, float]:
    """Convert quaternion [q_w, q_x, q_y, q_z] to Euler angles (roll, pitch, yaw) in radians."""
    qw, qx, qy, qz = quaternion_normalize(q)

    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (qw * qy - qz * qx)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def quaternion_to_heading_deg(q: np.ndarray) -> float:
    """Convert quaternion to Geographic azimuth heading in degrees [0°, 360°).

    0° = North, 90° = East, 180° = South, 270° = West.
    """
    R = quaternion_to_rotation_matrix(q)
    e = R[0, 0]
    n = R[1, 0]
    yaw_rad = math.atan2(n, e)
    heading_deg = math.degrees(math.pi / 2.0 - yaw_rad) % 360.0
    return float(heading_deg)


def heading_deg_to_yaw_rad(heading_deg: float) -> float:
    """Convert Geographic heading (0° = North, 90° = East) to math yaw angle in radians (0 = East, pi/2 = North)."""
    heading_rad = math.radians(heading_deg % 360.0)
    return (math.pi / 2.0 - heading_rad) % (2.0 * math.pi)


def gyro_to_quaternion_delta(gyro: np.ndarray, dt: float) -> np.ndarray:
    """Compute relative orientation increment quaternion delta_q from gyro rate vector over dt."""
    gx, gy, gz = gyro
    omega_mag = math.sqrt(gx**2 + gy**2 + gz**2)
    angle = omega_mag * dt

    if angle < 1e-12:
        dq = np.array(
            [1.0, 0.5 * gx * dt, 0.5 * gy * dt, 0.5 * gz * dt], dtype=np.float64
        )
    else:
        axis = np.array([gx, gy, gz], dtype=np.float64) / omega_mag
        sin_half = math.sin(angle * 0.5)
        dq = np.array(
            [
                math.cos(angle * 0.5),
                axis[0] * sin_half,
                axis[1] * sin_half,
                axis[2] * sin_half,
            ],
            dtype=np.float64,
        )
    return quaternion_normalize(dq)


def propagate_orientation(q_current: np.ndarray, gyro: np.ndarray, dt: float) -> np.ndarray:
    """Propagate body-to-ENU attitude quaternion given gyro measurement vector over dt."""
    dq = gyro_to_quaternion_delta(gyro, dt)
    return quaternion_multiply(q_current, dq)


def stationary_attitude_initialization(
    accel_static: np.ndarray, initial_heading_deg: float = 0.0
) -> np.ndarray:
    """Estimate initial attitude quaternion from stationary accelerometer reading and initial heading.

    Roll and pitch are derived from gravity alignment. Yaw is set from initial_heading_deg.

    Args:
        accel_static (np.ndarray): 3-element static accelerometer reading [a_x, a_y, a_z].
        initial_heading_deg (float): Geographic initial heading in degrees (0° = North, 90° = East).

    Returns:
        np.ndarray: Initial unit quaternion [q_w, q_x, q_y, q_z].
    """
    ax, ay, az = accel_static
    roll = math.atan2(-ay, az)
    pitch = math.atan2(ax, math.sqrt(ay**2 + az**2))
    yaw = heading_deg_to_yaw_rad(initial_heading_deg)

    return euler_to_quaternion(roll, pitch, yaw)
