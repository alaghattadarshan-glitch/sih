"""Unit tests for quaternion mathematics and attitude propagation."""

import pytest
import math
import numpy as np
from src.navigation.quaternion import (
    quaternion_normalize,
    quaternion_multiply,
    quaternion_conjugate,
    quaternion_to_rotation_matrix,
    euler_to_quaternion,
    quaternion_to_euler,
    quaternion_to_heading_deg,
    propagate_orientation,
)


def test_quaternion_normalization():
    """Test quaternion normalization."""
    q = np.array([2.0, 0.0, 0.0, 0.0])
    q_norm = quaternion_normalize(q)
    assert pytest.approx(q_norm[0]) == 1.0
    assert pytest.approx(np.linalg.norm(q_norm)) == 1.0


def test_quaternion_identity_multiplication():
    """Test quaternion multiplication with identity quaternion."""
    q_id = np.array([1.0, 0.0, 0.0, 0.0])
    q_v = np.array([0.7071, 0.0, 0.7071, 0.0])

    q_res = quaternion_multiply(q_v, q_id)
    assert pytest.approx(q_res[0], abs=1e-4) == 0.7071
    assert pytest.approx(q_res[2], abs=1e-4) == 0.7071


def test_zero_angular_velocity_propagation():
    """Test that zero angular velocity preserves orientation."""
    q_init = np.array([1.0, 0.0, 0.0, 0.0])
    zero_gyro = np.array([0.0, 0.0, 0.0])

    q_prop = propagate_orientation(q_init, zero_gyro, dt=0.01)
    assert pytest.approx(q_prop[0]) == 1.0
    assert pytest.approx(q_prop[1]) == 0.0
    assert pytest.approx(q_prop[2]) == 0.0
    assert pytest.approx(q_prop[3]) == 0.0


def test_known_simple_rotation():
    """Test 90-degree z-axis rotation propagation."""
    q_init = np.array([1.0, 0.0, 0.0, 0.0])  # Body X = East (Heading 90 deg)
    gyro_z = np.array([0.0, 0.0, np.pi / 2.0])  # 90 deg/s counterclockwise rotation

    # Rotate for 1 second (90 deg CCW turn -> Body X becomes North, Heading 0 deg)
    q_final = propagate_orientation(q_init, gyro_z, dt=1.0)
    heading = quaternion_to_heading_deg(q_final)

    assert pytest.approx(heading, abs=1e-4) == 0.0


def test_euler_quaternion_roundtrip():
    """Test conversion between Euler angles and quaternions."""
    roll_in, pitch_in, yaw_in = 0.1, -0.2, 0.5
    q = euler_to_quaternion(roll_in, pitch_in, yaw_in)
    roll_out, pitch_out, yaw_out = quaternion_to_euler(q)

    assert pytest.approx(roll_out, abs=1e-5) == roll_in
    assert pytest.approx(pitch_out, abs=1e-5) == pitch_in
    assert pytest.approx(yaw_out, abs=1e-5) == yaw_in
