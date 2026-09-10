"""Navigation State Representation.

Encapsulates the navigation state vector (Position ENU, Velocity ENU, Attitude Quaternion,
Accelerometer Bias, Gyroscope Bias) and epoch timestamp in local Cartesian space.

Units Convention:
- Position ENU: meters [East, North, Up]
- Velocity ENU: m/s [V_East, V_North, V_Up]
- Attitude Quaternion: unitless [q_w, q_x, q_y, q_z] representing Body -> ENU rotation
- Accel Bias: m/s^2 [b_ax, b_ay, b_az]
- Gyro Bias: rad/s [b_gx, b_gy, b_gz]
- Timestamp: seconds
"""

from dataclasses import dataclass, field
import math
import numpy as np


@dataclass
class NavigationState:
    """Represents the instantaneous navigation state vector."""

    timestamp: float
    position_enu: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    velocity_enu: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    orientation_quaternion: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    )
    accelerometer_bias: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    gyroscope_bias: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))

    def __post_init__(self):
        """Ensure numpy arrays and quaternion normalization."""
        self.position_enu = np.asarray(self.position_enu, dtype=np.float64).reshape(3)
        self.velocity_enu = np.asarray(self.velocity_enu, dtype=np.float64).reshape(3)
        self.orientation_quaternion = np.asarray(
            self.orientation_quaternion, dtype=np.float64
        ).reshape(4)
        self.accelerometer_bias = np.asarray(
            self.accelerometer_bias, dtype=np.float64
        ).reshape(3)
        self.gyroscope_bias = np.asarray(
            self.gyroscope_bias, dtype=np.float64
        ).reshape(3)

        # Normalize orientation quaternion
        q_norm = np.linalg.norm(self.orientation_quaternion)
        if q_norm > 1e-12:
            self.orientation_quaternion /= q_norm
        else:
            self.orientation_quaternion = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    @property
    def speed(self) -> float:
        """Ground speed magnitude in m/s."""
        return float(np.linalg.norm(self.velocity_enu[:2]))

    @property
    def total_speed(self) -> float:
        """3D velocity magnitude in m/s."""
        return float(np.linalg.norm(self.velocity_enu))

    @property
    def rotation_matrix(self) -> np.ndarray:
        """Compute 3x3 rotation matrix R_b2n (Body to ENU navigation frame)."""
        qw, qx, qy, qz = self.orientation_quaternion
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

    def heading_deg(self) -> float:
        """Calculate geographic heading in degrees [0°, 360°), where 0° = North, 90° = East.

        Derived from the body X-axis projection in the ENU navigation frame.
        Body X-axis in ENU: v_bX = R_b2n * [1, 0, 0]^T = [R11, R21, R31]^T = [East, North, Up]
        """
        R = self.rotation_matrix
        e = R[0, 0]  # East component of body X axis
        n = R[1, 0]  # North component of body X axis

        # Math yaw angle in ENU (0 = East, pi/2 = North): atan2(North, East)
        yaw_rad = math.atan2(n, e)

        # Convert to Geographic azimuth (0 = North, 90 = East): azimuth = 90 deg - yaw_rad
        heading_deg = math.degrees(math.pi / 2.0 - yaw_rad) % 360.0
        return float(heading_deg)

    def copy(self) -> "NavigationState":
        """Create a deep copy of the navigation state."""
        return NavigationState(
            timestamp=self.timestamp,
            position_enu=self.position_enu.copy(),
            velocity_enu=self.velocity_enu.copy(),
            orientation_quaternion=self.orientation_quaternion.copy(),
            accelerometer_bias=self.accelerometer_bias.copy(),
            gyroscope_bias=self.gyroscope_bias.copy(),
        )
