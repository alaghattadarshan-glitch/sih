"""Strapdown Inertial Navigation System (S-INS) Mechanization Engine.

Integrates high-rate IMU accelerometer and gyroscope observations to continuously
propagate the 3D attitude, velocity, and position in local ENU Cartesian coordinates.
"""

from typing import Tuple, Optional, List
import numpy as np

from src.data.observations import IMUObservation
from src.navigation.state.navigation_state import NavigationState
from src.navigation.quaternion import (
    quaternion_normalize,
    quaternion_to_rotation_matrix,
    propagate_orientation,
)
from src.coordinate_transforms import LocalFrame


class StrapdownINS:
    """Strapdown Inertial Navigation System engine."""

    def __init__(self, g_val: float = 9.80665):
        """Initialize Strapdown INS engine.

        Args:
            g_val (float): Gravitational acceleration magnitude in m/s^2. Default 9.80665 m/s^2.
        """
        self.g_val = g_val
        self._state: Optional[NavigationState] = None
        self._local_frame: Optional[LocalFrame] = None
        self._prev_a_enu: Optional[np.ndarray] = None
        self._is_initialized = False

    @property
    def is_initialized(self) -> bool:
        """Return True if INS state has been initialized."""
        return self._is_initialized

    @property
    def local_frame(self) -> Optional[LocalFrame]:
        """Return reference LocalFrame origin object."""
        return self._local_frame

    def initialize(
        self,
        initial_time: float,
        initial_llh: Tuple[float, float, float],
        initial_velocity_enu: np.ndarray,
        initial_quaternion: np.ndarray,
        accel_bias: Optional[np.ndarray] = None,
        gyro_bias: Optional[np.ndarray] = None,
    ):
        """Initialize navigation state and reference origin.

        Args:
            initial_time (float): Initial timestamp in seconds.
            initial_llh (Tuple[float, float, float]): Origin (lat_deg, lon_deg, h_m).
            initial_velocity_enu (np.ndarray): Initial velocity vector [V_E, V_N, V_U] in m/s.
            initial_quaternion (np.ndarray): Initial attitude quaternion [q_w, q_x, q_y, q_z].
            accel_bias (Optional[np.ndarray]): Estimated accelerometer bias [b_ax, b_ay, b_az] in m/s^2.
            gyro_bias (Optional[np.ndarray]): Estimated gyroscope bias [b_gx, b_gy, b_gz] in rad/s.
        """
        lat, lon, alt = initial_llh
        self._local_frame = LocalFrame(lat, lon, alt)

        b_a = np.zeros(3, dtype=np.float64) if accel_bias is None else np.asarray(accel_bias, dtype=np.float64)
        b_g = np.zeros(3, dtype=np.float64) if gyro_bias is None else np.asarray(gyro_bias, dtype=np.float64)

        self._state = NavigationState(
            timestamp=float(initial_time),
            position_enu=np.zeros(3, dtype=np.float64),  # Starts at (0, 0, 0) relative to origin
            velocity_enu=np.asarray(initial_velocity_enu, dtype=np.float64).reshape(3),
            orientation_quaternion=quaternion_normalize(initial_quaternion),
            accelerometer_bias=b_a,
            gyroscope_bias=b_g,
        )

        # Initial acceleration in ENU (assume static balance initially)
        self._prev_a_enu = np.zeros(3, dtype=np.float64)
        self._is_initialized = True

    def update(self, imu_obs: IMUObservation) -> NavigationState:
        """Step the Strapdown INS mechanization equations for a single IMU observation.

        Mechanization steps:
        1. Calculate dt = t_curr - t_prev.
        2. Correct IMU measurements using estimated biases.
        3. Propagate attitude quaternion using gyro rates.
        4. Transform body specific force into ENU navigation frame.
        5. Compensate gravity in ENU frame: a_enu = R_b2n * f_body - [0, 0, g]^T.
        6. Perform trapezoidal velocity integration: v_new = v_old + 0.5*(a_old + a_new)*dt.
        7. Perform trapezoidal position integration: p_new = p_old + 0.5*(v_old + v_new)*dt.
        8. Update state timestamp and parameters.

        Args:
            imu_obs (IMUObservation): Input IMU observation.

        Returns:
            NavigationState: Updated navigation state instance.

        Raises:
            RuntimeError: If INS engine has not been initialized.
            ValueError: If dt is <= 0 or invalid.
        """
        if not self._is_initialized or self._state is None:
            raise RuntimeError("StrapdownINS engine must be initialized before calling update().")

        dt = imu_obs.timestamp - self._state.timestamp
        if dt <= 0:
            if dt == 0:
                return self._state.copy()
            raise ValueError(f"Non-positive dt in INS update: {dt} (t_obs={imu_obs.timestamp}, t_state={self._state.timestamp})")

        if dt > 1.0:
            raise ValueError(f"Excessively large timestamp step in INS update: dt = {dt}s")

        # 2. Bias correction
        raw_accel = np.array(
            [imu_obs.accelerometer_x, imu_obs.accelerometer_y, imu_obs.accelerometer_z],
            dtype=np.float64,
        )
        raw_gyro = np.array(
            [imu_obs.gyroscope_x, imu_obs.gyroscope_y, imu_obs.gyroscope_z],
            dtype=np.float64,
        )

        f_body = raw_accel - self._state.accelerometer_bias
        omega_body = raw_gyro - self._state.gyroscope_bias

        # 3. Propagate orientation
        q_new = propagate_orientation(self._state.orientation_quaternion, omega_body, dt)
        R_b2n = quaternion_to_rotation_matrix(q_new)

        # 4. Transform body specific force into ENU frame
        f_enu = R_b2n @ f_body

        # 5. Compensate gravity in ENU: a_enu = f_enu - [0, 0, g]
        a_enu = f_enu - np.array([0.0, 0.0, self.g_val], dtype=np.float64)

        # 6. Trapezoidal velocity integration
        v_old = self._state.velocity_enu
        a_avg = 0.5 * (self._prev_a_enu + a_enu)
        v_new = v_old + a_avg * dt

        # 7. Trapezoidal position integration
        p_old = self._state.position_enu
        v_avg = 0.5 * (v_old + v_new)
        p_new = p_old + v_avg * dt

        # 8. Store updated state
        self._prev_a_enu = a_enu
        self._state = NavigationState(
            timestamp=imu_obs.timestamp,
            position_enu=p_new,
            velocity_enu=v_new,
            orientation_quaternion=q_new,
            accelerometer_bias=self._state.accelerometer_bias.copy(),
            gyroscope_bias=self._state.gyroscope_bias.copy(),
        )

        return self._state.copy()

    def get_state(self) -> Optional[NavigationState]:
        """Return current navigation state."""
        return self._state.copy() if self._state else None

    def reset(self):
        """Reset INS state."""
        self._state = None
        self._local_frame = None
        self._prev_a_enu = None
        self._is_initialized = False
