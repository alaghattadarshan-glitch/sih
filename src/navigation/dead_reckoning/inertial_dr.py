"""Vehicle Inertial Dead-Reckoning Navigation System Baseline.

Provides a vehicle-oriented dead-reckoning interface around the Strapdown INS engine
to process IMU streams during GNSS outages.
"""

from typing import List, Tuple, Optional
import numpy as np

from src.data.observations import IMUObservation
from src.navigation.state import NavigationState
from src.navigation.ins.strapdown import StrapdownINS


class InertialDeadReckoning:
    """Vehicle Inertial Dead-Reckoning baseline system."""

    def __init__(self, g_val: float = 9.80665):
        """Initialize Dead-Reckoning system.

        Args:
            g_val (float): Gravitational acceleration magnitude in m/s^2.
        """
        self.ins = StrapdownINS(g_val=g_val)
        self._initial_time: float = 0.0

    def initialize(
        self,
        initial_time: float,
        initial_llh: Tuple[float, float, float],
        initial_velocity_enu: np.ndarray,
        initial_quaternion: np.ndarray,
        accel_bias: Optional[np.ndarray] = None,
        gyro_bias: Optional[np.ndarray] = None,
    ):
        """Initialize dead reckoning baseline state."""
        self._initial_time = initial_time
        self.ins.initialize(
            initial_time=initial_time,
            initial_llh=initial_llh,
            initial_velocity_enu=initial_velocity_enu,
            initial_quaternion=initial_quaternion,
            accel_bias=accel_bias,
            gyro_bias=gyro_bias,
        )

    def process_imu_sample(self, imu_obs: IMUObservation) -> NavigationState:
        """Process a single IMU observation sample."""
        return self.ins.update(imu_obs)

    def process_imu_stream(self, imu_list: List[IMUObservation]) -> List[NavigationState]:
        """Process a continuous stream of IMU observations.

        Args:
            imu_list (List[IMUObservation]): High-rate IMU observation stream.

        Returns:
            List[NavigationState]: Sequence of dead-reckoned navigation states.
        """
        states: List[NavigationState] = []
        for obs in imu_list:
            state = self.process_imu_sample(obs)
            states.append(state)
        return states

    @property
    def current_state(self) -> Optional[NavigationState]:
        """Return current navigation state."""
        return self.ins.get_state()

    @property
    def elapsed_time(self) -> float:
        """Elapsed navigation time in seconds."""
        state = self.ins.get_state()
        return state.timestamp - self._initial_time if state else 0.0

    @property
    def position_enu(self) -> Optional[np.ndarray]:
        """Current position [East, North, Up] in meters."""
        state = self.ins.get_state()
        return state.position_enu if state else None

    @property
    def velocity_enu(self) -> Optional[np.ndarray]:
        """Current velocity [V_E, V_N, V_U] in m/s."""
        state = self.ins.get_state()
        return state.velocity_enu if state else None

    @property
    def heading_deg(self) -> Optional[float]:
        """Current geographic heading in degrees [0°, 360°)."""
        state = self.ins.get_state()
        return state.heading_deg() if state else None

    @property
    def orientation_quaternion(self) -> Optional[np.ndarray]:
        """Current attitude quaternion [q_w, q_x, q_y, q_z]."""
        state = self.ins.get_state()
        return state.orientation_quaternion if state else None
