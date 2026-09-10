"""Safe Zero-Velocity Update (ZUPT) Detector.

Implements a multi-condition stationary detector for vehicle IMU streams with:
- Accelerometer norm gravity matching
- Gyroscope norm thresholding
- Short-window acceleration & angular rate variance estimation
- Configurable persistence duration (counter-based)
- 4-state finite state machine (MOVING, STATIONARY_CANDIDATE, STATIONARY, LEAVING_STATIONARY)
- Hysteresis to eliminate rapid false-positive toggling.
"""

from enum import IntEnum
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import math
import collections
import numpy as np

from src.data.observations import IMUObservation


class ZUPTState(IntEnum):
    """ZUPT Finite State Machine enumeration."""
    MOVING = 0
    STATIONARY_CANDIDATE = 1
    STATIONARY = 2
    LEAVING_STATIONARY = 3


@dataclass
class ZUPTConfig:
    """Configuration parameters for ZUPT detection."""
    accel_norm_tol: float = 0.6  # |norm(a) - g| in m/s^2
    gyro_norm_threshold: float = 0.08  # norm(omega) in rad/s
    accel_var_threshold: float = 0.05  # var(norm(a)) in m^2/s^4
    gyro_var_threshold: float = 0.005  # var(norm(omega)) in rad^2/s^2
    persistence_samples: int = 30  # consecutive candidate samples (~0.3s at 100Hz)
    window_size_samples: int = 15  # sliding window size for variance calculation
    g_val: float = 9.80665  # local gravity magnitude
    leaving_hysteresis_samples: int = 3  # samples before fully confirming departure from stationary


class ZUPTDetector:
    """Multi-condition stationary detector with persistence and hysteresis."""

    def __init__(self, config: Optional[ZUPTConfig] = None):
        """Initialize ZUPT detector.

        Args:
            config (Optional[ZUPTConfig]): Detector thresholds and persistence settings.
        """
        self.config = config if config is not None else ZUPTConfig()
        self.state: ZUPTState = ZUPTState.MOVING
        self.persistence_counter: int = 0
        self.leaving_counter: int = 0

        # Ring buffers for sliding window variance
        self._accel_history: collections.deque = collections.deque(maxlen=self.config.window_size_samples)
        self._gyro_history: collections.deque = collections.deque(maxlen=self.config.window_size_samples)

        # Diagnostics for last processed sample
        self.last_accel_norm: float = 0.0
        self.last_gyro_norm: float = 0.0
        self.last_accel_var: float = 0.0
        self.last_gyro_var: float = 0.0
        self.last_is_candidate: bool = False

    @property
    def is_stationary(self) -> bool:
        """Return True ONLY if confirmed stationary (state == STATIONARY)."""
        return self.state == ZUPTState.STATIONARY

    def update(
        self,
        timestamp: float,
        ax: float,
        ay: float,
        az: float,
        gx: float,
        gy: float,
        gz: float,
    ) -> ZUPTState:
        """Process IMU sample and update stationary state machine.

        Args:
            timestamp (float): Sample timestamp in seconds.
            ax, ay, az (float): Accelerometer readings in m/s^2.
            gx, gy, gz (float): Gyroscope readings in rad/s.

        Returns:
            ZUPTState: Current updated state.
        """
        a_norm = math.sqrt(ax * ax + ay * ay + az * az)
        g_norm = math.sqrt(gx * gx + gy * gy + gz * gz)

        self._accel_history.append(a_norm)
        self._gyro_history.append(g_norm)

        self.last_accel_norm = a_norm
        self.last_gyro_norm = g_norm

        # Compute sliding variances if buffer has sufficient samples
        if len(self._accel_history) >= 3:
            acc_arr = np.array(self._accel_history, dtype=np.float64)
            gyro_arr = np.array(self._gyro_history, dtype=np.float64)
            self.last_accel_var = float(np.var(acc_arr))
            self.last_gyro_var = float(np.var(gyro_arr))
        else:
            self.last_accel_var = 0.0
            self.last_gyro_var = 0.0

        # Evaluate candidate criteria
        gravity_diff = abs(a_norm - self.config.g_val)
        c_accel_norm = gravity_diff <= self.config.accel_norm_tol
        c_gyro_norm = g_norm <= self.config.gyro_norm_threshold
        c_accel_var = self.last_accel_var <= self.config.accel_var_threshold
        c_gyro_var = self.last_gyro_var <= self.config.gyro_var_threshold

        is_candidate = c_accel_norm and c_gyro_norm and c_accel_var and c_gyro_var
        self.last_is_candidate = is_candidate

        # State Machine Transitions with Hysteresis
        if self.state == ZUPTState.MOVING:
            if is_candidate:
                self.state = ZUPTState.STATIONARY_CANDIDATE
                self.persistence_counter = 1
            else:
                self.persistence_counter = 0

        elif self.state == ZUPTState.STATIONARY_CANDIDATE:
            if is_candidate:
                self.persistence_counter += 1
                if self.persistence_counter >= self.config.persistence_samples:
                    self.state = ZUPTState.STATIONARY
            else:
                # Immediate drop back to MOVING if candidate condition broken
                self.state = ZUPTState.MOVING
                self.persistence_counter = 0

        elif self.state == ZUPTState.STATIONARY:
            if is_candidate:
                # Maintain stationary state
                self.leaving_counter = 0
            else:
                # Condition violated: enter LEAVING_STATIONARY
                # Severe spike (>2x tolerance) jumps directly to MOVING
                severe_accel = gravity_diff > (2.0 * self.config.accel_norm_tol)
                severe_gyro = g_norm > (2.0 * self.config.gyro_norm_threshold)
                if severe_accel or severe_gyro:
                    self.state = ZUPTState.MOVING
                    self.persistence_counter = 0
                    self.leaving_counter = 0
                else:
                    self.state = ZUPTState.LEAVING_STATIONARY
                    self.leaving_counter = 1

        elif self.state == ZUPTState.LEAVING_STATIONARY:
            if is_candidate:
                # Brief glitch: recover to STATIONARY
                self.state = ZUPTState.STATIONARY
                self.leaving_counter = 0
            else:
                self.leaving_counter += 1
                if self.leaving_counter >= self.config.leaving_hysteresis_samples:
                    self.state = ZUPTState.MOVING
                    self.persistence_counter = 0
                    self.leaving_counter = 0

        return self.state

    def update_obs(self, obs: IMUObservation) -> ZUPTState:
        """Helper to process an IMUObservation object."""
        return self.update(
            timestamp=obs.timestamp,
            ax=obs.accelerometer_x,
            ay=obs.accelerometer_y,
            az=obs.accelerometer_z,
            gx=obs.gyroscope_x,
            gy=obs.gyroscope_y,
            gz=obs.gyroscope_z,
        )

    def get_diagnostics(self) -> Dict[str, Any]:
        """Return diagnostic metrics dictionary."""
        return {
            "state": self.state.name,
            "state_value": int(self.state),
            "is_stationary": self.is_stationary,
            "persistence_counter": self.persistence_counter,
            "leaving_counter": self.leaving_counter,
            "accel_norm": self.last_accel_norm,
            "gyro_norm": self.last_gyro_norm,
            "accel_var": self.last_accel_var,
            "gyro_var": self.last_gyro_var,
            "is_candidate": self.last_is_candidate,
        }

    def reset(self):
        """Reset state machine and buffers."""
        self.state = ZUPTState.MOVING
        self.persistence_counter = 0
        self.leaving_counter = 0
        self._accel_history.clear()
        self._gyro_history.clear()
        self.last_accel_norm = 0.0
        self.last_gyro_norm = 0.0
        self.last_accel_var = 0.0
        self.last_gyro_var = 0.0
        self.last_is_candidate = False
