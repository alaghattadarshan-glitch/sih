"""15-State Error-State Vector Specification and Index Definitions.

Defines the structure and index mapping for the 15-element error-state vector δx:
- [0:3]   δp : Position Error in ENU frame (meters) [δp_E, δp_N, δp_U]
- [3:6]   δv : Velocity Error in ENU frame (m/s) [δv_E, δv_N, δv_U]
- [6:9]   δθ : Attitude Error vector (radians) [δθ_x, δθ_y, δθ_z]
- [9:12]  δba: Accelerometer Bias Error (m/s^2) [δb_ax, δb_ay, δb_az]
- [12:15] δbg: Gyroscope Bias Error (rad/s) [δb_gx, δb_gy, δb_gz]
"""

from dataclasses import dataclass
import numpy as np

# Total dimension of error-state vector
STATE_DIM = 15

# Slice definitions for state vector access
POS_SLICE = slice(0, 3)
VEL_SLICE = slice(3, 6)
ATT_SLICE = slice(6, 9)
AB_SLICE = slice(9, 12)
GB_SLICE = slice(12, 15)


@dataclass
class ESKFStateConfig:
    """Configuration container for ESKF initial uncertainties and process noise.

    Attributes:
        pos_std (float): Initial position error 1-sigma std dev in meters.
        vel_std (float): Initial velocity error 1-sigma std dev in m/s.
        att_std (float): Initial attitude error 1-sigma std dev in radians.
        accel_bias_std (float): Initial accel bias 1-sigma std dev in m/s^2.
        gyro_bias_std (float): Initial gyro bias 1-sigma std dev in rad/s.
        accel_noise_std (float): Accelerometer measurement noise std dev in m/s^2.
        gyro_noise_std (float): Gyroscope measurement noise std dev in rad/s.
        accel_bias_rw_std (float): Accelerometer bias random walk std dev in m/s^2/s.
        gyro_bias_rw_std (float): Gyroscope bias random walk std dev in rad/s/s.
        gnss_pos_std_default (float): Default GNSS position measurement noise std dev in meters.
    """

    pos_std: float = 1.0
    vel_std: float = 0.1
    att_std: float = 0.01
    accel_bias_std: float = 0.05
    gyro_bias_std: float = 0.005
    accel_noise_std: float = 0.05
    gyro_noise_std: float = 0.005
    accel_bias_rw_std: float = 0.0001
    gyro_bias_rw_std: float = 0.00001
    gnss_pos_std_default: float = 2.5

    def build_initial_covariance(self) -> np.ndarray:
        """Construct initial 15x15 error covariance matrix P_0."""
        P0 = np.zeros((STATE_DIM, STATE_DIM), dtype=np.float64)
        P0[POS_SLICE, POS_SLICE] = np.eye(3) * (self.pos_std**2)
        P0[VEL_SLICE, VEL_SLICE] = np.eye(3) * (self.vel_std**2)
        P0[ATT_SLICE, ATT_SLICE] = np.eye(3) * (self.att_std**2)
        P0[AB_SLICE, AB_SLICE] = np.eye(3) * (self.accel_bias_std**2)
        P0[GB_SLICE, GB_SLICE] = np.eye(3) * (self.gyro_bias_std**2)
        return P0

    def build_continuous_process_noise(self) -> np.ndarray:
        """Construct continuous 12x12 process noise spectral density matrix Q_c."""
        Qc = np.zeros((12, 12), dtype=np.float64)
        Qc[0:3, 0:3] = np.eye(3) * (self.accel_noise_std**2)
        Qc[3:6, 3:6] = np.eye(3) * (self.gyro_noise_std**2)
        Qc[6:9, 6:9] = np.eye(3) * (self.accel_bias_rw_std**2)
        Qc[9:12, 9:12] = np.eye(3) * (self.gyro_bias_rw_std**2)
        return Qc
