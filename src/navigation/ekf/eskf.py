"""15-State Error-State Kalman Filter (ESKF) Engine.

Fuses high-rate Strapdown INS nominal propagation with low-rate GNSS position fixes,
estimating 15 error states (position, velocity, attitude, accel bias, gyro bias)
and injecting corrections into the nominal navigation state.
"""

from typing import Tuple, Optional
import numpy as np

from src.data.observations import IMUObservation, GNSSObservation
from src.navigation.state import NavigationState
from src.navigation.ins import StrapdownINS
from src.navigation.quaternion import (
    quaternion_normalize,
    quaternion_multiply,
    quaternion_to_rotation_matrix,
)
from src.navigation.ekf.state import (
    STATE_DIM,
    POS_SLICE,
    VEL_SLICE,
    ATT_SLICE,
    AB_SLICE,
    GB_SLICE,
    ESKFStateConfig,
)
from src.navigation.ekf.models import (
    build_continuous_error_matrix,
    build_continuous_noise_matrix,
    discretize_dynamics,
    build_gnss_position_measurement_matrix,
    build_nhc_measurement_matrix,
    build_zupt_measurement_matrix,
)



class ErrorStateKalmanFilter:
    """15-State Error-State Kalman Filter for GNSS/INS Sensor Fusion."""

    def __init__(self, config: Optional[ESKFStateConfig] = None, g_val: float = 9.80665):
        """Initialize ESKF engine.

        Args:
            config (Optional[ESKFStateConfig]): Configuration settings and initial noise std devs.
            g_val (float): Gravitational acceleration magnitude in m/s^2.
        """
        self.config = config if config is not None else ESKFStateConfig()
        self.ins = StrapdownINS(g_val=g_val)
        self.P: np.ndarray = self.config.build_initial_covariance()
        self.Qc: np.ndarray = self.config.build_continuous_process_noise()
        self.H_gnss: np.ndarray = build_gnss_position_measurement_matrix()
        self._is_initialized = False

    @property
    def is_initialized(self) -> bool:
        """Return True if ESKF engine is initialized."""
        return self._is_initialized and self.ins.is_initialized

    def initialize(
        self,
        initial_time: float,
        initial_llh: Tuple[float, float, float],
        initial_velocity_enu: np.ndarray,
        initial_quaternion: np.ndarray,
        accel_bias: Optional[np.ndarray] = None,
        gyro_bias: Optional[np.ndarray] = None,
    ):
        """Initialize nominal INS state and ESKF error covariance matrix.

        Args:
            initial_time (float): Initial timestamp in seconds.
            initial_llh (Tuple[float, float, float]): Geographic origin (lat_deg, lon_deg, h_m).
            initial_velocity_enu (np.ndarray): Initial velocity vector [V_E, V_N, V_U] in m/s.
            initial_quaternion (np.ndarray): Initial attitude quaternion [q_w, q_x, q_y, q_z].
            accel_bias (Optional[np.ndarray]): Estimated initial accelerometer bias in m/s^2.
            gyro_bias (Optional[np.ndarray]): Estimated initial gyroscope bias in rad/s.
        """
        self.ins.initialize(
            initial_time=initial_time,
            initial_llh=initial_llh,
            initial_velocity_enu=initial_velocity_enu,
            initial_quaternion=initial_quaternion,
            accel_bias=accel_bias,
            gyro_bias=gyro_bias,
        )
        self.P = self.config.build_initial_covariance()
        self._is_initialized = True

    def predict(self, imu_obs: IMUObservation) -> NavigationState:
        """Predict nominal state via Strapdown INS and propagate error covariance matrix P.

        Args:
            imu_obs (IMUObservation): High-rate IMU observation sample.

        Returns:
            NavigationState: Updated nominal navigation state.
        """
        if not self.is_initialized:
            raise RuntimeError("ESKF engine must be initialized before calling predict().")

        t_prev = self.ins.get_state().timestamp
        dt = imu_obs.timestamp - t_prev

        if dt <= 0:
            if dt == 0:
                return self.ins.get_state()
            raise ValueError(f"Non-positive dt in ESKF prediction: {dt}")

        # Extract current nominal rotation matrix and corrected specific force
        state_old = self.ins.get_state()
        R_b2n = state_old.rotation_matrix
        raw_accel = np.array(
            [imu_obs.accelerometer_x, imu_obs.accelerometer_y, imu_obs.accelerometer_z],
            dtype=np.float64,
        )
        f_body = raw_accel - state_old.accelerometer_bias

        # 1. Step nominal INS integration
        nominal_state = self.ins.update(imu_obs)

        # 2. Build continuous error dynamics and noise matrices
        F = build_continuous_error_matrix(R_b2n, f_body)
        G = build_continuous_noise_matrix(R_b2n)

        # 3. Discretize dynamics
        Phi, Qd = discretize_dynamics(F, G, self.Qc, dt)

        # 4. Propagate covariance: P = Phi * P * Phi^T + Qd
        self.P = Phi @ self.P @ Phi.T + Qd
        self.P = 0.5 * (self.P + self.P.T)  # Ensure matrix symmetry

        return nominal_state

    def update_gnss(self, gnss_obs: GNSSObservation) -> Optional[NavigationState]:
        """Perform measurement correction using a GNSS position fix.

        Args:
            gnss_obs (GNSSObservation): GNSS position observation.

        Returns:
            Optional[NavigationState]: Corrected navigation state after error injection.
        """
        if not self.is_initialized or self.ins.local_frame is None:
            raise RuntimeError("ESKF engine must be initialized before calling update_gnss().")

        # Convert GNSS geographic coordinates to local ENU position
        e_gnss, n_gnss, u_gnss = self.ins.local_frame.to_enu(
            gnss_obs.latitude, gnss_obs.longitude, gnss_obs.altitude
        )
        z_gnss = np.array([e_gnss, n_gnss, u_gnss], dtype=np.float64)

        nominal_state = self.ins.get_state()
        p_ins = nominal_state.position_enu

        # Innovation vector y = z - h(x)
        innovation = z_gnss - p_ins

        # Determine measurement noise covariance R
        h_std = (
            gnss_obs.horizontal_accuracy
            if gnss_obs.horizontal_accuracy is not None and gnss_obs.horizontal_accuracy > 0
            else self.config.gnss_pos_std_default
        )
        v_std = (
            gnss_obs.vertical_accuracy
            if gnss_obs.vertical_accuracy is not None and gnss_obs.vertical_accuracy > 0
            else self.config.gnss_pos_std_default * 1.5
        )
        R_gnss = np.diag([h_std**2, h_std**2, v_std**2])

        # Innovation covariance S = H*P*H^T + R
        H = self.H_gnss
        S = H @ self.P @ H.T + R_gnss

        # Compute Kalman gain K = P * H^T * S^-1 using linear solver for stability
        try:
            K = np.linalg.solve(S.T, (self.P @ H.T).T).T
        except np.linalg.LinAlgError:
            K = self.P @ H.T @ np.linalg.inv(S)

        # 15-state error estimate dx = K * y
        dx = K @ innovation

        # Joseph form covariance update: P = (I - K*H) * P * (I - K*H)^T + K * R * K^T
        I_KH = np.eye(STATE_DIM, dtype=np.float64) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_gnss @ K.T
        self.P = 0.5 * (self.P + self.P.T)  # Ensure matrix symmetry

        # Error Injection into nominal INS state
        corrected_pos = nominal_state.position_enu + dx[POS_SLICE]
        corrected_vel = nominal_state.velocity_enu + dx[VEL_SLICE]

        # Small angle attitude error correction dq = [1, 0.5*dtheta_x, 0.5*dtheta_y, 0.5*dtheta_z]
        dtheta = dx[ATT_SLICE]
        dq = quaternion_normalize(
            np.array([1.0, 0.5 * dtheta[0], 0.5 * dtheta[1], 0.5 * dtheta[2]], dtype=np.float64)
        )
        corrected_q = quaternion_multiply(dq, nominal_state.orientation_quaternion)

        corrected_ab = nominal_state.accelerometer_bias + dx[AB_SLICE]
        corrected_gb = nominal_state.gyroscope_bias + dx[GB_SLICE]

        self.ins._state = NavigationState(
            timestamp=nominal_state.timestamp,
            position_enu=corrected_pos,
            velocity_enu=corrected_vel,
            orientation_quaternion=corrected_q,
            accelerometer_bias=corrected_ab,
            gyroscope_bias=corrected_gb,
        )

        return self.ins.get_state()

    def update_ai_displacement(
        self,
        ref_pos_enu: np.ndarray,
        delta_p_ai: np.ndarray,
        r_ai_std: Tuple[float, float, float] = (1.5, 1.5, 3.0),
        gate_threshold: float = 4.0,
    ) -> Tuple[Optional[NavigationState], bool, float]:
        """Perform AI position displacement pseudo-measurement update with innovation gating.

        Formulates position pseudo-measurement z_AI = ref_pos_enu + delta_p_ai.
        Calculates innovation y_AI = z_AI - p_ins(t_end) and Mahalanobis distance gating statistic.
        If accepted, applies Joseph-form covariance update and injects 15-state error corrections.

        Args:
            ref_pos_enu (np.ndarray): Nominal position [East, North, Up] in meters captured at window start.
            delta_p_ai (np.ndarray): AI predicted displacement vector [dE, dN, dU] over window.
            r_ai_std (Tuple[float, float, float]): AI measurement standard deviations (std_E, std_N, std_U).
            gate_threshold (float): Mahalanobis distance gating threshold.

        Returns:
            Tuple[Optional[NavigationState], bool, float]:
                - Navigation state after correction (or uncorrected state if rejected).
                - Boolean flag indicating whether the update was accepted (True) or rejected by gating (False).
                - Computed Mahalanobis distance value.
        """
        if not self.is_initialized:
            raise RuntimeError("ESKF engine must be initialized before calling update_ai_displacement().")

        z_ai = np.array(ref_pos_enu, dtype=np.float64) + np.array(delta_p_ai, dtype=np.float64)
        nominal_state = self.ins.get_state()
        p_ins = nominal_state.position_enu

        # Innovation vector y_AI = z_AI - p_ins
        innovation = z_ai - p_ins

        # AI Measurement Noise Covariance R_AI
        R_ai = np.diag([r_ai_std[0] ** 2, r_ai_std[1] ** 2, r_ai_std[2] ** 2])

        # Innovation covariance S = H*P*H^T + R_AI
        H = self.H_gnss
        S = H @ self.P @ H.T + R_ai

        # Calculate Mahalanobis distance / normalized innovation statistic
        try:
            mahalanobis_sq = float(innovation.T @ np.linalg.solve(S, innovation))
        except np.linalg.LinAlgError:
            mahalanobis_sq = float(innovation.T @ np.linalg.inv(S) @ innovation)

        mahalanobis_dist = float(np.sqrt(max(0.0, mahalanobis_sq)))

        # Innovation Gating Check
        if mahalanobis_dist > gate_threshold:
            # Reject update due to statistical inconsistency
            return self.ins.get_state(), False, mahalanobis_dist

        # Compute Kalman Gain K = P * H^T * S^-1
        try:
            K = np.linalg.solve(S.T, (self.P @ H.T).T).T
        except np.linalg.LinAlgError:
            K = self.P @ H.T @ np.linalg.inv(S)

        # 15-state error estimate dx = K * y
        dx = K @ innovation

        # Joseph form covariance update
        I_KH = np.eye(STATE_DIM, dtype=np.float64) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_ai @ K.T
        self.P = 0.5 * (self.P + self.P.T)  # Ensure matrix symmetry

        # Error Injection into nominal INS state
        corrected_pos = nominal_state.position_enu + dx[POS_SLICE]
        corrected_vel = nominal_state.velocity_enu + dx[VEL_SLICE]

        dtheta = dx[ATT_SLICE]
        dq = quaternion_normalize(
            np.array([1.0, 0.5 * dtheta[0], 0.5 * dtheta[1], 0.5 * dtheta[2]], dtype=np.float64)
        )
        corrected_q = quaternion_multiply(dq, nominal_state.orientation_quaternion)

        corrected_ab = nominal_state.accelerometer_bias + dx[AB_SLICE]
        corrected_gb = nominal_state.gyroscope_bias + dx[GB_SLICE]

        # Update nominal INS state
        self.ins._state = NavigationState(
            timestamp=nominal_state.timestamp,
            position_enu=corrected_pos,
            velocity_enu=corrected_vel,
            orientation_quaternion=corrected_q,
            accelerometer_bias=corrected_ab,
            gyroscope_bias=corrected_gb,
        )

        return self.ins.get_state(), True, mahalanobis_dist

    def update_map_constraint(
        self,
        projected_pos_enu: np.ndarray,
        confidence: float = 1.0,
        r_map_std_base: Tuple[float, float, float] = (2.0, 2.0, 5.0),
        gate_threshold: float = 4.0,
    ) -> Tuple[Optional[NavigationState], bool, float]:
        """Perform soft road-network map constraint pseudo-measurement update with innovation gating.

        Formulates position pseudo-measurement z_map = projected_pos_enu.
        Scales measurement covariance inversely with candidate confidence:
        R_map = diag((std_E / c)^2, (std_N / c)^2, std_U^2) where c = max(0.05, confidence).

        Calculates innovation y_map = z_map - p_ins and Mahalanobis gating statistic.
        If accepted, applies Joseph-form covariance update and injects 15-state error corrections.

        Args:
            projected_pos_enu (np.ndarray): Closest projected point on matched road polyline in meters [E, N, U].
            confidence (float): Match confidence in [0.0, 1.0].
            r_map_std_base (Tuple[float, float, float]): Base standard deviations (std_E, std_N, std_U) in meters.
            gate_threshold (float): Mahalanobis distance gating threshold.

        Returns:
            Tuple[Optional[NavigationState], bool, float]:
                - Navigation state after correction (or uncorrected state if rejected).
                - Boolean flag indicating whether the update was accepted (True) or rejected by gating (False).
                - Computed Mahalanobis distance value.
        """
        if not self.is_initialized:
            raise RuntimeError("ESKF engine must be initialized before calling update_map_constraint().")

        z_map = np.array(projected_pos_enu, dtype=np.float64)
        nominal_state = self.ins.get_state()
        p_ins = nominal_state.position_enu

        # Innovation vector y_map = z_map - p_ins
        innovation = z_map - p_ins

        # Adaptive Map Measurement Noise Covariance R_map
        # Lower confidence -> larger covariance (weaker constraint); higher confidence -> tighter constraint
        c = max(0.05, float(confidence))
        std_e = r_map_std_base[0] / c
        std_n = r_map_std_base[1] / c
        std_u = r_map_std_base[2]
        R_map = np.diag([std_e ** 2, std_n ** 2, std_u ** 2])

        # Innovation covariance S = H*P*H^T + R_map
        H = self.H_gnss
        S = H @ self.P @ H.T + R_map

        # Calculate Mahalanobis distance
        try:
            mahalanobis_sq = float(innovation.T @ np.linalg.solve(S, innovation))
        except np.linalg.LinAlgError:
            mahalanobis_sq = float(innovation.T @ np.linalg.inv(S) @ innovation)

        mahalanobis_dist = float(np.sqrt(max(0.0, mahalanobis_sq)))

        # Innovation Gating Check
        if mahalanobis_dist > gate_threshold:
            # Reject update due to statistical inconsistency
            return self.ins.get_state(), False, mahalanobis_dist

        # Compute Kalman Gain K = P * H^T * S^-1
        try:
            K = np.linalg.solve(S.T, (self.P @ H.T).T).T
        except np.linalg.LinAlgError:
            K = self.P @ H.T @ np.linalg.inv(S)

        # 15-state error estimate dx = K * y
        dx = K @ innovation

        # Joseph form covariance update
        I_KH = np.eye(STATE_DIM, dtype=np.float64) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_map @ K.T
        self.P = 0.5 * (self.P + self.P.T)  # Ensure matrix symmetry

        # Error Injection into nominal INS state
        corrected_pos = nominal_state.position_enu + dx[POS_SLICE]
        corrected_vel = nominal_state.velocity_enu + dx[VEL_SLICE]

        dtheta = dx[ATT_SLICE]
        dq = quaternion_normalize(
            np.array([1.0, 0.5 * dtheta[0], 0.5 * dtheta[1], 0.5 * dtheta[2]], dtype=np.float64)
        )
        corrected_q = quaternion_multiply(dq, nominal_state.orientation_quaternion)

        corrected_ab = nominal_state.accelerometer_bias + dx[AB_SLICE]
        corrected_gb = nominal_state.gyroscope_bias + dx[GB_SLICE]

        # Update nominal INS state
        self.ins._state = NavigationState(
            timestamp=nominal_state.timestamp,
            position_enu=corrected_pos,
            velocity_enu=corrected_vel,
            orientation_quaternion=corrected_q,
            accelerometer_bias=corrected_ab,
            gyroscope_bias=corrected_gb,
        )

        return self.ins.get_state(), True, mahalanobis_dist

    def update_nhc_constraint(
        self,
        sigma_y: float = 0.1,
        sigma_z: float = 0.1,
        gate_threshold: float = 4.0,
    ) -> Tuple[Optional[NavigationState], bool, float, np.ndarray]:
        """Perform Non-Holonomic Constraint (NHC) measurement update with Mahalanobis gating.

        Constrains vehicle body lateral and vertical velocities:
        v_body_y ≈ 0, v_body_z ≈ 0.
        z_NHC = [0, 0]^T
        Predicted measurement h(x) = [v_body_y, v_body_z]^T
        Innovation y = [ -v_body_y, -v_body_z ]^T

        Args:
            sigma_y (float): Body lateral velocity measurement noise std dev in m/s.
            sigma_z (float): Body vertical velocity measurement noise std dev in m/s.
            gate_threshold (float): Mahalanobis distance gating threshold.

        Returns:
            Tuple[Optional[NavigationState], bool, float, np.ndarray]:
                - Corrected (or uncorrected if rejected) NavigationState.
                - Boolean flag indicating whether the update was accepted (True) or gated out (False).
                - Mahalanobis distance.
                - Innovation vector [y_lateral, y_vertical].
        """
        if not self.is_initialized:
            raise RuntimeError("ESKF engine must be initialized before calling update_nhc_constraint().")

        nominal_state = self.ins.get_state()
        R_b2n = nominal_state.rotation_matrix
        v_enu = nominal_state.velocity_enu

        # Transform ENU velocity to Body frame: v_body = R_b2n^T * v_enu
        v_body = R_b2n.T @ v_enu
        vx_b, vy_b, vz_b = v_body

        # Innovation vector: z_nhc - h(x) = [0, 0]^T - [vy_b, vz_b]^T
        innovation = np.array([-vy_b, -vz_b], dtype=np.float64)

        # Measurement noise covariance R_nhc (2x2)
        R_nhc = np.diag([sigma_y**2, sigma_z**2])

        # Measurement matrix H_nhc (2x15)
        H = build_nhc_measurement_matrix(R_b2n, v_body)

        # Innovation covariance S = H * P * H^T + R_nhc (2x2)
        S = H @ self.P @ H.T + R_nhc

        # Calculate Mahalanobis distance
        try:
            mahalanobis_sq = float(innovation.T @ np.linalg.solve(S, innovation))
        except np.linalg.LinAlgError:
            mahalanobis_sq = float(innovation.T @ np.linalg.inv(S) @ innovation)

        mahalanobis_dist = float(np.sqrt(max(0.0, mahalanobis_sq)))

        # Innovation Gating Check
        if mahalanobis_dist > gate_threshold:
            return self.ins.get_state(), False, mahalanobis_dist, innovation

        # Compute Kalman Gain K = P * H^T * S^-1 (15x2)
        try:
            K = np.linalg.solve(S.T, (self.P @ H.T).T).T
        except np.linalg.LinAlgError:
            K = self.P @ H.T @ np.linalg.inv(S)

        # 15-state error estimate dx = K * y
        dx = K @ innovation

        # Joseph form covariance update: P = (I - K*H)*P*(I - K*H)^T + K*R*K^T
        I_KH = np.eye(STATE_DIM, dtype=np.float64) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_nhc @ K.T
        self.P = 0.5 * (self.P + self.P.T)  # Ensure matrix symmetry

        # Error Injection into nominal INS state
        corrected_pos = nominal_state.position_enu + dx[POS_SLICE]
        corrected_vel = nominal_state.velocity_enu + dx[VEL_SLICE]

        dtheta = dx[ATT_SLICE]
        dq = quaternion_normalize(
            np.array([1.0, 0.5 * dtheta[0], 0.5 * dtheta[1], 0.5 * dtheta[2]], dtype=np.float64)
        )
        corrected_q = quaternion_multiply(dq, nominal_state.orientation_quaternion)

        corrected_ab = nominal_state.accelerometer_bias + dx[AB_SLICE]
        corrected_gb = nominal_state.gyroscope_bias + dx[GB_SLICE]

        # Update nominal INS state
        self.ins._state = NavigationState(
            timestamp=nominal_state.timestamp,
            position_enu=corrected_pos,
            velocity_enu=corrected_vel,
            orientation_quaternion=corrected_q,
            accelerometer_bias=corrected_ab,
            gyroscope_bias=corrected_gb,
        )

        return self.ins.get_state(), True, mahalanobis_dist, innovation

    def update_zupt(
        self,
        sigma_zupt: float = 0.01,
        gate_threshold: float = 4.0,
    ) -> Tuple[Optional[NavigationState], bool, float, np.ndarray]:
        """Perform Zero-Velocity Update (ZUPT) measurement correction with Mahalanobis gating.

        When the vehicle is confidently stationary, enforces 3D velocity to zero:
        z_ZUPT = [0, 0, 0]^T
        Predicted measurement h(x) = v_enu
        Innovation y = [0, 0, 0]^T - v_enu = -v_enu

        Args:
            sigma_zupt (float): Zero-velocity measurement noise std dev per axis in m/s.
            gate_threshold (float): Mahalanobis distance gating threshold.

        Returns:
            Tuple[Optional[NavigationState], bool, float, np.ndarray]:
                - Corrected (or uncorrected if rejected) NavigationState.
                - Boolean flag indicating whether the update was accepted (True) or gated out (False).
                - Mahalanobis distance.
                - Innovation vector [y_east, y_north, y_up].
        """
        if not self.is_initialized:
            raise RuntimeError("ESKF engine must be initialized before calling update_zupt().")

        nominal_state = self.ins.get_state()
        v_enu = nominal_state.velocity_enu

        # Innovation vector: z_zupt - h(x) = [0, 0, 0]^T - v_enu
        innovation = -v_enu.copy()

        # Measurement noise covariance R_zupt (3x3)
        R_zupt = np.eye(3, dtype=np.float64) * (sigma_zupt**2)

        # Measurement matrix H_zupt (3x15)
        H = build_zupt_measurement_matrix()

        # Innovation covariance S = H * P * H^T + R_zupt = P_vv + R_zupt (3x3)
        S = self.P[VEL_SLICE, VEL_SLICE] + R_zupt

        # Calculate Mahalanobis distance
        try:
            mahalanobis_sq = float(innovation.T @ np.linalg.solve(S, innovation))
        except np.linalg.LinAlgError:
            mahalanobis_sq = float(innovation.T @ np.linalg.inv(S) @ innovation)

        mahalanobis_dist = float(np.sqrt(max(0.0, mahalanobis_sq)))

        # Innovation Gating Check
        if mahalanobis_dist > gate_threshold:
            return self.ins.get_state(), False, mahalanobis_dist, innovation

        # Compute Kalman Gain K = P * H^T * S^-1 (15x3)
        try:
            K = np.linalg.solve(S.T, (self.P @ H.T).T).T
        except np.linalg.LinAlgError:
            K = self.P @ H.T @ np.linalg.inv(S)

        # 15-state error estimate dx = K * y
        dx = K @ innovation

        # Joseph form covariance update: P = (I - K*H)*P*(I - K*H)^T + K*R*K^T
        I_KH = np.eye(STATE_DIM, dtype=np.float64) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_zupt @ K.T
        self.P = 0.5 * (self.P + self.P.T)  # Ensure matrix symmetry

        # Error Injection into nominal INS state
        corrected_pos = nominal_state.position_enu + dx[POS_SLICE]
        corrected_vel = nominal_state.velocity_enu + dx[VEL_SLICE]

        dtheta = dx[ATT_SLICE]
        dq = quaternion_normalize(
            np.array([1.0, 0.5 * dtheta[0], 0.5 * dtheta[1], 0.5 * dtheta[2]], dtype=np.float64)
        )
        corrected_q = quaternion_multiply(dq, nominal_state.orientation_quaternion)

        corrected_ab = nominal_state.accelerometer_bias + dx[AB_SLICE]
        corrected_gb = nominal_state.gyroscope_bias + dx[GB_SLICE]

        # Update nominal INS state
        self.ins._state = NavigationState(
            timestamp=nominal_state.timestamp,
            position_enu=corrected_pos,
            velocity_enu=corrected_vel,
            orientation_quaternion=corrected_q,
            accelerometer_bias=corrected_ab,
            gyroscope_bias=corrected_gb,
        )

        return self.ins.get_state(), True, mahalanobis_dist, innovation

    def get_state(self) -> Optional[NavigationState]:
        """Return current corrected navigation state."""
        return self.ins.get_state()


    def get_covariance(self) -> np.ndarray:
        """Return current 15x15 error covariance matrix P."""
        return self.P.copy()

    def reset(self):
        """Reset filter state and INS engine."""
        self.ins.reset()
        self.P = self.config.build_initial_covariance()
        self._is_initialized = False
