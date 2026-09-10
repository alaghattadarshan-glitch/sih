"""Canonical End-to-End Navigation Pipeline.

Consolidates the complete multi-sensor navigation stack into a single, unified source of truth:
1. Static IMU calibration and bias estimation.
2. 15-state Error-State Kalman Filter (ESKF) for nominal INS propagation and covariance management.
3. Online GNSS Outage Detection state machine with automatic degradation and recovery tracking.
4. Causal AI-based displacement pseudo-measurements via PyTorch/ONNX Temporal Convolutional Networks.
5. Vehicle Non-Holonomic Constraints (NHC) for lateral/vertical velocity suppression.
6. Zero-Velocity Updates (ZUPT) driven by stationary detector.
7. Road Network Soft Map Matching pseudo-measurements with Mahalanobis innovation gating.
8. Standardized navigation operational lifecycle and status modes.
"""

from enum import Enum
import math
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from src.data.observations import IMUObservation, GNSSObservation
from src.navigation.state import NavigationState
from src.navigation.ekf.eskf import ErrorStateKalmanFilter
from src.navigation.ekf.state import ESKFStateConfig
from src.outage_detection.detector import GNSSOutageDetector, GNSSStatus
from src.navigation.imu.zupt_detector import ZUPTDetector, ZUPTConfig, ZUPTState
from src.ml.inference import DriftPredictor
from src.map_matching.network import RoadNetwork
from src.map_matching.matcher import MapMatcher
from src.calibration.imu_calibration import estimate_static_bias, apply_imu_calibration
from src.navigation.ai_fusion import AIESKFPipeline


class NavigationMode(Enum):
    """Operational mode of the navigation system."""
    GNSS_AIDED = "GNSS_AIDED"
    DEGRADED = "DEGRADED"
    GNSS_OUTAGE = "GNSS_OUTAGE"
    RECOVERING = "RECOVERING"

    @classmethod
    def from_gnss_status(cls, status: GNSSStatus) -> "NavigationMode":
        """Map internal GNSS detector status to standardized NavigationMode."""
        if status == GNSSStatus.GOOD:
            return cls.GNSS_AIDED
        elif status == GNSSStatus.DEGRADED:
            return cls.DEGRADED
        elif status == GNSSStatus.OUTAGE:
            return cls.GNSS_OUTAGE
        elif status == GNSSStatus.RECOVERING:
            return cls.RECOVERING
        return cls.GNSS_AIDED


class EndToEndNavigationPipeline:
    """Canonical Unified End-to-End Navigation Pipeline."""

    def __init__(
        self,
        eskf: Optional[ErrorStateKalmanFilter] = None,
        detector: Optional[GNSSOutageDetector] = None,
        predictor: Optional[DriftPredictor] = None,
        network: Optional[RoadNetwork] = None,
        matcher: Optional[MapMatcher] = None,
        zupt_detector: Optional[ZUPTDetector] = None,
        enable_ai: bool = True,
        enable_nhc: bool = True,
        enable_zupt: bool = True,
        enable_map_matching: bool = False,
        r_ai_std: Tuple[float, float, float] = (1.5, 1.5, 3.0),
        gate_threshold: float = 4.0,
        nhc_sigma_y: float = 0.1,
        nhc_sigma_z: float = 0.1,
        nhc_gate_threshold: float = 4.0,
        zupt_sigma: float = 0.01,
        zupt_gate_threshold: float = 4.0,
        r_map_std: Tuple[float, float, float] = (2.0, 2.0, 5.0),
        map_gate_threshold: float = 4.0,
        window_size_samples: int = 100,
    ):
        """Initialize the unified end-to-end navigation pipeline.

        Args:
            eskf: 15-state ESKF engine. Defaults to a new ErrorStateKalmanFilter instance.
            detector: GNSS Outage Detector. Defaults to a new GNSSOutageDetector instance.
            predictor: AI DriftPredictor model. Optional.
            network: RoadNetwork for map matching. Optional.
            matcher: MapMatcher instance. Optional.
            zupt_detector: ZUPT detector state machine. Optional.
            enable_ai: Enable AI displacement updates during outages.
            enable_nhc: Enable Non-Holonomic Constraints during motion in outages.
            enable_zupt: Enable Zero-Velocity Updates when stationary.
            enable_map_matching: Enable soft map matching constraints.
            r_ai_std: AI measurement noise std dev in ENU (m).
            gate_threshold: Mahalanobis gating threshold for AI updates.
            nhc_sigma_y: NHC lateral velocity noise std dev (m/s).
            nhc_sigma_z: NHC vertical velocity noise std dev (m/s).
            nhc_gate_threshold: Mahalanobis gating threshold for NHC.
            zupt_sigma: ZUPT velocity noise std dev (m/s).
            zupt_gate_threshold: Mahalanobis gating threshold for ZUPT.
            r_map_std: Map matching measurement noise std dev in ENU (m).
            map_gate_threshold: Mahalanobis gating threshold for map matching.
            window_size_samples: Window length for AI feature buffer (default 100).
        """
        self.eskf = eskf if eskf is not None else ErrorStateKalmanFilter()
        self.detector = detector if detector is not None else GNSSOutageDetector()
        self.predictor = predictor
        self.network = network
        self.matcher = matcher if matcher is not None else (MapMatcher() if network is not None else None)
        self.zupt_detector = zupt_detector if zupt_detector is not None else ZUPTDetector()

        self.enable_ai = enable_ai
        self.enable_nhc = enable_nhc
        self.enable_zupt = enable_zupt
        self.enable_map_matching = enable_map_matching

        self.r_ai_std = r_ai_std
        self.gate_threshold = gate_threshold
        self.nhc_sigma_y = nhc_sigma_y
        self.nhc_sigma_z = nhc_sigma_z
        self.nhc_gate_threshold = nhc_gate_threshold
        self.zupt_sigma = zupt_sigma
        self.zupt_gate_threshold = zupt_gate_threshold
        self.r_map_std = r_map_std
        self.map_gate_threshold = map_gate_threshold
        self.window_size_samples = window_size_samples

        # Static calibration parameters
        self.accel_bias = np.zeros(3, dtype=np.float64)
        self.gyro_bias = np.zeros(3, dtype=np.float64)
        self.is_calibrated: bool = False

        # Internal core fusion engine
        self._fusion_engine: Optional[AIESKFPipeline] = None
        self._init_fusion_engine()

        # Operational state
        self._latest_state: Optional[NavigationState] = None
        self._current_mode: NavigationMode = NavigationMode.GNSS_AIDED
        self._mode_history: List[Tuple[float, NavigationMode]] = []
        self._is_initialized: bool = False
        self._sample_count: int = 0
        self._imu_sample_count: int = 0
        self._gnss_sample_count: int = 0

    def _init_fusion_engine(self) -> None:
        """Instantiate the internal AIESKFPipeline with current configuration."""
        self._fusion_engine = AIESKFPipeline(
            eskf=self.eskf,
            detector=self.detector,
            predictor=self.predictor,
            enable_ai=self.enable_ai,
            r_ai_std=self.r_ai_std,
            gate_threshold=self.gate_threshold,
            window_size_samples=self.window_size_samples,
            network=self.network,
            matcher=self.matcher,
            enable_map_matching=self.enable_map_matching,
            r_map_std=self.r_map_std,
            map_gate_threshold=self.map_gate_threshold,
            enable_nhc=self.enable_nhc,
            nhc_sigma_y=self.nhc_sigma_y,
            nhc_sigma_z=self.nhc_sigma_z,
            nhc_gate_threshold=self.nhc_gate_threshold,
            enable_zupt=self.enable_zupt,
            zupt_detector=self.zupt_detector,
            zupt_sigma=self.zupt_sigma,
            zupt_gate_threshold=self.zupt_gate_threshold,
        )

    # --------------------------------------------------------------------------
    # Lifecycle Methods
    # --------------------------------------------------------------------------

    def initialize(
        self,
        init_state: Optional[NavigationState] = None,
        origin_lat: float = 0.0,
        origin_lon: float = 0.0,
        origin_alt: float = 0.0,
        init_velocity_enu: Optional[np.ndarray] = None,
        init_quaternion: Optional[np.ndarray] = None,
    ) -> NavigationState:
        """Initialize pipeline with starting navigation state.

        Args:
            init_state: Full NavigationState object if available.
            origin_lat: Reference latitude (deg).
            origin_lon: Reference longitude (deg).
            origin_alt: Reference altitude (m).
            init_velocity_enu: Initial velocity [Ve, Vn, Vu] in m/s.
            init_quaternion: Initial attitude [qw, qx, qy, qz].

        Returns:
            NavigationState: Initialized state.
        """
        if init_state is not None:
            self._latest_state = init_state
            self.eskf.initialize(
                initial_time=init_state.timestamp,
                initial_llh=(origin_lat, origin_lon, origin_alt),
                initial_velocity_enu=init_state.velocity_enu,
                initial_quaternion=init_state.orientation_quaternion,
                accel_bias=init_state.accelerometer_bias if not self.is_calibrated else self.accel_bias,
                gyro_bias=init_state.gyroscope_bias if not self.is_calibrated else self.gyro_bias,
            )
        else:
            v0 = init_velocity_enu if init_velocity_enu is not None else np.zeros(3, dtype=np.float64)
            q0 = init_quaternion if init_quaternion is not None else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            self.eskf.initialize(
                initial_time=0.0,
                initial_llh=(origin_lat, origin_lon, origin_alt),
                initial_velocity_enu=v0,
                initial_quaternion=q0,
                accel_bias=self.accel_bias,
                gyro_bias=self.gyro_bias,
            )
            self._latest_state = self.eskf.get_state()

        self._is_initialized = True
        self._current_mode = NavigationMode.GNSS_AIDED
        self._mode_history = [(self._latest_state.timestamp, self._current_mode)]
        return self._latest_state

    def calibrate(
        self,
        stationary_samples: List[IMUObservation],
        expected_gravity: float = 9.80665,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Perform static IMU calibration using stationary sensor samples.

        Args:
            stationary_samples: List of IMU observations collected while completely stationary.
            expected_gravity: Local gravitational constant in m/s^2.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Estimated (accel_bias, gyro_bias).
        """
        if not stationary_samples:
            raise ValueError("Cannot calibrate with empty stationary observation list.")

        self.accel_bias, self.gyro_bias = estimate_static_bias(
            stationary_observations=stationary_samples,
            expected_gravity=expected_gravity,
        )
        self.is_calibrated = True

        # If ESKF is initialized, update its nominal biases
        if self._is_initialized and self.eskf.ins.is_initialized:
            st = self.eskf.ins.get_state()
            if st is not None:
                st.accelerometer_bias = self.accel_bias.copy()
                st.gyroscope_bias = self.gyro_bias.copy()

        return self.accel_bias, self.gyro_bias

    def process_imu(self, imu_obs: IMUObservation) -> Tuple[NavigationState, NavigationMode]:
        """Process a high-rate IMU observation.

        Args:
            imu_obs: Raw IMU observation.

        Returns:
            Tuple[NavigationState, NavigationMode]: Updated state and active mode.
        """
        return self.process_sample(imu_obs=imu_obs, gnss_obs=None)

    def process_gnss(self, gnss_obs: GNSSObservation) -> Tuple[NavigationState, NavigationMode]:
        """Process a low-rate GNSS observation.

        Args:
            gnss_obs: GNSS observation.

        Returns:
            Tuple[NavigationState, NavigationMode]: Updated state and active mode.
        """
        # If no IMU sample is paired, construct a dummy zero-rate observation based on latest time or timestamp
        t = gnss_obs.timestamp
        dummy_imu = IMUObservation(
            timestamp=t,
            accelerometer_x=0.0,
            accelerometer_y=0.0,
            accelerometer_z=9.80665,
            gyroscope_x=0.0,
            gyroscope_y=0.0,
            gyroscope_z=0.0,
        )
        return self.process_sample(imu_obs=dummy_imu, gnss_obs=gnss_obs)

    def process_sample(
        self,
        imu_obs: IMUObservation,
        gnss_obs: Optional[GNSSObservation] = None,
    ) -> Tuple[NavigationState, NavigationMode]:
        """Process high-rate IMU sample and optional low-rate GNSS fix.

        Args:
            imu_obs: High-rate IMU observation.
            gnss_obs: Optional GNSS observation.

        Returns:
            Tuple[NavigationState, NavigationMode]: Updated state and confirmed mode.
        """
        if not self._is_initialized:
            self.initialize(
                origin_lat=gnss_obs.latitude if gnss_obs is not None else 0.0,
                origin_lon=gnss_obs.longitude if gnss_obs is not None else 0.0,
                origin_alt=gnss_obs.altitude if gnss_obs is not None else 0.0,
            )

        # Apply static calibration if calibrated
        calibrated_imu = (
            apply_imu_calibration(imu_obs, self.accel_bias, self.gyro_bias)
            if self.is_calibrated
            else imu_obs
        )

        self._imu_sample_count += 1
        if gnss_obs is not None:
            self._gnss_sample_count += 1

        # Execute fusion step
        nav_state, status = self._fusion_engine.process_sample(
            imu_obs=calibrated_imu,
            gnss_obs=gnss_obs,
        )

        self._latest_state = nav_state
        mode = NavigationMode.from_gnss_status(status)

        if mode != self._current_mode:
            self._current_mode = mode
            self._mode_history.append((imu_obs.timestamp, mode))

        self._sample_count += 1
        return self._latest_state, self._current_mode

    def get_state(self) -> NavigationState:
        """Get the current latest navigation state."""
        if self._latest_state is None:
            return self.eskf.get_state()
        return self._latest_state

    @property
    def current_mode(self) -> NavigationMode:
        """Get current active operational mode."""
        return self._current_mode

    def get_diagnostics(self) -> Dict[str, Any]:
        """Compile a comprehensive diagnostic dictionary of the navigation pipeline.

        Returns:
            Dict[str, Any]: Detailed metrics, counter stats, covariance norms, and histories.
        """
        cov_diag = np.diag(self.eskf.P).tolist() if self.eskf is not None else []
        trace_p = float(np.trace(self.eskf.P)) if self.eskf is not None else 0.0
        pos_cov_norm = float(np.linalg.norm(cov_diag[:3])) if len(cov_diag) >= 3 else 0.0
        vel_cov_norm = float(np.linalg.norm(cov_diag[3:6])) if len(cov_diag) >= 6 else 0.0
        att_cov_norm = float(np.linalg.norm(cov_diag[6:9])) if len(cov_diag) >= 9 else 0.0

        return {
            "is_initialized": self._is_initialized,
            "is_calibrated": self.is_calibrated,
            "current_mode": self._current_mode.value,
            "sample_count": self._sample_count,
            "imu_sample_count": self._imu_sample_count,
            "gnss_sample_count": self._gnss_sample_count,
            "static_calibration": {
                "accel_bias": self.accel_bias.tolist(),
                "gyro_bias": self.gyro_bias.tolist(),
            },
            "covariance_metrics": {
                "trace_P": trace_p,
                "pos_cov_norm": pos_cov_norm,
                "vel_cov_norm": vel_cov_norm,
                "att_cov_norm": att_cov_norm,
                "covariance_diagonal": cov_diag,
            },
            "ai_fusion": {
                "enabled": self.enable_ai,
                "accepted_count": self._fusion_engine.ai_accepted_count,
                "rejected_count": self._fusion_engine.ai_rejected_count,
                "total_updates": self._fusion_engine.total_ai_updates,
                "acceptance_rate_pct": self._fusion_engine.acceptance_rate,
            },
            "motion_constraints": {
                "nhc_enabled": self.enable_nhc,
                "nhc_accepted_count": self._fusion_engine.nhc_accepted_count,
                "nhc_rejected_count": self._fusion_engine.nhc_rejected_count,
                "nhc_total_updates": self._fusion_engine.total_nhc_updates,
                "nhc_acceptance_rate_pct": self._fusion_engine.nhc_acceptance_rate,
                "zupt_enabled": self.enable_zupt,
                "zupt_accepted_count": self._fusion_engine.zupt_accepted_count,
                "zupt_rejected_count": self._fusion_engine.zupt_rejected_count,
                "zupt_total_updates": self._fusion_engine.total_zupt_updates,
                "zupt_acceptance_rate_pct": self._fusion_engine.zupt_acceptance_rate,
                "is_stationary": self.zupt_detector.is_stationary,
                "zupt_state": self.zupt_detector.state.name,
            },
            "map_matching": {
                "enabled": self.enable_map_matching,
                "accepted_count": self._fusion_engine.map_accepted_count,
                "rejected_count": self._fusion_engine.map_rejected_count,
                "total_updates": self._fusion_engine.total_map_updates,
                "acceptance_rate_pct": self._fusion_engine.map_acceptance_rate,
            },
            "mode_transitions_count": len(self._mode_history),
        }

    def reset(self) -> None:
        """Reset internal filter states, buffers, and diagnostic counters."""
        self.eskf.reset()
        self.detector.reset()
        self.zupt_detector.reset()
        self._latest_state = None
        self._current_mode = NavigationMode.GNSS_AIDED
        self._mode_history = []
        self._is_initialized = False
        self._sample_count = 0
        self._imu_sample_count = 0
        self._gnss_sample_count = 0
        self._init_fusion_engine()

    def finalize(self) -> Dict[str, Any]:
        """Finalize pipeline execution and return complete mission summary.

        Returns:
            Dict[str, Any]: Final mission summary diagnostics.
        """
        diag = self.get_diagnostics()
        diag["final_state"] = (
            {
                "timestamp": self._latest_state.timestamp,
                "position_enu": self._latest_state.position_enu.tolist(),
                "velocity_enu": self._latest_state.velocity_enu.tolist(),
                "heading_deg": self._latest_state.heading_deg(),
                "speed_mps": self._latest_state.speed,
                "quaternion": self._latest_state.orientation_quaternion.tolist(),
                "accel_bias": self._latest_state.accelerometer_bias.tolist(),
                "gyro_bias": self._latest_state.gyroscope_bias.tolist(),
            }
            if self._latest_state is not None
            else None
        )
        diag["mode_history"] = [
            {"timestamp": t, "mode": mode.value} for t, mode in self._mode_history
        ]
        return diag
