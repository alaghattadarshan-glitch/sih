"""AI-Enhanced Navigation Fusion Pipeline with Road Network Map Matching.

Orchestrates strapdown INS nominal propagation, online GNSS outage detection,
PyTorch DriftPredictor inference, causal sliding IMU window buffering, 15-state ESKF fusion,
and soft Road Network Map Matching pseudo-measurements with Mahalanobis innovation gating.
"""

import math
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

from src.data.observations import IMUObservation, GNSSObservation
from src.navigation.state import NavigationState
from src.navigation.ekf.eskf import ErrorStateKalmanFilter
from src.outage_detection.detector import GNSSOutageDetector, GNSSStatus
from src.ml.inference import DriftPredictor
from src.map_matching.types import MapMatchResult
from src.map_matching.network import RoadNetwork
from src.map_matching.matcher import MapMatcher
from src.navigation.imu.zupt_detector import ZUPTDetector, ZUPTConfig, ZUPTState


class AIESKFPipeline:
    """Integrated Causal AI-ESKF Navigation Pipeline with Motion Constraints (NHC + Safe ZUPT) and Soft Map Matching."""

    def __init__(
        self,
        eskf: ErrorStateKalmanFilter,
        detector: GNSSOutageDetector,
        predictor: Optional[DriftPredictor] = None,
        enable_ai: bool = True,
        r_ai_std: Tuple[float, float, float] = (1.5, 1.5, 3.0),
        gate_threshold: float = 4.0,
        window_size_samples: int = 100,
        network: Optional[RoadNetwork] = None,
        matcher: Optional[MapMatcher] = None,
        enable_map_matching: bool = False,
        r_map_std: Tuple[float, float, float] = (2.0, 2.0, 5.0),
        map_gate_threshold: float = 4.0,
        enable_nhc: bool = False,
        nhc_sigma_y: float = 0.1,
        nhc_sigma_z: float = 0.1,
        nhc_gate_threshold: float = 4.0,
        nhc_min_velocity_mps: float = 0.5,
        enable_zupt: bool = False,
        zupt_detector: Optional[ZUPTDetector] = None,
        zupt_sigma: float = 0.01,
        zupt_gate_threshold: float = 4.0,
    ):
        """Initialize pipeline.

        Args:
            eskf (ErrorStateKalmanFilter): 15-state ESKF engine.
            detector (GNSSOutageDetector): GNSS quality detector state machine.
            predictor (Optional[DriftPredictor]): Trained AI inference engine.
            enable_ai (bool): Enable AI pseudo-measurements during outages.
            r_ai_std (Tuple[float, float, float]): AI measurement noise std (East, North, Up).
            gate_threshold (float): Mahalanobis distance gating threshold for AI updates.
            window_size_samples (int): IMU window length (default 100 samples = 1.0s at 100Hz).
            network (Optional[RoadNetwork]): Road network topology in local ENU frame.
            matcher (Optional[MapMatcher]): Map matching candidate evaluator.
            enable_map_matching (bool): Enable soft map constraints during outages.
            r_map_std (Tuple[float, float, float]): Base map measurement noise std (East, North, Up).
            map_gate_threshold (float): Mahalanobis distance gating threshold for map updates.
            enable_nhc (bool): Enable Non-Holonomic Constraints during outages.
            nhc_sigma_y (float): NHC lateral velocity noise std dev (m/s).
            nhc_sigma_z (float): NHC vertical velocity noise std dev (m/s).
            nhc_gate_threshold (float): Mahalanobis gating threshold for NHC.
            nhc_min_velocity_mps (float): Minimum velocity threshold to apply NHC.
            enable_zupt (bool): Enable Zero-Velocity Updates when stationary during outages.
            zupt_detector (Optional[ZUPTDetector]): Stationary detection state machine.
            zupt_sigma (float): ZUPT measurement noise std dev (m/s).
            zupt_gate_threshold (float): Mahalanobis gating threshold for ZUPT.
        """
        self.eskf = eskf
        self.detector = detector
        self.predictor = predictor
        self.enable_ai = enable_ai
        self.r_ai_std = r_ai_std
        self.gate_threshold = gate_threshold
        self.window_size_samples = window_size_samples

        self.network = network
        self.matcher = matcher if matcher is not None else (MapMatcher() if network is not None else None)
        self.enable_map_matching = enable_map_matching
        self.r_map_std = r_map_std
        self.map_gate_threshold = map_gate_threshold

        self.enable_nhc = enable_nhc
        self.nhc_sigma_y = nhc_sigma_y
        self.nhc_sigma_z = nhc_sigma_z
        self.nhc_gate_threshold = nhc_gate_threshold
        self.nhc_min_velocity_mps = nhc_min_velocity_mps

        self.enable_zupt = enable_zupt
        self.zupt_detector = zupt_detector if zupt_detector is not None else ZUPTDetector()
        self.zupt_sigma = zupt_sigma
        self.zupt_gate_threshold = zupt_gate_threshold

        # Internal online state
        self.imu_buffer: List[IMUObservation] = []
        self.window_ref_pos: Optional[np.ndarray] = None
        self.window_start_time: Optional[float] = None

        # AI Diagnostic counters and logs
        self.ai_accepted_count: int = 0
        self.ai_rejected_count: int = 0
        self.mahalanobis_history: List[float] = []
        self.update_history: List[Dict[str, Any]] = []

        # Map Matching Diagnostic counters and logs
        self.map_accepted_count: int = 0
        self.map_rejected_count: int = 0
        self.map_match_history: List[MapMatchResult] = []
        self.map_update_history: List[Dict[str, Any]] = []

        # NHC Diagnostic counters and logs
        self.nhc_accepted_count: int = 0
        self.nhc_rejected_count: int = 0
        self.nhc_update_history: List[Dict[str, Any]] = []

        # ZUPT Diagnostic counters and logs
        self.zupt_accepted_count: int = 0
        self.zupt_rejected_count: int = 0
        self.zupt_update_history: List[Dict[str, Any]] = []


    def process_sample(
        self,
        imu_obs: IMUObservation,
        gnss_obs: Optional[GNSSObservation] = None,
    ) -> Tuple[NavigationState, GNSSStatus]:
        """Process high-rate IMU sample and optional GNSS fix.

        Args:
            imu_obs (IMUObservation): High-rate IMU sample.
            gnss_obs (Optional[GNSSObservation]): Optional low-rate GNSS fix.

        Returns:
            Tuple[NavigationState, GNSSStatus]: Updated navigation state and confirmed GNSS status.
        """
        current_time = imu_obs.timestamp

        # 1. Update stationary ZUPT detector state machine
        zupt_state = self.zupt_detector.update_obs(imu_obs)

        # 2. Update GNSS status state machine
        if gnss_obs is not None:
            status = self.detector.process_observation(gnss_obs, current_time=current_time)
        elif self.detector._last_obs_time is None or (current_time - self.detector._last_obs_time) > self.detector.max_gap_sec:
            status = self.detector.process_observation(None, current_time=current_time)
        else:
            status = self.detector.current_status

        # 3. INS nominal propagation and ESKF error covariance prediction step
        nav_state = self.eskf.predict(imu_obs)

        # 4. Handle GNSS available (GOOD status)
        if status == GNSSStatus.GOOD and gnss_obs is not None:
            nav_state = self.eskf.update_gnss(gnss_obs)
            # Reset window reference position and clear buffer on true GNSS update
            self.window_ref_pos = nav_state.position_enu.copy()
            self.imu_buffer.clear()
            return nav_state, status

        # 5. Handle GNSS RECOVERING status (prefer true GNSS fix, avoid stacking AI/map updates)
        if status == GNSSStatus.RECOVERING and gnss_obs is not None:
            nav_state = self.eskf.update_gnss(gnss_obs)
            self.window_ref_pos = nav_state.position_enu.copy()
            self.imu_buffer.clear()
            return nav_state, status

        # 6. During GNSS OUTAGE: Apply Vehicle Motion Constraints in causal sequence
        if status == GNSSStatus.OUTAGE:
            # A. Non-Holonomic Constraint (NHC) Update when moving/valid
            if self.enable_nhc and not self.zupt_detector.is_stationary:
                vel_before_nhc = nav_state.velocity_enu.copy()
                nav_state, nhc_accepted, nhc_mah, nhc_innov = self.eskf.update_nhc_constraint(
                    sigma_y=self.nhc_sigma_y,
                    sigma_z=self.nhc_sigma_z,
                    gate_threshold=self.nhc_gate_threshold,
                )
                vel_after_nhc = nav_state.velocity_enu.copy()

                if nhc_accepted:
                    self.nhc_accepted_count += 1
                else:
                    self.nhc_rejected_count += 1

                self.nhc_update_history.append({
                    "timestamp": current_time,
                    "innovation": nhc_innov.tolist(),
                    "innovation_norm": float(np.linalg.norm(nhc_innov)),
                    "mahalanobis_distance": nhc_mah,
                    "gate_threshold": self.nhc_gate_threshold,
                    "accepted": nhc_accepted,
                    "vel_before": vel_before_nhc.tolist(),
                    "vel_after": vel_after_nhc.tolist(),
                })

            # B. Safe Zero-Velocity Update (ZUPT) when vehicle is confirmed stationary
            if self.enable_zupt and self.zupt_detector.is_stationary:
                vel_before_zupt = nav_state.velocity_enu.copy()
                nav_state, zupt_accepted, zupt_mah, zupt_innov = self.eskf.update_zupt(
                    sigma_zupt=self.zupt_sigma,
                    gate_threshold=self.zupt_gate_threshold,
                )
                vel_after_zupt = nav_state.velocity_enu.copy()

                if zupt_accepted:
                    self.zupt_accepted_count += 1
                else:
                    self.zupt_rejected_count += 1

                self.zupt_update_history.append({
                    "timestamp": current_time,
                    "innovation": zupt_innov.tolist(),
                    "innovation_norm": float(np.linalg.norm(zupt_innov)),
                    "mahalanobis_distance": zupt_mah,
                    "gate_threshold": self.zupt_gate_threshold,
                    "accepted": zupt_accepted,
                    "vel_before": vel_before_zupt.tolist(),
                    "vel_after": vel_after_zupt.tolist(),
                    "zupt_state": self.zupt_detector.state.name,
                })

        # 7. Maintain causal sliding IMU window buffer
        if self.window_ref_pos is None:
            self.window_ref_pos = nav_state.position_enu.copy()
            self.window_start_time = current_time

        self.imu_buffer.append(imu_obs)

        # 8. Check if window is complete (100 samples) during OUTAGE
        if status == GNSSStatus.OUTAGE and len(self.imu_buffer) >= self.window_size_samples:
            # C. AI Pseudo-Measurement Update
            if self.enable_ai and self.predictor is not None:
                # Extract feature matrix [100, 8] from causal buffer
                window_matrix = self._extract_buffer_matrix(
                    self.imu_buffer[: self.window_size_samples], self.window_start_time
                )

                # AI Model Inference
                delta_p_ai = self.predictor.predict_window(window_matrix)

                pos_before = nav_state.position_enu.copy()
                z_ai = self.window_ref_pos + delta_p_ai
                innov = z_ai - pos_before

                # Perform ESKF AI Measurement Update with Mahalanobis Gating
                nav_state, accepted, mah_dist = self.eskf.update_ai_displacement(
                    ref_pos_enu=self.window_ref_pos,
                    delta_p_ai=delta_p_ai,
                    r_ai_std=self.r_ai_std,
                    gate_threshold=self.gate_threshold,
                )

                pos_after = nav_state.position_enu.copy()

                self.mahalanobis_history.append(mah_dist)
                self.update_history.append({
                    "timestamp": current_time,
                    "innovation": innov.tolist(),
                    "innovation_norm": float(np.linalg.norm(innov[:2])),
                    "mahalanobis_distance": mah_dist,
                    "gate_threshold": self.gate_threshold,
                    "accepted": accepted,
                    "pos_before": pos_before.tolist(),
                    "pos_after": pos_after.tolist(),
                })

                if accepted:
                    self.ai_accepted_count += 1
                else:
                    self.ai_rejected_count += 1

            # D. Road Network Soft Map Matching Constraint Update
            if self.enable_map_matching and self.network is not None and self.matcher is not None:
                match_res = self.matcher.match(nav_state, self.network)
                self.map_match_history.append(match_res)

                if match_res.is_matched and not match_res.gated_out and match_res.projected_position_enu is not None:
                    pos_before_map = nav_state.position_enu.copy()
                    innov_map = match_res.projected_position_enu - pos_before_map

                    nav_state, map_accepted, map_mah_dist = self.eskf.update_map_constraint(
                        projected_pos_enu=match_res.projected_position_enu,
                        confidence=match_res.confidence,
                        r_map_std_base=self.r_map_std,
                        gate_threshold=self.map_gate_threshold,
                    )
                    pos_after_map = nav_state.position_enu.copy()

                    self.map_update_history.append({
                        "timestamp": current_time,
                        "selected_segment": match_res.selected_candidate.segment.segment_id if match_res.selected_candidate else None,
                        "distance_to_road": match_res.distance_to_road_m,
                        "heading_error": match_res.heading_error_deg,
                        "confidence": match_res.confidence,
                        "innovation": innov_map.tolist(),
                        "innovation_norm": float(np.linalg.norm(innov_map[:2])),
                        "mahalanobis_distance": map_mah_dist,
                        "gate_threshold": self.map_gate_threshold,
                        "accepted": map_accepted,
                        "pos_before": pos_before_map.tolist(),
                        "pos_after": pos_after_map.tolist(),
                    })

                    if map_accepted:
                        self.map_accepted_count += 1
                    else:
                        self.map_rejected_count += 1
                else:
                    self.map_rejected_count += 1

            # Reset window reference position for next window
            self.window_ref_pos = nav_state.position_enu.copy()
            # Clear processed window samples
            self.imu_buffer = self.imu_buffer[self.window_size_samples :]
            self.window_start_time = (
                self.imu_buffer[0].timestamp if self.imu_buffer else current_time
            )

        return nav_state, status


    def _extract_buffer_matrix(
        self, imu_list: List[IMUObservation], t_start_base: Optional[float]
    ) -> np.ndarray:
        """Extract raw feature matrix [100, 8] from IMU observation list."""
        raw_rows = []
        base_t = t_start_base if t_start_base is not None else imu_list[0].timestamp

        for obs in imu_list:
            ax, ay, az = obs.accelerometer_x, obs.accelerometer_y, obs.accelerometer_z
            gx, gy, gz = obs.gyroscope_x, obs.gyroscope_y, obs.gyroscope_z
            a_norm = math.sqrt(ax**2 + ay**2 + az**2)
            g_norm = math.sqrt(gx**2 + gy**2 + gz**2)
            rel_time = obs.timestamp - base_t

            raw_rows.append([ax, ay, az, gx, gy, gz, a_norm, g_norm, rel_time])

        arr = np.array(raw_rows, dtype=np.float32)
        return arr[:, :8]

    @property
    def total_ai_updates(self) -> int:
        """Total AI update attempts."""
        return self.ai_accepted_count + self.ai_rejected_count

    @property
    def acceptance_rate(self) -> float:
        """Percentage of AI updates accepted by innovation gating."""
        total = self.total_ai_updates
        return (self.ai_accepted_count / total * 100.0) if total > 0 else 0.0

    @property
    def total_map_updates(self) -> int:
        """Total map constraint update attempts."""
        return self.map_accepted_count + self.map_rejected_count

    @property
    def map_acceptance_rate(self) -> float:
        """Percentage of map updates accepted by innovation gating."""
        total = self.total_map_updates
        return (self.map_accepted_count / total * 100.0) if total > 0 else 0.0

    @property
    def total_nhc_updates(self) -> int:
        """Total NHC update attempts."""
        return self.nhc_accepted_count + self.nhc_rejected_count

    @property
    def nhc_acceptance_rate(self) -> float:
        """Percentage of NHC updates accepted by innovation gating."""
        total = self.total_nhc_updates
        return (self.nhc_accepted_count / total * 100.0) if total > 0 else 0.0

    @property
    def total_zupt_updates(self) -> int:
        """Total ZUPT update attempts."""
        return self.zupt_accepted_count + self.zupt_rejected_count

    @property
    def zupt_acceptance_rate(self) -> float:
        """Percentage of ZUPT updates accepted by innovation gating."""
        total = self.total_zupt_updates
        return (self.zupt_accepted_count / total * 100.0) if total > 0 else 0.0

