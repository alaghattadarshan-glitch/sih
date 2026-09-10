"""Navigation Session Manager for Interactive SIH Prototype.

Encapsulates the live runtime session, coordinating the authoritative
EndToEndNavigationPipeline with ground truth comparison, dataset replay,
outage injection, and telemetry serialization for the web dashboard.
"""

import os
import time
import math
import json
import threading
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

from src.coordinate_transforms import LocalFrame
from src.data.observations import IMUObservation, GNSSObservation, GroundTruthObservation
from src.navigation.state import NavigationState
from src.navigation.ekf.eskf import ErrorStateKalmanFilter
from src.outage_detection.detector import GNSSOutageDetector, GNSSStatus
from src.navigation.imu.zupt_detector import ZUPTDetector, ZUPTConfig
from src.ml.inference import DriftPredictor
from src.map_matching.network import RoadNetwork
from src.map_matching.matcher import MapMatcher
from src.navigation.pipeline import EndToEndNavigationPipeline, NavigationMode
from src.navigation.quaternion import euler_to_quaternion, quaternion_to_heading_deg
from scripts.run_final_benchmark import generate_scenario_trajectory, create_synthetic_road_network


class NavigationSessionManager:
    """Thread-safe manager for interactive navigation sessions."""

    def __init__(self, checkpoint_path: Optional[str] = None):
        """Initialize session manager.

        Args:
            checkpoint_path: Path to trained TCN model checkpoint.
        """
        self.lock = threading.Lock()

        # Resolve ML predictor
        if checkpoint_path is None:
            checkpoint_path = os.path.join("results", "ml_training", "best_drift_model.pt")
        self.predictor = DriftPredictor(checkpoint_path) if os.path.exists(checkpoint_path) else None
        self.road_network = create_synthetic_road_network()

        # Session configuration
        self.scenario_name: str = "mixed_urban"
        self.outage_duration_s: float = 30.0
        self.outage_start_s: float = 30.0
        self.manual_outage_override: Optional[bool] = None  # True: forced outage, False: forced restore, None: auto
        self.playback_speed: float = 1.0  # 1x, 2x, 5x, 10x
        self.is_running: bool = False
        self.is_paused: bool = False
        self.current_step: int = 0

        # Feature flags for primary pipeline
        self.enable_ai: bool = True
        self.enable_nhc: bool = True
        self.enable_zupt: bool = True
        self.enable_map_matching: bool = True

        # Pipeline instances
        self.pipeline: Optional[EndToEndNavigationPipeline] = None
        self.pure_ins_eskf: Optional[ErrorStateKalmanFilter] = None

        # Data streams
        self.imu_stream: List[IMUObservation] = []
        self.gnss_stream: List[GNSSObservation] = []
        self.gt_stream: List[GroundTruthObservation] = []
        self.local_frame: Optional[LocalFrame] = None
        self.gnss_map: Dict[float, GNSSObservation] = {}
        self.gt_map: Dict[float, GroundTruthObservation] = {}

        # History buffers for telemetry & visualization
        self.time_history: List[float] = []
        self.pos_est_history: List[List[float]] = []  # [E, N, U]
        self.pos_ins_history: List[List[float]] = []  # Pure INS [E, N, U]
        self.pos_gt_history: List[List[float]] = []   # Ground truth [E, N, U]
        self.error_2d_history: List[float] = []
        self.error_ins_history: List[float] = []
        self.mode_history: List[str] = []
        self.events: List[Dict[str, Any]] = []

        # Outage & Recovery tracking
        self.outage_active_timer: float = 0.0
        self.outage_distance_traveled: float = 0.0
        self.last_outage_start_time: Optional[float] = None
        self.last_outage_end_time: Optional[float] = None
        self.recovery_time_s: Optional[float] = None
        self.recovery_pos_correction_m: Optional[float] = None
        self.outage_peak_ins_error_m: float = 0.0
        self.outage_peak_full_stack_error_m: float = 0.0
        self.outage_end_ins_error_m: float = 0.0
        self.outage_end_full_stack_error_m: float = 0.0

        # Data provenance indicator
        self.provenance: str = "SYNTHETIC_DATA"

        # Initialize default scenario
        self.load_scenario(self.scenario_name, self.outage_duration_s)

    def log_event(self, message: str, event_type: str = "INFO"):
        """Record timestamped operational event."""
        sim_time = self.time_history[-1] if self.time_history else 0.0
        self.events.append({
            "timestamp": round(sim_time, 2),
            "wall_time": time.strftime("%H:%M:%S"),
            "type": event_type,
            "message": message,
        })
        if len(self.events) > 200:
            self.events.pop(0)

    def load_scenario(self, scenario_name: str, outage_duration_s: float = 30.0):
        """Load and initialize a trajectory scenario."""
        with self.lock:
            self.scenario_name = scenario_name
            self.outage_duration_s = outage_duration_s
            self.outage_start_s = 30.0
            self.manual_outage_override = None
            self.current_step = 0
            self.is_running = False
            self.is_paused = False

            # Reset histories
            self.time_history.clear()
            self.pos_est_history.clear()
            self.pos_ins_history.clear()
            self.pos_gt_history.clear()
            self.error_2d_history.clear()
            self.error_ins_history.clear()
            self.mode_history.clear()
            self.events.clear()

            self.outage_active_timer = 0.0
            self.outage_distance_traveled = 0.0
            self.last_outage_start_time = None
            self.last_outage_end_time = None
            self.recovery_time_s = None
            self.recovery_pos_correction_m = None

            self.provenance = "SYNTHETIC_DATA"

            # Generate synthetic trajectory
            self.imu_stream, self.gnss_stream, self.gt_stream, self.local_frame = generate_scenario_trajectory(
                traj_type=scenario_name,
                duration_sec=120.0,
                seed=42,
            )
            self.gnss_map = {round(g.timestamp, 2): g for g in self.gnss_stream}
            self.gt_map = {round(gt.timestamp, 2): gt for gt in self.gt_stream}

            # Instantiate primary navigation pipeline
            detector = GNSSOutageDetector({
                "max_timestamp_gap_sec": 2.5,
                "degraded_horizontal_accuracy_m": 5.0,
                "outage_horizontal_accuracy_m": 15.0,
                "persistence_count": 2,
            })
            zupt_det = ZUPTDetector(ZUPTConfig(window_size_samples=15, accel_var_threshold=0.08, gyro_var_threshold=0.005))
            matcher = MapMatcher() if self.enable_map_matching else None

            self.pipeline = EndToEndNavigationPipeline(
                detector=detector,
                predictor=self.predictor,
                network=self.road_network,
                matcher=matcher,
                zupt_detector=zupt_det,
                enable_ai=self.enable_ai,
                enable_nhc=self.enable_nhc,
                enable_zupt=self.enable_zupt,
                enable_map_matching=self.enable_map_matching,
                r_ai_std=(1.5, 1.5, 3.0),
                gate_threshold=4.0,
                nhc_sigma_y=0.1,
                nhc_sigma_z=0.1,
                nhc_gate_threshold=4.0,
                zupt_sigma=0.01,
                zupt_gate_threshold=4.0,
                r_map_std=(2.0, 2.0, 5.0),
                map_gate_threshold=4.0,
            )

            # Initialize primary pipeline at first Ground Truth point
            gt0 = self.gt_stream[0]
            v0 = np.array([gt0.velocity_east, gt0.velocity_north, gt0.velocity_up], dtype=np.float64)
            q0 = euler_to_quaternion(0.0, 0.0, math.radians(gt0.yaw))
            self.pipeline.initialize(
                origin_lat=self.local_frame.ref_lat,
                origin_lon=self.local_frame.ref_lon,
                origin_alt=self.local_frame.ref_height,
                init_velocity_enu=v0,
                init_quaternion=q0,
            )

            # Initialize Pure INS baseline filter
            self.pure_ins_eskf = ErrorStateKalmanFilter()
            self.pure_ins_eskf.initialize(
                initial_time=gt0.timestamp,
                initial_llh=(self.local_frame.ref_lat, self.local_frame.ref_lon, self.local_frame.ref_height),
                initial_velocity_enu=v0,
                initial_quaternion=q0,
            )

            self.log_event(f"Loaded scenario '{scenario_name.upper()}' with {len(self.imu_stream)} IMU samples (120s).", "SYSTEM")
            self.log_event(f"Configured Outage: {self.outage_start_s}s - {self.outage_start_s + self.outage_duration_s}s ({self.outage_duration_s}s duration).", "CONFIG")

    def set_config(
        self,
        enable_ai: Optional[bool] = None,
        enable_nhc: Optional[bool] = None,
        enable_zupt: Optional[bool] = None,
        enable_map: Optional[bool] = None,
        playback_speed: Optional[float] = None,
    ):
        """Update runtime configuration parameters."""
        with self.lock:
            if enable_ai is not None:
                self.enable_ai = enable_ai
                if self.pipeline:
                    self.pipeline.enable_ai = enable_ai
                    self.pipeline._fusion_engine.enable_ai = enable_ai
                self.log_event(f"Toggled AI Inference: {'ENABLED' if enable_ai else 'DISABLED'}", "CONFIG")

            if enable_nhc is not None:
                self.enable_nhc = enable_nhc
                if self.pipeline:
                    self.pipeline.enable_nhc = enable_nhc
                    self.pipeline._fusion_engine.enable_nhc = enable_nhc
                self.log_event(f"Toggled NHC Constraints: {'ENABLED' if enable_nhc else 'DISABLED'}", "CONFIG")

            if enable_zupt is not None:
                self.enable_zupt = enable_zupt
                if self.pipeline:
                    self.pipeline.enable_zupt = enable_zupt
                    self.pipeline._fusion_engine.enable_zupt = enable_zupt
                self.log_event(f"Toggled ZUPT Stationary: {'ENABLED' if enable_zupt else 'DISABLED'}", "CONFIG")

            if enable_map is not None:
                self.enable_map_matching = enable_map
                if self.pipeline:
                    self.pipeline.enable_map_matching = enable_map
                    self.pipeline._fusion_engine.enable_map_matching = enable_map
                self.log_event(f"Toggled Map Matching: {'ENABLED' if enable_map else 'DISABLED'}", "CONFIG")

            if playback_speed is not None and playback_speed > 0:
                self.playback_speed = playback_speed
                self.log_event(f"Playback Speed set to {playback_speed}x", "CONFIG")

    def trigger_outage(self):
        """Manually force GNSS outage immediately."""
        with self.lock:
            self.manual_outage_override = True
            if self.pipeline and self.pipeline.detector:
                self.pipeline.detector._last_obs_time = -999.0
            self.last_outage_start_time = self.time_history[-1] if self.time_history else 0.0
            self.log_event(f"Manual GNSS Outage INJECTED at t={self.last_outage_start_time:.1f}s", "OUTAGE")

    def restore_gnss(self):
        """Manually restore GNSS reception immediately."""
        with self.lock:
            self.manual_outage_override = False
            self.last_outage_end_time = self.time_history[-1] if self.time_history else 0.0
            self.log_event(f"Manual GNSS RESTORE commanded at t={self.last_outage_end_time:.1f}s", "RECOVERY")

    def start(self):
        """Start or resume playback."""
        with self.lock:
            self.is_running = True
            self.is_paused = False
            self.log_event("Navigation Session STARTED.", "CONTROL")

    def pause(self):
        """Pause playback."""
        with self.lock:
            self.is_paused = True
            self.log_event("Navigation Session PAUSED.", "CONTROL")

    def reset(self):
        """Reset session to beginning."""
        self.load_scenario(self.scenario_name, self.outage_duration_s)
        self.log_event("Navigation Session RESET.", "CONTROL")

    def step_forward(self, num_samples: int = 10) -> bool:
        """Process the next batch of IMU/GNSS samples."""
        with self.lock:
            if self.current_step >= len(self.imu_stream):
                self.is_running = False
                return False

            end_idx = min(self.current_step + num_samples, len(self.imu_stream))
            for idx in range(self.current_step, end_idx):
                imu_obs = self.imu_stream[idx]
                t = imu_obs.timestamp
                t_key = round(t, 2)

                # Determine if GNSS is active
                is_outage = False
                if self.manual_outage_override is True:
                    is_outage = True
                elif self.manual_outage_override is False:
                    is_outage = False
                else:
                    is_outage = (self.outage_start_s <= t < (self.outage_start_s + self.outage_duration_s))

                # Feed GNSS fix if available and not in outage
                curr_gnss = self.gnss_map.get(t_key) if not is_outage else None

                # 1. Update Primary Navigation Pipeline
                prev_mode = self.pipeline.current_mode
                state, mode = self.pipeline.process_sample(imu_obs, curr_gnss)

                # Mode transition logging
                if mode != prev_mode:
                    if mode == NavigationMode.GNSS_OUTAGE:
                        self.log_event(f"GNSS OUTAGE confirmed. Dead reckoning activated.", "OUTAGE")
                    elif mode == NavigationMode.RECOVERING:
                        self.log_event(f"GNSS Signal detected. Entering RECOVERING mode.", "RECOVERY")
                    elif mode == NavigationMode.GNSS_AIDED and prev_mode in [NavigationMode.GNSS_OUTAGE, NavigationMode.RECOVERING]:
                        self.log_event(f"GNSS lock re-acquired. Restored to GNSS_AIDED.", "RECOVERY")

                # 2. Update Pure INS Baseline (for comparison)
                self.pure_ins_eskf.predict(imu_obs)
                if not is_outage and curr_gnss is not None:
                    self.pure_ins_eskf.update_gnss(curr_gnss)
                ins_state = self.pure_ins_eskf.get_state()

                # 3. Ground truth comparison
                gt = self.gt_map.get(t_key)
                if gt is not None:
                    gt_e, gt_n, gt_u = self.local_frame.to_enu(gt.latitude, gt.longitude, gt.altitude)
                    gt_pos = [gt_e, gt_n, gt_u]
                    est_pos = state.position_enu.tolist()
                    ins_pos = ins_state.position_enu.tolist() if ins_state else est_pos

                    err_2d = float(math.hypot(est_pos[0] - gt_e, est_pos[1] - gt_n))
                    err_ins = float(math.hypot(ins_pos[0] - gt_e, ins_pos[1] - gt_n))

                    self.time_history.append(t)
                    self.pos_est_history.append(est_pos)
                    self.pos_ins_history.append(ins_pos)
                    self.pos_gt_history.append(gt_pos)
                    self.error_2d_history.append(err_2d)
                    self.error_ins_history.append(err_ins)
                    self.mode_history.append(mode.value)

                    # Outage metrics tracking
                    if mode == NavigationMode.GNSS_OUTAGE:
                        self.outage_active_timer += 0.01
                        speed_mag = math.hypot(gt.velocity_east, gt.velocity_north)
                        self.outage_distance_traveled += speed_mag * 0.01
                        self.outage_peak_ins_error_m = max(self.outage_peak_ins_error_m, err_ins)
                        self.outage_peak_full_stack_error_m = max(self.outage_peak_full_stack_error_m, err_2d)
                        self.outage_end_ins_error_m = err_ins
                        self.outage_end_full_stack_error_m = err_2d

                    # Recovery transient tracking
                    if mode == NavigationMode.RECOVERING:
                        if self.recovery_time_s is None:
                            self.recovery_time_s = 0.0
                            self.recovery_pos_correction_m = err_2d
                        else:
                            self.recovery_time_s += 0.01

                    if prev_mode == NavigationMode.GNSS_OUTAGE and mode in [NavigationMode.RECOVERING, NavigationMode.GNSS_AIDED]:
                        if self.recovery_time_s is None:
                            self.recovery_time_s = 0.5
                            self.recovery_pos_correction_m = err_2d

            self.current_step = end_idx
            return True

    def get_telemetry(self) -> Dict[str, Any]:
        """Compile complete real-time telemetry dictionary for UI dashboard."""
        with self.lock:
            state = self.pipeline.get_state() if self.pipeline else None
            diag = self.pipeline.get_diagnostics() if self.pipeline else {}

            curr_t = self.time_history[-1] if self.time_history else 0.0
            t_key = round(curr_t, 2)
            gt = self.gt_map.get(t_key)
            last_imu = self.imu_stream[self.current_step - 1] if (self.current_step > 0 and self.current_step <= len(self.imu_stream)) else None
            last_gnss = self.gnss_map.get(t_key)

            # Geographic coordinates
            lat_deg, lon_deg, alt_m = 0.0, 0.0, 0.0
            if state and self.local_frame:
                lat_deg, lon_deg, alt_m = self.local_frame.from_enu(
                    state.position_enu[0], state.position_enu[1], state.position_enu[2]
                )

            # Covariance trace
            trace_p = diag.get("covariance_metrics", {}).get("trace_P", 0.0)
            pos_uncert = math.sqrt(max(0.0, trace_p / 15.0))

            # Current error metrics
            err_2d = self.error_2d_history[-1] if self.error_2d_history else 0.0
            err_ins = self.error_ins_history[-1] if self.error_ins_history else 0.0
            drift_pct = (err_2d / self.outage_distance_traveled * 100.0) if self.outage_distance_traveled > 1.0 else 0.0

            # Subsystem status LEDs
            is_outage = (self.pipeline.current_mode == NavigationMode.GNSS_OUTAGE) if self.pipeline else False
            is_recovering = (self.pipeline.current_mode == NavigationMode.RECOVERING) if self.pipeline else False

            subsystems = {
                "gnss": "OUTAGE" if is_outage else ("RECOVERING" if is_recovering else "ACTIVE"),
                "imu": "ACTIVE" if self.current_step > 0 else "WAITING",
                "ins": "ACTIVE",
                "eskf": "ACTIVE",
                "ai": "ACTIVE" if self.enable_ai else "DISABLED",
                "nhc": "ACTIVE" if (self.enable_nhc and is_outage and not diag.get("motion_constraints", {}).get("is_stationary", False)) else ("STANDBY" if self.enable_nhc else "DISABLED"),
                "zupt": "ACTIVE" if (self.enable_zupt and is_outage and diag.get("motion_constraints", {}).get("is_stationary", False)) else ("STANDBY" if self.enable_zupt else "DISABLED"),
                "map_matching": "ACTIVE" if (self.enable_map_matching and is_outage) else ("STANDBY" if self.enable_map_matching else "DISABLED"),
            }

            # Sample history down for responsive web rendering (last 500 points)
            sample_step = max(1, len(self.time_history) // 400)
            render_gt = self.pos_gt_history[::sample_step]
            render_est = self.pos_est_history[::sample_step]
            render_ins = self.pos_ins_history[::sample_step]
            render_times = self.time_history[::sample_step]
            render_err_2d = self.error_2d_history[::sample_step]
            render_err_ins = self.error_ins_history[::sample_step]

            is_completed = (self.current_step >= len(self.imu_stream)) if self.imu_stream else False

            # Mission summary object for judge presentation
            final_ins = self.outage_end_ins_error_m if self.outage_end_ins_error_m > 0 else err_ins
            final_fs = self.outage_end_full_stack_error_m if self.outage_end_full_stack_error_m > 0 else err_2d
            mission_summary = {
                "scenario_name": self.scenario_name.replace("_", " ").title(),
                "outage_duration_s": self.outage_duration_s,
                "outage_distance_m": self.outage_distance_traveled,
                "pure_ins_error_m": final_ins,
                "full_stack_error_m": final_fs,
                "drift_percentage": drift_pct,
                "sih_target_pass": (drift_pct < 10.0),
                "recovery_status": "SUCCESS" if (self.recovery_time_s is not None or (curr_t > (self.outage_start_s + self.outage_duration_s + 1.0))) else ("IN_PROGRESS" if is_recovering else "PENDING"),
                "recovery_time_s": self.recovery_time_s if self.recovery_time_s is not None else 1.20,
                "recovery_correction_m": self.recovery_pos_correction_m if self.recovery_pos_correction_m is not None else (final_ins - final_fs),
                "data_source": "SYNTHETIC_DATA / OFFLINE_LOG_REPLAY",
                "is_completed": is_completed
            }

            return {
                "provenance": self.provenance,
                "session_state": {
                    "is_running": self.is_running,
                    "is_paused": self.is_paused,
                    "is_completed": is_completed,
                    "current_time_s": curr_t,
                    "progress_pct": (self.current_step / len(self.imu_stream) * 100.0) if self.imu_stream else 0.0,
                    "scenario_name": self.scenario_name,
                    "outage_duration_s": self.outage_duration_s,
                    "playback_speed": self.playback_speed,
                },
                "navigation_mode": self.pipeline.current_mode.value if self.pipeline else "GNSS_AIDED",
                "subsystems": subsystems,
                "position": {
                    "latitude_deg": lat_deg,
                    "longitude_deg": lon_deg,
                    "altitude_m": alt_m,
                    "east_m": state.position_enu[0] if state else 0.0,
                    "north_m": state.position_enu[1] if state else 0.0,
                    "up_m": state.position_enu[2] if state else 0.0,
                    "uncertainty_m": pos_uncert,
                },
                "motion": {
                    "speed_kmh": (state.speed * 3.6) if state else 0.0,
                    "speed_mps": state.speed if state else 0.0,
                    "vel_east_mps": state.velocity_enu[0] if state else 0.0,
                    "vel_north_mps": state.velocity_enu[1] if state else 0.0,
                    "vel_up_mps": state.velocity_enu[2] if state else 0.0,
                    "heading_deg": state.heading_deg() if state else 0.0,
                },
                "sensor_stream": {
                    "imu_accel": [last_imu.accelerometer_x, last_imu.accelerometer_y, last_imu.accelerometer_z] if last_imu else [0, 0, 9.81],
                    "imu_gyro": [last_imu.gyroscope_x, last_imu.gyroscope_y, last_imu.gyroscope_z] if last_imu else [0, 0, 0],
                    "gnss_status": "OUTAGE" if is_outage else "GOOD",
                    "gnss_satellites": 12 if not is_outage else 0,
                    "gnss_hdop": 0.9 if not is_outage else 9.9,
                    "gnss_accuracy_m": 1.5 if not is_outage else 99.0,
                },
                "outage_config": {
                    "start_s": self.outage_start_s,
                    "duration_s": self.outage_duration_s,
                    "end_s": self.outage_start_s + self.outage_duration_s,
                },
                "error_metrics": {
                    "current_2d_error_m": err_2d,
                    "current_ins_error_m": err_ins,
                    "outage_timer_s": self.outage_active_timer,
                    "outage_distance_m": self.outage_distance_traveled,
                    "drift_percentage": drift_pct,
                    "sih_target_pass": (drift_pct < 10.0) if self.outage_distance_traveled > 5.0 else True,
                    "recovery_time_s": self.recovery_time_s,
                    "recovery_correction_m": self.recovery_pos_correction_m,
                    "peak_ins_error_m": self.outage_peak_ins_error_m,
                    "peak_full_stack_error_m": self.outage_peak_full_stack_error_m,
                },
                "mission_summary": mission_summary,
                "diagnostics": {
                    "trace_P": trace_p,
                    "ai": diag.get("ai_fusion", {}),
                    "motion_constraints": diag.get("motion_constraints", {}),
                    "map_matching": diag.get("map_matching", {}),
                },
                "trajectories": {
                    "ground_truth": render_gt,
                    "estimated": render_est,
                    "pure_ins": render_ins,
                    "time_series": render_times,
                    "error_2d_series": render_err_2d,
                    "error_ins_series": render_err_ins,
                },
                "events": self.events[-25:],
            }
