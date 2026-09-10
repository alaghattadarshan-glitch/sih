"""Python vs C++ Edge Navigation Core Parity and Benchmark Evaluator.

Connects to the compiled C++ shared library via ctypes, executes identical
IMU/GNSS sensor observations through both the Python reference ESKF and the C++
engine, and measures numerical differences, gating decisions, and latency statistics.
"""

import ctypes
import os
import sys
import time
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from src.data.observations import IMUObservation, GNSSObservation
from src.navigation.ekf.eskf import ErrorStateKalmanFilter
from src.navigation.quaternion import quaternion_to_rotation_matrix


class CTypesNavCore:
    """Python ctypes wrapper for C++ Navigation Core shared library."""

    def __init__(self, lib_path: Optional[str] = None, g_val: float = 9.80665):
        if lib_path is None:
            # Auto-locate built library
            root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
            candidates = [
                os.path.join(root, "android/native/build/libnav_core.dylib"),
                os.path.join(root, "android/native/build/libnav_core.so"),
            ]
            for c in candidates:
                if os.path.exists(c):
                    lib_path = c
                    break

        if lib_path is None or not os.path.exists(lib_path):
            raise FileNotFoundError(f"Cannot find compiled navigation core library at: {lib_path}")

        self.lib = ctypes.CDLL(lib_path)
        self._setup_function_signatures()
        self.handle = self.lib.nav_core_create(ctypes.c_double(g_val))

    def _setup_function_signatures(self):
        # nav_core_create
        self.lib.nav_core_create.argtypes = [ctypes.c_double]
        self.lib.nav_core_create.restype = ctypes.c_void_p

        # nav_core_destroy
        self.lib.nav_core_destroy.argtypes = [ctypes.c_void_p]
        self.lib.nav_core_destroy.restype = None

        # nav_core_initialize
        self.lib.nav_core_initialize.argtypes = [
            ctypes.c_void_p,
            ctypes.c_double,
            ctypes.c_double, ctypes.c_double, ctypes.c_double,
            ctypes.c_double, ctypes.c_double, ctypes.c_double,
            ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
            ctypes.c_double, ctypes.c_double, ctypes.c_double,
            ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ]
        self.lib.nav_core_initialize.restype = ctypes.c_int

        # nav_core_process_imu
        self.lib.nav_core_process_imu.argtypes = [
            ctypes.c_void_p,
            ctypes.c_double,
            ctypes.c_double, ctypes.c_double, ctypes.c_double,
            ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ]
        self.lib.nav_core_process_imu.restype = ctypes.c_int

        # nav_core_process_gnss
        self.lib.nav_core_process_gnss.argtypes = [
            ctypes.c_void_p,
            ctypes.c_double,
            ctypes.c_double, ctypes.c_double, ctypes.c_double,
            ctypes.c_double, ctypes.c_double,
            ctypes.c_double, ctypes.c_double,
        ]
        self.lib.nav_core_process_gnss.restype = ctypes.c_int

        # nav_core_process_ai_displacement
        self.lib.nav_core_process_ai_displacement.argtypes = [
            ctypes.c_void_p,
            ctypes.c_double, ctypes.c_double, ctypes.c_double,
            ctypes.c_double, ctypes.c_double, ctypes.c_double,
            ctypes.c_double, ctypes.c_double, ctypes.c_double,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_double),
        ]
        self.lib.nav_core_process_ai_displacement.restype = ctypes.c_int

        # nav_core_process_nhc
        self.lib.nav_core_process_nhc.argtypes = [
            ctypes.c_void_p,
            ctypes.c_double, ctypes.c_double, ctypes.c_double,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_double),
        ]
        self.lib.nav_core_process_nhc.restype = ctypes.c_int

        # nav_core_process_zupt
        self.lib.nav_core_process_zupt.argtypes = [
            ctypes.c_void_p,
            ctypes.c_double, ctypes.c_double,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_double),
        ]
        self.lib.nav_core_process_zupt.restype = ctypes.c_int

        # nav_core_set_motion_constraints
        self.lib.nav_core_set_motion_constraints.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.lib.nav_core_set_motion_constraints.restype = ctypes.c_int

        # nav_core_get_zupt_state
        self.lib.nav_core_get_zupt_state.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        self.lib.nav_core_get_zupt_state.restype = ctypes.c_int

        # nav_core_get_state
        self.lib.nav_core_get_state.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_double),
        ]
        self.lib.nav_core_get_state.restype = ctypes.c_int

        # nav_core_extract_ai_features
        self.lib.nav_core_extract_ai_features.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
        ]
        self.lib.nav_core_extract_ai_features.restype = ctypes.c_int

        # nav_core_reset
        self.lib.nav_core_reset.argtypes = [ctypes.c_void_p]
        self.lib.nav_core_reset.restype = ctypes.c_int

    def initialize(
        self,
        t0: float,
        lat_deg: float, lon_deg: float, alt_m: float,
        vel_enu: np.ndarray,
        quat: np.ndarray,
        accel_bias: Optional[np.ndarray] = None,
        gyro_bias: Optional[np.ndarray] = None,
    ):
        ab = accel_bias if accel_bias is not None else np.zeros(3)
        gb = gyro_bias if gyro_bias is not None else np.zeros(3)

        self.lib.nav_core_initialize(
            self.handle,
            ctypes.c_double(t0),
            ctypes.c_double(lat_deg), ctypes.c_double(lon_deg), ctypes.c_double(alt_m),
            ctypes.c_double(vel_enu[0]), ctypes.c_double(vel_enu[1]), ctypes.c_double(vel_enu[2]),
            ctypes.c_double(quat[0]), ctypes.c_double(quat[1]), ctypes.c_double(quat[2]), ctypes.c_double(quat[3]),
            ctypes.c_double(ab[0]), ctypes.c_double(ab[1]), ctypes.c_double(ab[2]),
            ctypes.c_double(gb[0]), ctypes.c_double(gb[1]), ctypes.c_double(gb[2]),
        )

    def process_imu(self, t: float, ax: float, ay: float, az: float, gx: float, gy: float, gz: float):
        self.lib.nav_core_process_imu(
            self.handle,
            ctypes.c_double(t),
            ctypes.c_double(ax), ctypes.c_double(ay), ctypes.c_double(az),
            ctypes.c_double(gx), ctypes.c_double(gy), ctypes.c_double(gz),
        )

    def process_gnss(
        self,
        t: float,
        lat: float, lon: float, alt: float,
        h_acc: float = -1.0, v_acc: float = -1.0,
        hdop: float = -1.0, cn0: float = -1.0
    ):
        self.lib.nav_core_process_gnss(
            self.handle,
            ctypes.c_double(t),
            ctypes.c_double(lat), ctypes.c_double(lon), ctypes.c_double(alt),
            ctypes.c_double(h_acc), ctypes.c_double(v_acc),
            ctypes.c_double(hdop), ctypes.c_double(cn0),
        )

    def process_ai_displacement(
        self,
        ref_enu: np.ndarray,
        delta_ai: np.ndarray,
        r_std: Tuple[float, float, float] = (1.5, 1.5, 3.0)
    ) -> Tuple[bool, float]:
        acc_c = ctypes.c_int(0)
        mah_c = ctypes.c_double(0.0)

        self.lib.nav_core_process_ai_displacement(
            self.handle,
            ctypes.c_double(ref_enu[0]), ctypes.c_double(ref_enu[1]), ctypes.c_double(ref_enu[2]),
            ctypes.c_double(delta_ai[0]), ctypes.c_double(delta_ai[1]), ctypes.c_double(delta_ai[2]),
            ctypes.c_double(r_std[0]), ctypes.c_double(r_std[1]), ctypes.c_double(r_std[2]),
            ctypes.byref(acc_c),
            ctypes.byref(mah_c),
        )
        return bool(acc_c.value), float(mah_c.value)

    def process_nhc(
        self,
        sigma_y: float = 0.1,
        sigma_z: float = 0.1,
        gate_threshold: float = 4.0
    ) -> Tuple[bool, float]:
        acc_c = ctypes.c_int(0)
        mah_c = ctypes.c_double(0.0)

        self.lib.nav_core_process_nhc(
            self.handle,
            ctypes.c_double(sigma_y),
            ctypes.c_double(sigma_z),
            ctypes.c_double(gate_threshold),
            ctypes.byref(acc_c),
            ctypes.byref(mah_c),
        )
        return bool(acc_c.value), float(mah_c.value)

    def process_zupt(
        self,
        sigma_zupt: float = 0.01,
        gate_threshold: float = 4.0
    ) -> Tuple[bool, float]:
        acc_c = ctypes.c_int(0)
        mah_c = ctypes.c_double(0.0)

        self.lib.nav_core_process_zupt(
            self.handle,
            ctypes.c_double(sigma_zupt),
            ctypes.c_double(gate_threshold),
            ctypes.byref(acc_c),
            ctypes.byref(mah_c),
        )
        return bool(acc_c.value), float(mah_c.value)

    def set_motion_constraints(self, enable_nhc: bool, enable_zupt: bool):
        self.lib.nav_core_set_motion_constraints(
            self.handle,
            ctypes.c_int(1 if enable_nhc else 0),
            ctypes.c_int(1 if enable_zupt else 0),
        )

    def get_zupt_state(self) -> Tuple[int, bool]:
        state_c = ctypes.c_int(0)
        stat_c = ctypes.c_int(0)
        self.lib.nav_core_get_zupt_state(
            self.handle,
            ctypes.byref(state_c),
            ctypes.byref(stat_c),
        )
        return int(state_c.value), bool(stat_c.value)


    def get_state(self) -> Dict[str, Any]:
        t_c = ctypes.c_double(0.0)
        lat_c = ctypes.c_double(0.0)
        lon_c = ctypes.c_double(0.0)
        alt_c = ctypes.c_double(0.0)
        pos_c = (ctypes.c_double * 3)()
        vel_c = (ctypes.c_double * 3)()
        quat_c = (ctypes.c_double * 4)()
        heading_c = ctypes.c_double(0.0)
        status_c = ctypes.c_int(0)
        conf_c = ctypes.c_double(0.0)

        self.lib.nav_core_get_state(
            self.handle,
            ctypes.byref(t_c),
            ctypes.byref(lat_c), ctypes.byref(lon_c), ctypes.byref(alt_c),
            pos_c,
            vel_c,
            quat_c,
            ctypes.byref(heading_c),
            ctypes.byref(status_c),
            ctypes.byref(conf_c),
        )

        return {
            "timestamp": t_c.value,
            "latitude": lat_c.value,
            "longitude": lon_c.value,
            "altitude": alt_c.value,
            "pos_enu": np.array([pos_c[0], pos_c[1], pos_c[2]]),
            "vel_enu": np.array([vel_c[0], vel_c[1], vel_c[2]]),
            "quat": np.array([quat_c[0], quat_c[1], quat_c[2], quat_c[3]]),
            "heading_deg": heading_c.value,
            "status": status_c.value,
            "confidence": conf_c.value,
        }

    def extract_ai_features(self) -> Optional[np.ndarray]:
        buf = (ctypes.c_float * 800)()
        count = self.lib.nav_core_extract_ai_features(self.handle, buf, 800)
        if count == 100:
            arr = np.array(buf, dtype=np.float32).reshape(100, 8)
            return arr
        return None

    def reset(self):
        self.lib.nav_core_reset(self.handle)

    def __del__(self):
        if hasattr(self, "handle") and self.handle:
            self.lib.nav_core_destroy(self.handle)
            self.handle = None


class EdgeParityEvaluator:
    """Evaluates numerical parity and latency between Python reference and C++ engine."""

    def __init__(self, lib_path: Optional[str] = None):
        self.cpp_core = CTypesNavCore(lib_path)
        self.py_eskf = ErrorStateKalmanFilter()

    def run_replay_comparison(
        self,
        imu_observations: List[IMUObservation],
        gnss_observations: List[GNSSObservation],
        initial_time: float,
        initial_llh: Tuple[float, float, float],
        initial_velocity: np.ndarray,
        initial_quat: np.ndarray,
    ) -> Dict[str, Any]:
        """Run synchronized trajectory replay through both Python and C++ engines."""
        # Initialize both
        self.py_eskf.initialize(
            initial_time=initial_time,
            initial_llh=initial_llh,
            initial_velocity_enu=initial_velocity,
            initial_quaternion=initial_quat,
        )
        self.cpp_core.initialize(
            t0=initial_time,
            lat_deg=initial_llh[0],
            lon_deg=initial_llh[1],
            alt_m=initial_llh[2],
            vel_enu=initial_velocity,
            quat=initial_quat,
        )

        pos_diffs = []
        vel_diffs = []
        heading_diffs = []
        cpp_latencies_us = []

        gnss_map = {round(g.timestamp, 2): g for g in gnss_observations}

        for imu in imu_observations:
            # 1. Measure C++ IMU step latency
            t_start = time.perf_counter_ns()
            self.cpp_core.process_imu(
                imu.timestamp,
                imu.accelerometer_x, imu.accelerometer_y, imu.accelerometer_z,
                imu.gyroscope_x, imu.gyroscope_y, imu.gyroscope_z,
            )
            t_end = time.perf_counter_ns()
            cpp_latencies_us.append((t_end - t_start) / 1000.0)

            # 2. Python reference step
            self.py_eskf.predict(imu)

            # 3. Check GNSS
            t_rounded = round(imu.timestamp, 2)
            if t_rounded in gnss_map:
                gnss = gnss_map[t_rounded]
                self.cpp_core.process_gnss(
                    gnss.timestamp,
                    gnss.latitude, gnss.longitude, gnss.altitude,
                    gnss.horizontal_accuracy if gnss.horizontal_accuracy is not None else 2.5
                )
                self.py_eskf.update_gnss(gnss)

            # Compare States
            py_state = self.py_eskf.get_state()
            cpp_state = self.cpp_core.get_state()

            p_diff = float(np.linalg.norm(py_state.position_enu - cpp_state["pos_enu"]))
            v_diff = float(np.linalg.norm(py_state.velocity_enu - cpp_state["vel_enu"]))

            py_heading = np.degrees(np.arctan2(py_state.rotation_matrix[1, 0], py_state.rotation_matrix[0, 0]))
            if py_heading < 0: py_heading += 360.0
            h_diff = abs(py_heading - cpp_state["heading_deg"])
            if h_diff > 180.0: h_diff = 360.0 - h_diff

            pos_diffs.append(p_diff)
            vel_diffs.append(v_diff)
            heading_diffs.append(h_diff)

        lat_arr = np.array(cpp_latencies_us)

        return {
            "num_samples": len(imu_observations),
            "pos_rmse": float(np.sqrt(np.mean(np.array(pos_diffs) ** 2))),
            "pos_max_diff": float(np.max(pos_diffs)),
            "vel_rmse": float(np.sqrt(np.mean(np.array(vel_diffs) ** 2))),
            "vel_max_diff": float(np.max(vel_diffs)),
            "heading_mean_diff_deg": float(np.mean(heading_diffs)),
            "heading_max_diff_deg": float(np.max(heading_diffs)),
            "latency_mean_us": float(np.mean(lat_arr)),
            "latency_median_us": float(np.median(lat_arr)),
            "latency_p95_us": float(np.percentile(lat_arr, 95)),
            "latency_p99_us": float(np.percentile(lat_arr, 99)),
            "latency_max_us": float(np.max(lat_arr)),
            "parity_status": "PASS" if np.max(pos_diffs) < 0.1 else "FAIL",
        }
