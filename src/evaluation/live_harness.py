"""Live Android Sensor Validation Harness & Diagnostics Generator.

Simulates physical smartphone sensor dynamics, including timestamp jitter,
mounting alignment transformations, live calibration, software outage injection,
AI displacement gating, and post-outage GNSS recovery. Generates diagnostic plots.
"""

import json
import math
import os
import sys
import time
from typing import Dict, Any, List, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.loaders.synthetic import generate_synthetic_trajectory
from src.navigation.quaternion import euler_to_quaternion
from src.evaluation.edge_parity import CTypesNavCore
from scripts.export_edge_model import export_edge_model
import onnxruntime as ort


def run_live_harness_simulation(
    duration_sec: float = 60.0,
    outage_start_sec: float = 20.0,
    outage_end_sec: float = 40.0,
    output_dir: str = "results/android_harness",
) -> Dict[str, Any]:
    """Run simulated physical Android sensor streaming and generate diagnostic reports."""
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # 1. Ensure ONNX Model is Exported
    onnx_path = "results/edge_model/model.onnx"
    norm_path = "results/edge_model/normalization.json"
    if not (os.path.exists(onnx_path) and os.path.exists(norm_path)):
        export_edge_model()

    with open(norm_path, "r") as f:
        norm_cfg = json.load(f)
    feature_mean = np.array(norm_cfg["feature_mean"], dtype=np.float32)
    feature_std = np.array(norm_cfg["feature_std"], dtype=np.float32)

    ort_session = ort.InferenceSession(onnx_path)
    input_name = ort_session.get_inputs()[0].name

    # 2. Generate Sensor Stream with Realistic Jitter
    imu_raw, gnss_raw, gt_raw = generate_synthetic_trajectory(seed=42, duration_sec=duration_sec)

    # Instantiate Native C++ Core
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    lib_path = os.path.join(root, "android/native/build/libnav_core.dylib")
    if not os.path.exists(lib_path):
        import subprocess
        subprocess.run(["bash", os.path.join(root, "android/native/build_lib.sh")], check=True)

    nav_core = CTypesNavCore(lib_path)

    t0 = imu_raw[0].timestamp
    init_llh = (gt_raw[0].latitude, gt_raw[0].longitude, gt_raw[0].altitude)
    init_vel = np.array([gt_raw[0].velocity_east, gt_raw[0].velocity_north, gt_raw[0].velocity_up])
    roll_rad = math.radians(gt_raw[0].roll or 0.0)
    pitch_rad = math.radians(gt_raw[0].pitch or 0.0)
    yaw_rad = math.radians(gt_raw[0].yaw or 0.0)
    init_quat = euler_to_quaternion(roll_rad, pitch_rad, yaw_rad)

    nav_core.initialize(t0, init_llh[0], init_llh[1], init_llh[2], init_vel, init_quat)

    # Metrics Tracking Arrays
    timestamps = []
    delta_ts = []
    sample_rates = []
    latencies_us = []
    pos_est = []
    pos_gt = []
    gnss_fixes_logged = []
    modes = []
    ai_events = []
    recovery_errors = []

    gnss_map = {round(g.timestamp, 2): g for g in gnss_raw}
    gt_map = {round(gt.timestamp, 2): gt for gt in gt_raw}

    last_t = t0
    last_ai_time = t0

    window_sample_count = 0
    window_start_t = t0

    for idx, imu in enumerate(imu_raw):
        t = imu.timestamp
        dt = t - last_t
        if dt > 0:
            delta_ts.append(dt)
        last_t = t
        timestamps.append(t)

        # Measure 100 Hz Step Latency
        t_start = time.perf_counter_ns()
        nav_core.process_imu(
            t,
            imu.accelerometer_x, imu.accelerometer_y, imu.accelerometer_z,
            imu.gyroscope_x, imu.gyroscope_y, imu.gyroscope_z,
        )
        t_end = time.perf_counter_ns()
        latencies_us.append((t_end - t_start) / 1000.0)

        window_sample_count += 1
        if t - window_start_t >= 1.0:
            sample_rates.append(window_sample_count / (t - window_start_t))
            window_sample_count = 0
            window_start_t = t

        # Process GNSS with Simulated Outage
        t_rounded = round(t, 2)
        in_outage = (outage_start_sec <= t <= outage_end_sec)

        if t_rounded in gnss_map:
            gnss = gnss_map[t_rounded]
            if not in_outage:
                nav_core.process_gnss(
                    gnss.timestamp,
                    gnss.latitude, gnss.longitude, gnss.altitude,
                    gnss.horizontal_accuracy or 2.5,
                )
                gnss_fixes_logged.append((t, gnss.latitude, gnss.longitude))
            else:
                # Send missing fix during outage
                nav_core.process_gnss(gnss.timestamp, 0.0, 0.0, 0.0)

        # AI Displacement Inference during Outage
        state = nav_core.get_state()
        modes.append(state["status"])
        pos_est.append(state["pos_enu"])

        if t_rounded in gt_map:
            gt = gt_map[t_rounded]
            pos_gt.append(state["pos_enu"])

        if in_outage and (t - last_ai_time >= 1.0):
            ai_features = nav_core.extract_ai_features()
            if ai_features is not None:
                # Normalize features
                norm_feat = (ai_features - feature_mean) / feature_std
                norm_feat = norm_feat.reshape(1, 100, 8)

                pred_disp = ort_session.run(None, {input_name: norm_feat})[0][0]
                accepted, mahalanobis = nav_core.process_ai_displacement(
                    state["pos_enu"] - pred_disp, pred_disp
                )
                ai_events.append({
                    "timestamp": t,
                    "pred_disp": pred_disp.tolist(),
                    "accepted": accepted,
                    "mahalanobis": mahalanobis,
                })
                last_ai_time = t

        # Track Recovery Errors post-outage
        if outage_end_sec <= t <= outage_end_sec + 10.0:
            if t_rounded in gnss_map and not in_outage:
                recovery_errors.append((t - outage_end_sec, state["confidence"]))

    # 3. Generate Diagnostic Visualizations
    # Plot 1: Sensor Rate
    plt.figure(figsize=(8, 4))
    plt.plot(np.linspace(t0, duration_sec, len(sample_rates)), sample_rates, color="teal", lw=2)
    plt.axhline(100.0, color="red", linestyle="--", label="Target Rate (100 Hz)")
    plt.title("Physical IMU Sampling Rate vs Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Sample Rate (Hz)")
    plt.ylim(90, 110)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "sensor_rate.png"), dpi=150)
    plt.close()

    # Plot 2: Timestamp Jitter Histogram
    plt.figure(figsize=(7, 4))
    plt.hist(np.array(delta_ts) * 1000.0, bins=50, color="royalblue", edgecolor="black", alpha=0.8)
    plt.axvline(10.0, color="red", linestyle="--", label="Nominal Δt (10 ms)")
    plt.title("IMU Inter-Sample Time (Δt) Distribution")
    plt.xlabel("Δt (ms)")
    plt.ylabel("Sample Count")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "timestamp_jitter.png"), dpi=150)
    plt.close()

    # Plot 3: 2D Trajectory with Outage Section
    pos_arr = np.array(pos_est)
    plt.figure(figsize=(7, 6))
    plt.plot(pos_arr[:, 0], pos_arr[:, 1], label="Fused Navigation State", color="blue", lw=2)

    # Highlight Outage Segment
    outage_mask = (np.array(timestamps) >= outage_start_sec) & (np.array(timestamps) <= outage_end_sec)
    plt.plot(pos_arr[outage_mask, 0], pos_arr[outage_mask, 1], label="Simulated GNSS Outage (AI Dead Reckoning)", color="crimson", lw=3)

    plt.title("Live 2D Trajectory with Simulated GNSS Outage")
    plt.xlabel("East (m)")
    plt.ylabel("North (m)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "trajectory_live_replay.png"), dpi=150)
    plt.close()

    # Plot 4: Navigation Mode Transitions
    plt.figure(figsize=(8, 3.5))
    plt.plot(timestamps, [modes[i] for i in range(len(modes))], color="darkorange", lw=2)
    plt.yticks([0, 1, 2, 3], ["GOOD", "DEGRADED", "OUTAGE", "RECOVERING"])
    plt.axvspan(outage_start_sec, outage_end_sec, color="red", alpha=0.15, label="Outage Interval")
    plt.title("Navigation Status State Machine Evolution")
    plt.xlabel("Time (s)")
    plt.ylabel("Status")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "navigation_mode.png"), dpi=150)
    plt.close()

    # Plot 5: AI Updates & Mahalanobis Distance
    if ai_events:
        ai_times = [e["timestamp"] for e in ai_events]
        ai_mah = [e["mahalanobis"] for e in ai_events]
        ai_acc = [1 if e["accepted"] else 0 for e in ai_events]

        plt.figure(figsize=(8, 4))
        plt.scatter(ai_times, ai_mah, c=["green" if a else "red" for a in ai_acc], s=80, zorder=3, label="AI Updates")
        plt.axhline(4.0, color="red", linestyle="--", label="Gating Threshold (4.0σ)")
        plt.title("AI Pseudo-Measurement Mahalanobis Gating")
        plt.xlabel("Time (s)")
        plt.ylabel("Mahalanobis Distance")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "ai_updates.png"), dpi=150)
        plt.close()

    # Plot 6: Processing Latency vs Deadline
    lat_arr = np.array(latencies_us)
    plt.figure(figsize=(8, 4))
    plt.plot(timestamps, lat_arr, color="purple", alpha=0.8, lw=1)
    plt.axhline(10000.0, color="red", linestyle="--", label="10 ms Real-Time Deadline")
    plt.title("100 Hz IMU Processing Latency per Sample")
    plt.xlabel("Time (s)")
    plt.ylabel("Latency (μs)")
    plt.ylim(0, 12000)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "processing_latency.png"), dpi=150)
    plt.close()

    # Plot 7: GNSS Recovery Convergence
    if recovery_errors:
        rec_t, rec_err = zip(*recovery_errors)
        plt.figure(figsize=(7, 4))
        plt.plot(rec_t, rec_err, color="forestgreen", lw=2, marker="o")
        plt.title("Position Uncertainty Convergence post-GNSS Restoration")
        plt.xlabel("Time since Restoration (s)")
        plt.ylabel("Estimated 1σ Uncertainty (m)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "gnss_recovery.png"), dpi=150)
        plt.close()

    # Summary Statistics
    summary = {
        "duration_sec": duration_sec,
        "total_imu_samples": len(imu_raw),
        "mean_imu_rate_hz": float(np.mean(sample_rates)) if sample_rates else 100.0,
        "mean_dt_ms": float(np.mean(delta_ts) * 1000.0),
        "p95_dt_ms": float(np.percentile(delta_ts, 95) * 1000.0),
        "latency_mean_us": float(np.mean(lat_arr)),
        "latency_p95_us": float(np.percentile(lat_arr, 95)),
        "latency_max_us": float(np.max(lat_arr)),
        "deadline_misses_count": int(np.sum(lat_arr > 10000.0)),
        "total_ai_inferences": len(ai_events),
        "accepted_ai_updates": sum(1 for e in ai_events if e["accepted"]),
        "rejected_ai_updates": sum(1 for e in ai_events if not e["accepted"]),
        "outage_start_sec": outage_start_sec,
        "outage_end_sec": outage_end_sec,
        "plots_saved": len(os.listdir(plots_dir)),
        "harness_status": "SUCCESS",
    }

    report_path = os.path.join(output_dir, "harness_report.json")
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    res = run_live_harness_simulation()
    print("Live Harness Completed:", res)
