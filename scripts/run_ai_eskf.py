#!/usr/bin/env python3
"""Step 8 — AI-Integrated 15-State Error-State Kalman Filter Experiment Script.

Simulates 4 comparative navigation scenarios:
- Scenario A: Open-loop Strapdown INS (IMU only)
- Scenario B: ESKF + Continuous GNSS (no outage)
- Scenario C: ESKF with GNSS Outage (30s-60s), AI disabled
- Scenario D: ESKF + AI Pseudo-Measurement updates during GNSS Outage (30s-60s)

Evaluates position error trajectory dynamics, gating statistics, and generates diagnostic plots.
"""

import os
import sys
import json
import yaml
import numpy as np
import matplotlib.pyplot as plt

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.coordinate_transforms import LocalFrame
from src.data.loaders.synthetic import generate_synthetic_trajectory
from src.data.observations import GNSSObservation
from src.outage_detection.detector import GNSSOutageDetector, GNSSStatus
from src.navigation.ins import StrapdownINS
from src.navigation.ekf.eskf import ErrorStateKalmanFilter
from src.navigation.ekf.state import ESKFStateConfig
from src.ml.inference import DriftPredictor
from src.navigation.ai_fusion import AIESKFPipeline
from src.evaluation.trajectory_evaluation import evaluate_trajectory


def main():
    print("=" * 80)
    print("SIH26168 STEP 8 — AI-INTEGRATED 15-STATE ESKF NAVIGATION EXPERIMENT")
    print("=" * 80)

    # 1. Load configuration and paths
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "ai_eskf")
    os.makedirs(results_dir, exist_ok=True)

    ckpt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "ml_training", "best_drift_model.pt")
    has_predictor = os.path.exists(ckpt_path)

    if has_predictor:
        print(f"[Init] Loading trained DriftPredictor model from: {ckpt_path}")
        predictor = DriftPredictor(ckpt_path)
    else:
        print("[Init] Warning: Trained checkpoint not found. Operating in fallback mode.")
        predictor = None

    # 2. Generate Deterministic Synthetic Trajectory (seed=42, duration=80.0s)
    duration_sec = 80.0
    outage_start = 30.0
    outage_end = 60.0

    print(f"\n[1/6] Generating deterministic synthetic trajectory (80.0s at 100Hz)...")
    imu_list, raw_gnss_list, gt_list = generate_synthetic_trajectory(seed=42, duration_sec=duration_sec)

    origin_lat = gt_list[0].latitude
    origin_lon = gt_list[0].longitude
    origin_alt = gt_list[0].altitude
    local_frame = LocalFrame(origin_lat, origin_lon, origin_alt)

    # Extract ground truth ENU positions
    gt_times = np.array([gt.timestamp for gt in gt_list], dtype=np.float64)
    gt_enu = np.array(
        [local_frame.to_enu(gt.latitude, gt.longitude, gt.altitude) for gt in gt_list],
        dtype=np.float64,
    )

    # Build GNSS observation lists (Continuous vs Outage 30s-60s)
    gnss_continuous = raw_gnss_list
    gnss_outage_list = [g for g in raw_gnss_list if not (outage_start <= g.timestamp <= outage_end)]

    # Initial states
    init_time = imu_list[0].timestamp
    init_llh = (origin_lat, origin_lon, origin_alt)
    init_vel = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    init_q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    # --------------------------------------------------------------------------
    # SCENARIO A: Open-Loop Strapdown INS (IMU Only)
    # --------------------------------------------------------------------------
    print("\n[2/6] Simulating Scenario A: Open-loop Strapdown INS (IMU only)...")
    ins_a = StrapdownINS()
    ins_a.initialize(init_time, init_llh, init_vel, init_q)
    pos_a = []
    for imu in imu_list:
        st = ins_a.update(imu)
        pos_a.append(st.position_enu)
    pos_a = np.array(pos_a)

    # --------------------------------------------------------------------------
    # SCENARIO B: ESKF + Continuous GNSS (No Outage)
    # --------------------------------------------------------------------------
    print("\n[3/6] Simulating Scenario B: ESKF + Continuous GNSS (No Outage)...")
    eskf_b = ErrorStateKalmanFilter()
    eskf_b.initialize(init_time, init_llh, init_vel, init_q)
    detector_b = GNSSOutageDetector(config.get("outage_detection", {}))
    pipeline_b = AIESKFPipeline(eskf=eskf_b, detector=detector_b, predictor=None, enable_ai=False)

    gnss_map_b = {round(g.timestamp, 2): g for g in gnss_continuous}
    pos_b = []
    for imu in imu_list:
        t_key = round(imu.timestamp, 2)
        g_fix = gnss_map_b.get(t_key)
        st, _ = pipeline_b.process_sample(imu, g_fix)
        pos_b.append(st.position_enu)
    pos_b = np.array(pos_b)

    # --------------------------------------------------------------------------
    # SCENARIO C: ESKF with GNSS Outage (30s-60s), AI Disabled
    # --------------------------------------------------------------------------
    print("\n[4/6] Simulating Scenario C: ESKF with GNSS Outage (30s-60s, AI Disabled)...")
    eskf_c = ErrorStateKalmanFilter()
    eskf_c.initialize(init_time, init_llh, init_vel, init_q)
    detector_c = GNSSOutageDetector(config.get("outage_detection", {}))
    pipeline_c = AIESKFPipeline(eskf=eskf_c, detector=detector_c, predictor=None, enable_ai=False)

    gnss_map_c = {round(g.timestamp, 2): g for g in gnss_outage_list}
    pos_c = []
    for imu in imu_list:
        t_key = round(imu.timestamp, 2)
        g_fix = gnss_map_c.get(t_key)
        st, _ = pipeline_c.process_sample(imu, g_fix)
        pos_c.append(st.position_enu)
    pos_c = np.array(pos_c)

    # --------------------------------------------------------------------------
    # SCENARIO D: ESKF + AI Pseudo-Measurement Updates during GNSS Outage (30s-60s)
    # --------------------------------------------------------------------------
    print("\n[5/6] Simulating Scenario D: ESKF + AI Pseudo-Measurement Updates during Outage...")
    eskf_d = ErrorStateKalmanFilter()
    eskf_d.initialize(init_time, init_llh, init_vel, init_q)
    detector_d = GNSSOutageDetector(config.get("outage_detection", {}))

    ai_cfg = config.get("ekf", {}).get("ai_pseudo_measurement", {})
    r_ai_std = (
        ai_cfg.get("position_std_east", 1.5),
        ai_cfg.get("position_std_north", 1.5),
        ai_cfg.get("position_std_up", 3.0),
    )
    gate_thresh = ai_cfg.get("mahalanobis_gate_threshold", 4.0)

    pipeline_d = AIESKFPipeline(
        eskf=eskf_d,
        detector=detector_d,
        predictor=predictor,
        enable_ai=True,
        r_ai_std=r_ai_std,
        gate_threshold=gate_thresh,
    )

    gnss_map_d = {round(g.timestamp, 2): g for g in gnss_outage_list}
    pos_d = []
    for imu in imu_list:
        t_key = round(imu.timestamp, 2)
        g_fix = gnss_map_d.get(t_key)
        st, _ = pipeline_d.process_sample(imu, g_fix)
        pos_d.append(st.position_enu)
    pos_d = np.array(pos_d)

    # --------------------------------------------------------------------------
    # METRICS EVALUATION AT KEY TIMESTAMPS
    # --------------------------------------------------------------------------
    print("\n[6/6] Computing Performance Metrics and Comparative Benchmark Table...")
    
    def get_err(pos_array):
        return np.linalg.norm(gt_enu[:, :2] - pos_array[:, :2], axis=1)

    err_a = get_err(pos_a)
    err_b = get_err(pos_b)
    err_c = get_err(pos_c)
    err_d = get_err(pos_d)

    # Outage mask (30s to 60s)
    outage_mask = (gt_times >= outage_start) & (gt_times <= outage_end)

    idx_start = int(np.argmin(np.abs(gt_times - outage_start)))
    idx_mid = int(np.argmin(np.abs(gt_times - 45.0)))
    idx_end = int(np.argmin(np.abs(gt_times - outage_end)))
    idx_rec = int(np.argmin(np.abs(gt_times - 62.0)))

    metrics_table = {
        "Scenario A (Open-Loop INS)": {
            "err_start_30s": float(err_a[idx_start]),
            "err_mid_45s": float(err_a[idx_mid]),
            "err_end_60s": float(err_a[idx_end]),
            "err_rec_62s": float(err_a[idx_rec]),
            "outage_rmse": float(np.sqrt(np.mean(err_a[outage_mask] ** 2))),
            "overall_rmse": float(np.sqrt(np.mean(err_a**2))),
            "final_error": float(err_a[-1]),
        },
        "Scenario C (ESKF Outage - AI Off)": {
            "err_start_30s": float(err_c[idx_start]),
            "err_mid_45s": float(err_c[idx_mid]),
            "err_end_60s": float(err_c[idx_end]),
            "err_rec_62s": float(err_c[idx_rec]),
            "outage_rmse": float(np.sqrt(np.mean(err_c[outage_mask] ** 2))),
            "overall_rmse": float(np.sqrt(np.mean(err_c**2))),
            "final_error": float(err_c[-1]),
        },
        "Scenario D (ESKF Outage - AI On)": {
            "err_start_30s": float(err_d[idx_start]),
            "err_mid_45s": float(err_d[idx_mid]),
            "err_end_60s": float(err_d[idx_end]),
            "err_rec_62s": float(err_d[idx_rec]),
            "outage_rmse": float(np.sqrt(np.mean(err_d[outage_mask] ** 2))),
            "overall_rmse": float(np.sqrt(np.mean(err_d**2))),
            "final_error": float(err_d[-1]),
            "ai_accepted_count": pipeline_d.ai_accepted_count,
            "ai_rejected_count": pipeline_d.ai_rejected_count,
            "acceptance_rate_pct": pipeline_d.acceptance_rate,
        },
    }

    print("\n" + "-" * 85)
    print(f"{'System Scenario':<35} | {'Outage RMSE':<12} | {'Error @ 60s':<12} | {'Recovery @ 62s':<12}")
    print("-" * 85)
    for sc_name, m in metrics_table.items():
        print(f"{sc_name:<35} | {m['outage_rmse']:>10.2f} m | {m['err_end_60s']:>10.2f} m | {m['err_rec_62s']:>10.2f} m")
    print("-" * 85)

    print(f"\n[AI Gating Statistics]")
    print(f"  - AI Updates Accepted : {pipeline_d.ai_accepted_count}")
    print(f"  - AI Updates Rejected : {pipeline_d.ai_rejected_count}")
    print(f"  - Acceptance Rate     : {pipeline_d.acceptance_rate:.1f}%")

    metrics_json_path = os.path.join(results_dir, "ai_eskf_metrics.json")
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_table, f, indent=2)

    # --------------------------------------------------------------------------
    # DIAGNOSTIC PLOTS GENERATION
    # --------------------------------------------------------------------------
    print("\nGenerating Diagnostic Plots...")

    # Plot 1: Trajectory Comparison (2D ENU)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(gt_enu[:, 0], gt_enu[:, 1], "k--", label="Ground Truth", linewidth=2)
    ax.plot(pos_c[:, 0], pos_c[:, 1], "r-", label="Scenario C (ESKF Outage, AI Off)", alpha=0.8)
    ax.plot(pos_d[:, 0], pos_d[:, 1], "b-", label="Scenario D (ESKF + AI On)", alpha=0.9)
    ax.plot(pos_b[:, 0], pos_b[:, 1], "g:", label="Scenario B (Continuous GNSS)", alpha=0.8)
    ax.set_title("2D ENU Trajectory Comparison under GNSS Outage (30s-60s)")
    ax.set_xlabel("East Position (m)")
    ax.set_ylabel("North Position (m)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "trajectory_comparison.png"), dpi=300)
    plt.close()

    # Plot 2 & 3: Horizontal Error vs Time
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.axvspan(outage_start, outage_end, color="red", alpha=0.15, label="GNSS Outage Window (30s-60s)")
    ax.plot(gt_times, err_c, "r-", label="ESKF Outage (AI Off)", linewidth=1.8)
    ax.plot(gt_times, err_d, "b-", label="ESKF + AI Pseudo-Measurements", linewidth=2.0)
    ax.plot(gt_times, err_b, "g:", label="Continuous GNSS Baseline", linewidth=1.5)
    ax.set_title("Horizontal Position Error Over Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Horizontal Error (m)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "horizontal_error_vs_time.png"), dpi=300)
    plt.savefig(os.path.join(results_dir, "position_error_comparison.png"), dpi=300)
    plt.close()

    # Plot 4: AI Update Innovation Mahalanobis Distance
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if pipeline_d.mahalanobis_history:
        ax.plot(pipeline_d.mahalanobis_history, "bo-", label="AI Innovation Mahalanobis Dist")
        ax.axhline(gate_thresh, color="r", linestyle="--", label=f"Gating Threshold ({gate_thresh:.1f}σ)")
    ax.set_title("AI Pseudo-Measurement Innovation Mahalanobis Statistic")
    ax.set_xlabel("AI Update Index")
    ax.set_ylabel("Mahalanobis Distance")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "ai_update_innovation.png"), dpi=300)
    plt.close()

    # Plot 5: Acceptance Rate Pie Chart
    fig, ax = plt.subplots(figsize=(5, 5))
    acc = pipeline_d.ai_accepted_count
    rej = pipeline_d.ai_rejected_count
    if acc + rej > 0:
        ax.pie([acc, rej], labels=["Accepted", "Rejected"], autopct="%1.1f%%", colors=["#2ca02c", "#d62728"], startangle=140)
    else:
        ax.text(0.5, 0.5, "No AI Updates Attempted", ha="center", va="center")
    ax.set_title("AI Innovation Gating Acceptance Rate")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "ai_update_acceptance.png"), dpi=300)
    plt.close()

    # Plot 6: East/North Error Decomposition
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    e_c = np.abs(gt_enu[:, 0] - pos_c[:, 0])
    e_d = np.abs(gt_enu[:, 0] - pos_d[:, 0])
    n_c = np.abs(gt_enu[:, 1] - pos_c[:, 1])
    n_d = np.abs(gt_enu[:, 1] - pos_d[:, 1])

    axes[0].axvspan(outage_start, outage_end, color="red", alpha=0.15)
    axes[0].plot(gt_times, e_c, "r-", label="ESKF (AI Off)")
    axes[0].plot(gt_times, e_d, "b-", label="ESKF + AI")
    axes[0].set_ylabel("East Error (m)")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    axes[1].axvspan(outage_start, outage_end, color="red", alpha=0.15)
    axes[1].plot(gt_times, n_c, "r-", label="ESKF (AI Off)")
    axes[1].plot(gt_times, n_d, "b-", label="ESKF + AI")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("North Error (m)")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "east_north_error_comparison.png"), dpi=300)
    plt.close()

    # Plot 7: Zoomed Outage Window (30s to 60s)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(gt_times[outage_mask], err_c[outage_mask], "r-", label="ESKF Outage (AI Off)", linewidth=2.0)
    ax.plot(gt_times[outage_mask], err_d[outage_mask], "b-", label="ESKF + AI Pseudo-Measurements", linewidth=2.2)
    ax.set_title("Zoomed Position Drift During 30s GNSS Outage Window")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Horizontal Error (m)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "gnss_outage_window.png"), dpi=300)
    plt.close()

    print(f"Saved diagnostic plot artifacts to: {results_dir}")

    print("\n" + "=" * 80)
    print("STEP 8 AI-INTEGRATED ESKF NAVIGATION EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
