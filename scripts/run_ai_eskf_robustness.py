#!/usr/bin/env python3
"""Step 8.1 — AI-ESKF Robustness & Failure-Injection Validation Script.

Simulates 6 controlled failure modes during a 30s GNSS outage (30s-60s) to evaluate
Mahalanobis innovation gating protection and verify system safety:
1. Mode 6: AI Disabled (ESKF-only reference)
2. Mode 1: Normal AI (TCN model)
3. Mode 2: Small Bias (+1.0m East, +1.0m North)
4. Mode 3: Moderate Bias (+5.0m East, +5.0m North)
5. Mode 4: Large Outlier (+50.0m East, -50.0m North injected at specific windows)
6. Mode 5: Random Outliers (15% random corrupted predictions with fixed seed)

Generates comparative benchmark tables and saves diagnostic plots under results/ai_eskf_robustness/.
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
from src.outage_detection.detector import GNSSOutageDetector
from src.navigation.ekf.eskf import ErrorStateKalmanFilter
from src.ml.inference import DriftPredictor
from src.navigation.ai_fusion import AIESKFPipeline


class CorruptedDriftPredictor:
    """Wrapper around DriftPredictor to inject controlled failure modes."""

    def __init__(self, base_predictor: DriftPredictor, mode: str, seed: int = 42):
        self.base_predictor = base_predictor
        self.mode = mode
        self.rng = np.random.RandomState(seed)
        self.window_counter = 0

    def predict_window(self, imu_window: np.ndarray) -> np.ndarray:
        self.window_counter += 1
        delta_p = self.base_predictor.predict_window(imu_window)

        if self.mode == "normal":
            return delta_p
        elif self.mode == "small_bias":
            return delta_p + np.array([1.0, 1.0, 0.0], dtype=np.float32)
        elif self.mode == "moderate_bias":
            return delta_p + np.array([5.0, 5.0, 0.0], dtype=np.float32)
        elif self.mode == "large_outlier":
            # Inject huge outlier every 8th window
            if self.window_counter % 8 == 0:
                return delta_p + np.array([50.0, -50.0, 0.0], dtype=np.float32)
            return delta_p
        elif self.mode == "random_outliers":
            # Randomly corrupt ~15% of predictions
            if self.rng.rand() < 0.15:
                rand_vec = self.rng.uniform(30.0, 60.0, size=2)
                signs = self.rng.choice([-1.0, 1.0], size=2)
                return delta_p + np.array([rand_vec[0] * signs[0], rand_vec[1] * signs[1], 0.0], dtype=np.float32)
            return delta_p
        else:
            return delta_p


def main():
    print("=" * 80)
    print("SIH26168 STEP 8.1 — AI-ESKF ROBUSTNESS & FAILURE-INJECTION VALIDATION")
    print("=" * 80)

    # 1. Load Configuration
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "ai_eskf_robustness")
    os.makedirs(results_dir, exist_ok=True)

    ckpt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "ml_training", "best_drift_model.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Trained model checkpoint not found at {ckpt_path}. Run train_drift_model.py first.")

    base_predictor = DriftPredictor(ckpt_path)

    # 2. Generate Synthetic Trajectory
    duration_sec = 80.0
    outage_start = 30.0
    outage_end = 60.0

    imu_list, raw_gnss_list, gt_list = generate_synthetic_trajectory(seed=42, duration_sec=duration_sec)
    origin_lat, origin_lon, origin_alt = gt_list[0].latitude, gt_list[0].longitude, gt_list[0].altitude
    local_frame = LocalFrame(origin_lat, origin_lon, origin_alt)

    gt_times = np.array([gt.timestamp for gt in gt_list], dtype=np.float64)
    gt_enu = np.array([local_frame.to_enu(gt.latitude, gt.longitude, gt.altitude) for gt in gt_list], dtype=np.float64)

    gnss_outage_list = [g for g in raw_gnss_list if not (outage_start <= g.timestamp <= outage_end)]
    gnss_map = {round(g.timestamp, 2): g for g in gnss_outage_list}

    init_time = imu_list[0].timestamp
    init_llh = (origin_lat, origin_lon, origin_alt)
    init_vel = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    init_q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    ai_cfg = config.get("ekf", {}).get("ai_pseudo_measurement", {})
    r_ai_std = (
        ai_cfg.get("position_std_east", 1.5),
        ai_cfg.get("position_std_north", 1.5),
        ai_cfg.get("position_std_up", 3.0),
    )
    gate_thresh = ai_cfg.get("mahalanobis_gate_threshold", 4.0)

    modes = [
        ("ai_disabled", "AI Disabled (Reference)", None, False),
        ("normal", "Normal AI (TCN Model)", "normal", True),
        ("small_bias", "Small Bias (+1m E/N)", "small_bias", True),
        ("moderate_bias", "Moderate Bias (+5m E/N)", "moderate_bias", True),
        ("large_outlier", "Large Outlier (+50m Peak)", "large_outlier", True),
        ("random_outliers", "Random Outliers (15% Corrupted)", "random_outliers", True),
    ]

    simulation_results = {}

    print(f"\n[1/4] Running 6 Failure-Injection Simulation Modes (Gate Threshold = {gate_thresh:.1f}σ)...")

    for mode_key, mode_label, mode_type, enable_ai in modes:
        eskf = ErrorStateKalmanFilter()
        eskf.initialize(init_time, init_llh, init_vel, init_q)
        detector = GNSSOutageDetector(config.get("outage_detection", {}))

        if enable_ai and mode_type:
            c_predictor = CorruptedDriftPredictor(base_predictor, mode=mode_type, seed=42)
        else:
            c_predictor = None

        pipeline = AIESKFPipeline(
            eskf=eskf,
            detector=detector,
            predictor=c_predictor,
            enable_ai=enable_ai,
            r_ai_std=r_ai_std,
            gate_threshold=gate_thresh,
        )

        pos_history = []
        for imu in imu_list:
            t_key = round(imu.timestamp, 2)
            g_fix = gnss_map.get(t_key)
            st, _ = pipeline.process_sample(imu, g_fix)
            pos_history.append(st.position_enu)

        pos_arr = np.array(pos_history)
        err_arr = np.linalg.norm(gt_enu[:, :2] - pos_arr[:, :2], axis=1)

        outage_mask = (gt_times >= outage_start) & (gt_times <= outage_end)
        outage_rmse = float(np.sqrt(np.mean(err_arr[outage_mask] ** 2)))
        idx_60s = int(np.argmin(np.abs(gt_times - 60.0)))

        max_mah = float(np.max(pipeline.mahalanobis_history)) if pipeline.mahalanobis_history else 0.0

        simulation_results[mode_key] = {
            "label": mode_label,
            "positions": pos_arr,
            "errors": err_arr,
            "outage_rmse": outage_rmse,
            "err_60s": float(err_arr[idx_60s]),
            "max_err": float(np.max(err_arr)),
            "accepted": pipeline.ai_accepted_count,
            "rejected": pipeline.ai_rejected_count,
            "acceptance_rate": float(pipeline.acceptance_rate),
            "max_mahalanobis": max_mah,
            "history": pipeline.update_history,
        }
        print(f"  - {mode_label:<32}: Outage RMSE = {outage_rmse:>6.2f}m | Err@60s = {err_arr[idx_60s]:>6.2f}m | Accepted = {pipeline.ai_accepted_count:>2} | Rejected = {pipeline.ai_rejected_count:>2}")

    # 3. Print Benchmark Table
    print("\n[2/4] ROBUSTNESS & FAILURE-INJECTION COMPARATIVE BENCHMARK TABLE")
    print("-" * 100)
    print(f"{'Mode':<32} | {'Outage RMSE':<11} | {'Err @ 60s':<11} | {'Max Err':<10} | {'Accepted':<8} | {'Rejected':<8} | {'Rate':<6}")
    print("-" * 100)
    for mk, data in simulation_results.items():
        print(
            f"{data['label']:<32} | "
            f"{data['outage_rmse']:>9.2f} m | "
            f"{data['err_60s']:>9.2f} m | "
            f"{data['max_err']:>8.2f} m | "
            f"{data['accepted']:>8} | "
            f"{data['rejected']:>8} | "
            f"{data['acceptance_rate']:>5.1f}%"
        )
    print("-" * 100)

    # Save metrics JSON
    summary_metrics = {}
    for mk, d in simulation_results.items():
        summary_metrics[mk] = {
            "label": d["label"],
            "outage_rmse": d["outage_rmse"],
            "err_60s": d["err_60s"],
            "max_err": d["max_err"],
            "accepted": d["accepted"],
            "rejected": d["rejected"],
            "acceptance_rate": d["acceptance_rate"],
            "max_mahalanobis": d["max_mahalanobis"],
        }
    metrics_json_path = os.path.join(results_dir, "ai_eskf_robustness_metrics.json")
    with open(metrics_json_path, "w") as f:
        json.dump(summary_metrics, f, indent=2)

    # 4. Generate Visualizations
    print("\n[3/4] Generating Diagnostic Plots...")

    # Plot 1: Position Error Comparison across Failure Modes
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axvspan(outage_start, outage_end, color="red", alpha=0.15, label="GNSS Outage (30s-60s)")
    colors = {
        "ai_disabled": "#7f7f7f",
        "normal": "#1f77b4",
        "small_bias": "#ff7f0e",
        "moderate_bias": "#2ca02c",
        "large_outlier": "#d62728",
        "random_outliers": "#9467bd",
    }
    for mk, d in simulation_results.items():
        ls = "--" if mk == "ai_disabled" else "-"
        ax.plot(gt_times, d["errors"], label=d["label"], color=colors[mk], linestyle=ls, linewidth=1.8)
    ax.set_title("Position Error Over Time Under Failure Injection & Innovation Gating")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Horizontal Error (m)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "robustness_position_error.png"), dpi=300)
    plt.close()

    # Plot 2: Mahalanobis Distance for Outlier Modes
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for mk in ["normal", "large_outlier", "random_outliers"]:
        hist = [h["mahalanobis_distance"] for h in simulation_results[mk]["history"]]
        if hist:
            ax.plot(hist, "o-", label=simulation_results[mk]["label"], color=colors[mk], alpha=0.8)
    ax.axhline(gate_thresh, color="red", linestyle="--", linewidth=2.0, label=f"Gating Threshold ({gate_thresh:.1f}σ)")
    ax.set_title("Mahalanobis Innovation Distance Gating Behavior")
    ax.set_xlabel("AI Update Index")
    ax.set_ylabel("Mahalanobis Distance")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "mahalanobis_distance.png"), dpi=300)
    plt.close()

    # Plot 3: Accepted vs Rejected Updates Bar Chart
    fig, ax = plt.subplots(figsize=(8, 4.5))
    mode_keys = ["normal", "small_bias", "moderate_bias", "large_outlier", "random_outliers"]
    labels = [simulation_results[mk]["label"] for mk in mode_keys]
    accepted = [simulation_results[mk]["accepted"] for mk in mode_keys]
    rejected = [simulation_results[mk]["rejected"] for mk in mode_keys]

    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width / 2, accepted, width, label="Accepted", color="#2ca02c")
    ax.bar(x + width / 2, rejected, width, label="Rejected", color="#d62728")
    ax.set_ylabel("Number of AI Updates")
    ax.set_title("AI Innovation Gating Decision Breakdown by Failure Mode")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "accepted_rejected_updates.png"), dpi=300)
    plt.close()

    # Plot 4: Outlier Injection Timeline (Large Outlier Mode)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    d_out = simulation_results["large_outlier"]
    h_out = d_out["history"]
    ts = [item["timestamp"] for item in h_out]
    mahs = [item["mahalanobis_distance"] for item in h_out]
    accs = [item["accepted"] for item in h_out]

    ax.axvspan(outage_start, outage_end, color="red", alpha=0.15, label="GNSS Outage Window")
    for t, m, a in zip(ts, mahs, accs):
        col = "green" if a else "red"
        marker = "o" if a else "X"
        ms = 6 if a else 10
        ax.plot(t, m, marker=marker, color=col, markersize=ms)
    ax.axhline(gate_thresh, color="black", linestyle="--", label=f"Gating Threshold ({gate_thresh:.1f}σ)")
    ax.set_title("Outlier Injection Timeline (Large Outlier Mode 4: Green=Accepted, Red X=Rejected)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mahalanobis Distance")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "outlier_injection_timeline.png"), dpi=300)
    plt.close()

    print(f"Saved diagnostic plots to: {results_dir}")

    print("\n" + "=" * 80)
    print("STEP 8.1 AI-ESKF ROBUSTNESS EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
