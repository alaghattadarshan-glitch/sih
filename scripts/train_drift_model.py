#!/usr/bin/env python3
"""Step 7 — AI/ML Inertial Drift-Correction Model Training & Evaluation Script.

Builds synthetic dataset, executes strict trajectory-aware anti-leakage splitting,
computes classical physical IMU baseline, trains 1D CNN / TCN drift model with early stopping,
saves model checkpoint and configuration, and generates comparative evaluation plots & metrics.
"""

import os
import sys
import json
import yaml
import numpy as np

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from src.coordinate_transforms import LocalFrame
from src.data.loaders.synthetic import generate_synthetic_trajectory
from src.data.observations import GNSSObservation
from src.outage_detection.detector import GNSSOutageDetector, GNSSStatus
from src.ml.datasets.feature_extractor import extract_imu_windows
from src.ml.datasets.target_builder import build_window_targets
from src.ml.datasets.imu_dataset import IMUWindowDataset
from src.ml.datasets.splitter import split_by_trajectory
from src.ml.models.tcn_drift import TCNDriftModel
from src.ml.baseline import compute_dataset_classical_baseline
from src.ml.training.trainer import ModelTrainer, set_seed
from src.ml.evaluation.metrics import evaluate_comparative_performance


def inject_gnss_anomalies(gnss_list: list, traj_id: int) -> list:
    """Inject controlled GNSS outages and degradation into synthetic GNSS stream for evaluation."""
    modified_gnss = []
    for obs in gnss_list:
        t = obs.timestamp
        if traj_id == 1 and 40.0 <= t <= 70.0:
            continue  # Outage
        elif traj_id == 2 and 30.0 <= t <= 50.0:
            obs = GNSSObservation(
                timestamp=obs.timestamp,
                latitude=obs.latitude + 0.0001,
                longitude=obs.longitude - 0.0001,
                altitude=obs.altitude + 15.0,
                horizontal_accuracy=12.0,  # Degraded
                vertical_accuracy=20.0,
            )
        elif traj_id == 4 and 45.0 <= t <= 75.0:
            # Outage in held-out test trajectory
            continue
        modified_gnss.append(obs)
    return modified_gnss


def main():
    print("=" * 80)
    print("SIH26168 STEP 7 — AI/ML INERTIAL DRIFT-CORRECTION MODEL TRAINING")
    print("=" * 80)

    # 1. Load Configuration & Set Reproducibility Seed
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    ml_cfg = config.get("machine_learning", {})
    outage_cfg = config.get("outage_detection", {})
    dataset_cfg = config.get("ml_dataset", {})

    seed = ml_cfg.get("seed", 42)
    set_seed(seed)

    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "ml_training")
    os.makedirs(results_dir, exist_ok=True)

    window_size = dataset_cfg.get("window_size_samples", 100)
    stride = dataset_cfg.get("stride_samples", 50)
    feature_cols = dataset_cfg.get("feature_columns", [
        "accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z", "accel_norm", "gyro_norm", "rel_time"
    ])

    # 2. Generate 5 Trajectory Sessions
    n_trajectories = 5
    duration_sec = 120.0
    raw_trajectory_data = []

    print(f"\n[1/7] Generating {n_trajectories} synthetic trajectory sessions...")
    for i in range(1, n_trajectories + 1):
        traj_seed = seed + i
        traj_id = f"trajectory_session_{i:02d}"

        imu_list, raw_gnss, gt_list = generate_synthetic_trajectory(seed=traj_seed, duration_sec=duration_sec)
        gnss_list = inject_gnss_anomalies(raw_gnss, traj_id=i)

        # Detect GNSS status per timestamp
        detector = GNSSOutageDetector(outage_cfg)
        status_timeline = []
        gnss_idx = 0
        n_gnss = len(gnss_list)

        for imu in imu_list:
            current_time = imu.timestamp
            current_obs = None
            if gnss_idx < n_gnss and abs(gnss_list[gnss_idx].timestamp - current_time) < 0.5:
                current_obs = gnss_list[gnss_idx]
                gnss_idx += 1

            status = detector.process_observation(current_obs, current_time=current_time)
            status_timeline.append(status)

        # Window extraction
        windows, time_ranges, actual_feature_names = extract_imu_windows(
            imu_list=imu_list,
            window_size_samples=window_size,
            stride_samples=stride,
            feature_columns=feature_cols,
        )

        origin_lat = gt_list[0].latitude
        origin_lon = gt_list[0].longitude
        origin_alt = gt_list[0].altitude
        local_frame = LocalFrame(origin_lat, origin_lon, origin_alt)

        targets, actual_target_names = build_window_targets(
            window_time_ranges=time_ranges,
            ground_truth_list=gt_list,
            local_frame=local_frame,
            target_type="displacement_enu",
        )

        # Window-level GNSS status tagging (take status at end of window)
        window_statuses = []
        for t_start, t_end in time_ranges:
            idx = int(round(t_end * 100))
            idx = min(idx, len(status_timeline) - 1)
            window_statuses.append(status_timeline[idx].value)

        raw_trajectory_data.append({
            "trajectory_id": traj_id,
            "windows": windows,
            "targets": targets,
            "time_ranges": time_ranges,
            "statuses": window_statuses,
            "feature_names": actual_feature_names,
            "target_names": actual_target_names,
        })
        print(f"  - {traj_id}: {len(windows)} windows extracted.")

    # 3. Trajectory-Aware Partitioning (Before Normalization!)
    print("\n[2/7] Executing Trajectory-Aware Train/Val/Test Partitioning...")
    # Train: sessions 01, 02, 03 (60%); Val: session 05 (20%); Test: session 04 (20%)
    train_raw = [d for d in raw_trajectory_data if d["trajectory_id"] in ["trajectory_session_01", "trajectory_session_02", "trajectory_session_03"]]
    val_raw = [d for d in raw_trajectory_data if d["trajectory_id"] == "trajectory_session_05"]
    test_raw = [d for d in raw_trajectory_data if d["trajectory_id"] == "trajectory_session_04"]

    # 4. Strict Normalization Anti-Leakage Protocol
    print("\n[3/7] Computing Normalization Statistics STRICTLY from Training Trajectories...")
    all_train_features = np.vstack([d["windows"] for d in train_raw])  # [N_train, 100, D]
    feature_mean = np.mean(all_train_features, axis=(0, 1)).astype(np.float32)
    feature_std = np.std(all_train_features, axis=(0, 1)).astype(np.float32)
    feature_std[feature_std < 1e-8] = 1.0

    print(f"  - Calculated feature mean across {len(all_train_features)} train windows.")
    print(f"  - Applied normalization parameters UNCHANGED to Validation and Test splits.")

    def create_datasets(raw_list):
        ds_list = []
        for r in raw_list:
            ds = IMUWindowDataset(
                features=r["windows"],
                targets=r["targets"],
                time_ranges=r["time_ranges"],
                feature_names=r["feature_names"],
                target_names=r["target_names"],
                trajectory_id=r["trajectory_id"],
                feature_mean=feature_mean,
                feature_std=feature_std,
            )
            ds_list.append(ds)
        return ds_list

    train_ds_list = create_datasets(train_raw)
    val_ds_list = create_datasets(val_raw)
    test_ds_list = create_datasets(test_raw)

    train_concat, val_concat, test_concat, split_metadata = split_by_trajectory(
        datasets=train_ds_list + val_ds_list + test_ds_list,
        train_ratio=0.60,
        val_ratio=0.20,
        test_ratio=0.20,
        seed=seed,
    )

    batch_size = ml_cfg.get("batch_size", 32)
    train_loader = DataLoader(train_concat, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_concat, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_concat, batch_size=batch_size, shuffle=False)

    print(f"  - Train Windows: {len(train_concat)}, Val Windows: {len(val_concat)}, Test Windows: {len(test_concat)}")

    # 5. Build Model & Trainer
    print("\n[4/7] Instantiating 1D CNN / TCN Drift Model & PyTorch Optimizer...")
    in_channels = train_ds_list[0].feature_dim
    output_dim = train_ds_list[0].target_dim

    model = TCNDriftModel(
        in_channels=in_channels,
        output_dim=output_dim,
        num_channels=ml_cfg.get("num_channels", [32, 64, 128]),
        kernel_size=ml_cfg.get("kernel_size", 3),
        dropout=ml_cfg.get("dropout", 0.1),
    )

    learning_rate = ml_cfg.get("learning_rate", 0.001)
    weight_decay = ml_cfg.get("weight_decay", 0.0001)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    epochs = ml_cfg.get("epochs", 50)
    patience = ml_cfg.get("early_stopping_patience", 10)
    trainer = ModelTrainer(model=model, optimizer=optimizer, patience=patience)

    # 6. Train Model
    print(f"\n[5/7] Training Model for max {epochs} epochs (Early Stopping Patience = {patience})...")
    best_pt_path = os.path.join(results_dir, "best_drift_model.pt")
    history = trainer.fit(train_loader, val_loader, epochs=epochs, checkpoint_save_path=best_pt_path)

    # Save full model checkpoint with normalization parameters and metadata
    full_checkpoint = {
        "model_config": model.get_config(),
        "model_state_dict": model.state_dict(),
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "feature_names": train_ds_list[0].feature_names,
        "target_names": train_ds_list[0].target_names,
        "best_epoch": trainer.best_epoch,
        "best_val_loss": trainer.best_val_loss,
    }
    torch.save(full_checkpoint, best_pt_path)
    print(f"  - Saved best checkpoint to: {best_pt_path}")

    # Save model config JSON
    config_json_path = os.path.join(results_dir, "model_config.json")
    with open(config_json_path, "w") as f:
        json.dump(full_checkpoint["model_config"], f, indent=2)

    # 7. Evaluate Model & Classical Baseline on Held-Out Test Trajectory
    print("\n[6/7] Evaluating AI Model & Classical Baseline on Held-Out Test Trajectory...")
    test_raw_features = test_raw[0]["windows"]  # [N_test, 100, 8]
    test_gt_targets = test_raw[0]["targets"]   # [N_test, 3]
    test_statuses = test_raw[0]["statuses"]    # List of status strings

    # Compute Classical IMU Baseline Predictions
    classical_preds = compute_dataset_classical_baseline(test_raw_features)

    # Compute AI Model Predictions
    model.eval()
    test_inputs = torch.from_numpy(
        (test_raw_features - feature_mean) / feature_std
    ).to(torch.float32)

    with torch.no_grad():
        ai_preds_tensor = model(test_inputs)
        ai_preds = ai_preds_tensor.numpy()

    # Metrics evaluation
    comp_report = evaluate_comparative_performance(
        y_true=test_gt_targets,
        y_classical=classical_preds,
        y_ai=ai_preds,
        gnss_statuses=test_statuses,
    )

    metrics_json_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_json_path, "w") as f:
        json.dump(comp_report, f, indent=2)
    print(f"  - Saved comparative metrics report to: {metrics_json_path}")

    print("\n  --- COMPARATIVE BENCHMARK METRICS (HELD-OUT TEST TRAJECTORY) ---")
    c_m = comp_report["classical_baseline"]
    a_m = comp_report["ai_model"]
    print(f"  Classical Baseline -> Horizontal RMSE: {c_m['classical_rmse_horizontal']:.4f} m | 3D RMSE: {c_m['classical_rmse_3d']:.4f} m")
    print(f"  AI Model           -> Horizontal RMSE: {a_m['ai_rmse_horizontal']:.4f} m | 3D RMSE: {a_m['ai_rmse_3d']:.4f} m")
    print(f"  Horizontal Error Improvement: {comp_report['horizontal_rmse_improvement_pct']:.2f}%")

    # 8. Generate Visualizations
    print("\n[7/7] Generating Visualizations...")

    # Plot 1: Training & Validation Loss
    fig, ax = plt.subplots(figsize=(7, 4.5))
    epochs_range = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs_range, history["train_loss"], label="Train Loss (MSE)", color="#1f77b4", linewidth=2)
    ax.plot(epochs_range, history["val_loss"], label="Val Loss (MSE)", color="#ff7f0e", linewidth=2)
    ax.axvline(trainer.best_epoch, linestyle="--", color="green", alpha=0.7, label=f"Best Epoch ({trainer.best_epoch})")
    ax.set_title("1D CNN / TCN Model Training & Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Squared Error (MSE)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "training_validation_loss.png"), dpi=300)
    plt.close()

    # Plot 2: Predicted vs True Displacement (East & North)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(test_gt_targets[:, 0], ai_preds[:, 0], alpha=0.6, color="#1f77b4", label="AI Prediction")
    axes[0].plot([test_gt_targets[:, 0].min(), test_gt_targets[:, 0].max()], [test_gt_targets[:, 0].min(), test_gt_targets[:, 0].max()], "r--", label="Ideal")
    axes[0].set_title("East Displacement: True vs Predicted")
    axes[0].set_xlabel("True Δp East (m)")
    axes[0].set_ylabel("Predicted Δp East (m)")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    axes[1].scatter(test_gt_targets[:, 1], ai_preds[:, 1], alpha=0.6, color="#2ca02c", label="AI Prediction")
    axes[1].plot([test_gt_targets[:, 1].min(), test_gt_targets[:, 1].max()], [test_gt_targets[:, 1].min(), test_gt_targets[:, 1].max()], "r--", label="Ideal")
    axes[1].set_title("North Displacement: True vs Predicted")
    axes[1].set_xlabel("True Δp North (m)")
    axes[1].set_ylabel("Predicted Δp North (m)")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "predicted_vs_true_displacement.png"), dpi=300)
    plt.close()

    # Plot 3: Baseline vs AI Error Over Time
    c_err = np.linalg.norm(test_gt_targets[:, :2] - classical_preds[:, :2], axis=1)
    ai_err = np.linalg.norm(test_gt_targets[:, :2] - ai_preds[:, :2], axis=1)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    window_indices = np.arange(len(c_err))
    ax.plot(window_indices, c_err, label="Classical Integration Error (m)", color="#d62728", alpha=0.8)
    ax.plot(window_indices, ai_err, label="AI Model Prediction Error (m)", color="#1f77b4", alpha=0.8)
    ax.set_title("Window Displacement Error Comparison (Held-Out Test Trajectory)")
    ax.set_xlabel("Window Index")
    ax.set_ylabel("Horizontal Error (m)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "baseline_vs_ai_error.png"), dpi=300)
    plt.close()

    # Plot 4: Error Distribution Histogram
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(c_err, bins=25, alpha=0.6, color="#d62728", label="Classical Baseline Error")
    ax.hist(ai_err, bins=25, alpha=0.6, color="#1f77b4", label="AI Model Error")
    ax.set_title("Error Distribution: Classical Baseline vs AI Model")
    ax.set_xlabel("Horizontal Error (m)")
    ax.set_ylabel("Frequency")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "error_distribution.png"), dpi=300)
    plt.close()

    print(f"  - Saved diagnostic plots to: {results_dir}")

    print("\n" + "=" * 80)
    print("STEP 7 AI/ML DRIFT-CORRECTION TRAINING EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
