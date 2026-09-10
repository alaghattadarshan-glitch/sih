#!/usr/bin/env python3
"""Step 11 — Multi-Session Real Data Benchmark & TCN Generalization Experiment.

Executes:
1. Session discovery and independent quality auditing.
2. TCN model generalization evaluation on held-out sessions.
3. Multi-duration GNSS outage navigation benchmarking (10s, 30s, 60s).
4. GNSS post-outage recovery tracking (0s, 1s, 2s, 5s).
5. Preliminary SIH target checks (< 10% drift threshold).
6. Domain-shift feature distribution analysis.
7. Consolidated report generation (summary.json, summary.csv) and 8 visualization figures.
"""

import os
import sys
import json
import csv
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import torch

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.session import (
    DataSourceType,
    DatasetSession,
    discover_available_sessions,
    create_synthetic_multi_session_catalog,
)
from src.ml.models.tcn_drift import TCNDriftModel
from src.ml.inference.predictor import DriftPredictor
from src.ml.datasets.real_session_builder import MultiSessionDatasetBuilder
from src.ml.training.trainer import ModelTrainer
from src.evaluation.multi_session_evaluator import (
    evaluate_tcn_session_generalization,
    run_session_navigation_benchmark,
    generate_domain_shift_analysis,
    generate_multi_session_benchmark_plots,
)


def load_or_train_benchmark_model(
    train_sessions: List[DatasetSession],
    val_sessions: List[DatasetSession],
    checkpoint_path: Path,
) -> Tuple[TCNDriftModel, np.ndarray, np.ndarray]:
    """Load existing model checkpoint or train on specified training sessions."""
    builder = MultiSessionDatasetBuilder()
    train_ds, feat_mean, feat_std = builder.build_dataset_from_sessions(train_sessions)
    val_ds, _, _ = builder.build_dataset_from_sessions(val_sessions, feat_mean, feat_std)

    if checkpoint_path.exists():
        print(f"Loading existing model checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = TCNDriftModel(
            in_channels=checkpoint.get("in_channels", 8),
            output_dim=checkpoint.get("output_dim", 3),
            num_channels=checkpoint.get("num_channels", [32, 64, 128]),
            kernel_size=checkpoint.get("kernel_size", 3),
            dropout=checkpoint.get("dropout", 0.1),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        if "feature_mean" in checkpoint and "feature_std" in checkpoint:
            feat_mean = checkpoint["feature_mean"]
            feat_std = checkpoint["feature_std"]
        return model, feat_mean, feat_std

    print("Checkpoint not found. Training TCN on training sessions...")
    model = TCNDriftModel(in_channels=8, output_dim=3, num_channels=[32, 64, 128], kernel_size=3, dropout=0.1)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=32, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    trainer = ModelTrainer(
        model=model,
        optimizer=optimizer,
        patience=10,
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    history = trainer.fit(train_loader=train_loader, val_loader=val_loader, epochs=25, checkpoint_save_path=str(checkpoint_path))
    model.eval()

    # Save checkpoint with normalization metadata and model config
    torch.save({
        "model_state_dict": model.state_dict(),
        "feature_mean": feat_mean,
        "feature_std": feat_std,
        "model_config": {
            "in_channels": 8,
            "output_dim": 3,
            "num_channels": [32, 64, 128],
            "kernel_size": 3,
            "dropout": 0.1,
        },
    }, checkpoint_path)

    return model, feat_mean, feat_std


def main():
    print("\n" + "=" * 80)
    print("SIH26168 STEP 11 — MULTI-SESSION BENCHMARK & TCN GENERALIZATION")
    print("=" * 80 + "\n")

    output_dir = Path("results/multi_session")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Discover Sessions
    print("[1/6] Discovering Dataset Sessions...")
    disk_sessions = discover_available_sessions()
    synthetic_sessions = create_synthetic_multi_session_catalog(num_sessions=5, base_seed=100, duration_sec=120.0)
    all_sessions = disk_sessions + synthetic_sessions
    print(f"  • Total Sessions Loaded: {len(all_sessions)}")
    for s in all_sessions:
        print(f"    - {s.session_id} ({s.source_type.value}): {s.duration_sec:.1f}s | {len(s.imu_observations)} IMU | {len(s.gnss_observations)} GNSS")

    # 2. Partition Sessions for Generalization Testing
    # Sessions 001, 002, 003 -> Train
    # Session 004 -> Validation
    # Session 005, sample_01 -> Held-out Unseen Test Sessions
    train_sessions = [s for s in all_sessions if s.session_id in ("session_001", "session_002", "session_003")]
    val_sessions = [s for s in all_sessions if s.session_id == "session_004"]
    test_sessions = [s for s in all_sessions if s.session_id in ("session_005", "sample_01")]

    # 3. Model Preparation
    print("\n[2/6] Preparing TCN Drift Model & Normalization Parameters...")
    checkpoint_path = Path("checkpoints/best_drift_model.pt")
    model, feat_mean, feat_std = load_or_train_benchmark_model(train_sessions, val_sessions, checkpoint_path)
    predictor = DriftPredictor(str(checkpoint_path))

    # 4. TCN Standalone Generalization Evaluation
    print("\n[3/6] Evaluating TCN Standalone Generalization on Unseen Test Sessions...")
    generalization_results = []
    for s in all_sessions:
        gen_eval = evaluate_tcn_session_generalization(s, predictor)
        generalization_results.append(gen_eval)
        is_held_out = s.session_id in [ts.session_id for ts in test_sessions]
        tag = "[HELD-OUT TEST]" if is_held_out else "[TRAIN/VAL]"
        if gen_eval["status"] == "SUCCESS":
            print(f"  • {s.session_id:<12} {tag:<16} | AI 2D RMSE: {gen_eval['ai_2d_rmse_m']:.4f}m | Classical 2D RMSE: {gen_eval['classical_2d_rmse_m']:.4f}m | Improvement: {gen_eval['improvement_over_classical_pct']:+.1f}%")

    # 5. Multi-Session Navigation & Outage Benchmarking
    print("\n[4/6] Executing Navigation Pipeline & GNSS Outage Benchmarks across Sessions...")
    session_nav_results = []
    for s in all_sessions:
        sess_dir = output_dir / s.session_id
        sess_dir.mkdir(parents=True, exist_ok=True)
        
        # Save session quality & calibration
        with open(sess_dir / "dataset_quality.json", "w") as f:
            json.dump(s.quality_report, f, indent=2)

        nav_res = run_session_navigation_benchmark(
            session=s,
            predictor=predictor,
            outage_durations=[10.0, 30.0, 60.0],
            outage_start_sec=30.0,
        )
        session_nav_results.append(nav_res)

        with open(sess_dir / "outage_metrics.json", "w") as f:
            json.dump(nav_res, f, indent=2)

    # 6. Domain Shift Analysis
    print("\n[5/6] Performing Feature Distribution Domain-Shift Analysis...")
    domain_shift_rep = generate_domain_shift_analysis(
        baseline_sessions=train_sessions,
        target_sessions=test_sessions,
        output_dir=output_dir / "domain_shift",
    )

    # 7. Consolidated Reporting & Visualizations
    print("\n[6/6] Generating Consolidated Reports and Summary Plots...")
    generate_multi_session_benchmark_plots(session_nav_results, generalization_results, output_dir)

    # Write summary.json
    summary_data = {
        "benchmark_timestamp": "2026-09-09",
        "total_sessions": len(all_sessions),
        "real_sessions_count": len([s for s in all_sessions if s.source_type == DataSourceType.REAL]),
        "synthetic_sessions_count": len([s for s in all_sessions if s.source_type == DataSourceType.SYNTHETIC]),
        "generalization_evaluations": generalization_results,
        "navigation_evaluations": session_nav_results,
        "domain_shift_summary": domain_shift_rep,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary_data, f, indent=2)

    # Write summary.csv for easy spreadsheet comparison
    csv_path = output_dir / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Session_ID", "Source", "Duration_s", "Outage_Duration_s",
            "Distance_Travelled_m", "ESKF_RMSE_m", "AI_RMSE_m",
            "ESKF_Drift_Pct", "AI_Drift_Pct", "SIH_Target_Check", "Recovery_5s_Error_m"
        ])
        for nr in session_nav_results:
            sess_id = nr["session_id"]
            src = nr["source_type"]
            dur = nr["duration_sec"]
            for out_name, out_eval in nr["outage_evaluations"].items():
                out_dur = out_eval["outage_duration_sec"]
                dist = out_eval["distance_travelled_during_outage_m"]
                scs = out_eval["scenarios"]
                eskf_sc = scs.get("Scenario_B_ESKF_Outage", {})
                ai_sc = scs.get("Scenario_C_ESKF_AI", {})
                
                writer.writerow([
                    sess_id, src, dur, out_dur, dist,
                    eskf_sc.get("outage_rmse_m", "N/A"),
                    ai_sc.get("outage_rmse_m", "N/A"),
                    eskf_sc.get("drift_percentage", "N/A"),
                    ai_sc.get("drift_percentage", "N/A"),
                    ai_sc.get("preliminary_sih_target_check", {}).get("status", "N/A"),
                    ai_sc.get("recovery", {}).get("error_at_recovery_5s_m", "N/A"),
                ])

    print("\n" + "=" * 115)
    print(f"{'SESSION ID':<12} | {'OUTAGE':<8} | {'DIST (m)':<9} | {'ESKF RMSE':<10} | {'AI RMSE':<9} | {'ESKF DRIFT':<11} | {'AI DRIFT':<9} | {'SIH TARGET':<10}")
    print("=" * 115)
    for nr in session_nav_results:
        sess_id = nr["session_id"]
        for out_name, out_eval in nr["outage_evaluations"].items():
            out_dur_str = f"{int(out_eval['outage_duration_sec'])}s"
            dist_str = f"{out_eval['distance_travelled_during_outage_m']:.1f}"
            scs = out_eval["scenarios"]
            eskf_rmse_str = f"{scs.get('Scenario_B_ESKF_Outage', {}).get('outage_rmse_m', 0.0):.2f}m"
            ai_rmse_str = f"{scs.get('Scenario_C_ESKF_AI', {}).get('outage_rmse_m', 0.0):.2f}m"
            eskf_drift_str = f"{scs.get('Scenario_B_ESKF_Outage', {}).get('drift_percentage', 0.0):.1f}%"
            ai_drift_str = f"{scs.get('Scenario_C_ESKF_AI', {}).get('drift_percentage', 0.0):.1f}%"
            sih_status = scs.get('Scenario_C_ESKF_AI', {}).get('preliminary_sih_target_check', {}).get('status', 'N/A')
            print(f"{sess_id:<12} | {out_dur_str:<8} | {dist_str:<9} | {eskf_rmse_str:<10} | {ai_rmse_str:<9} | {eskf_drift_str:<11} | {ai_drift_str:<9} | {sih_status:<10}")
    print("=" * 115)

    print(f"\nConsolidated results and plots saved under: {output_dir.resolve()}")
    print("\n" + "=" * 80)
    print("STEP 11 MULTI-SESSION BENCHMARK COMPLETED SUCCESSFULLY!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
