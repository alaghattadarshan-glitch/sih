"""Map Matching and Road Network Constraint Evaluation Script (Step 9).

Evaluates and compares navigation performance across 4 scenarios:
- Scenario A: Open-loop INS
- Scenario B: ESKF Outage (AI Off, Map Off)
- Scenario C: ESKF + AI Outage (Map Off)
- Scenario D: ESKF + AI + Map Matching Outage

Generates metrics tables, JSON summaries, and diagnostic visualization plots.
"""

import os
import sys
import json
import math
from typing import Dict, List, Tuple, Any
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.loaders.synthetic import generate_synthetic_trajectory
from src.coordinate_transforms import LocalFrame
from src.data.observations import IMUObservation, GNSSObservation, GroundTruthObservation
from src.navigation.state import NavigationState
from src.navigation.ins import StrapdownINS
from src.navigation.ekf.eskf import ErrorStateKalmanFilter
from src.outage_detection.detector import GNSSOutageDetector, GNSSStatus
from src.ml.inference import DriftPredictor
from src.map_matching.network import RoadNetwork
from src.map_matching.matcher import MapMatcher, MapMatcherConfig
from src.map_matching.synthetic_network import build_synthetic_road_network
from src.map_matching.geometry import point_to_polyline_projection
from src.navigation.ai_fusion import AIESKFPipeline


def compute_road_distance(pos_enu: np.ndarray, network: RoadNetwork) -> float:
    """Calculate minimum distance from 3D position to any segment in road network."""
    min_d = float("inf")
    for seg in network.segments.values():
        _, d, _, _ = point_to_polyline_projection(pos_enu, seg.polyline_enu)
        if d < min_d:
            min_d = d
    return min_d


def run_scenario_a_ins(
    imu_list: List[IMUObservation],
    gt_list: List[GroundTruthObservation],
    local_frame: LocalFrame,
) -> Dict[str, Any]:
    """Run Scenario A: Pure Open-Loop INS Dead Reckoning."""
    ins = StrapdownINS()
    gt0 = gt_list[0]
    e0, n0, u0 = local_frame.to_enu(gt0.latitude, gt0.longitude, gt0.altitude)
    ins.initialize(
        initial_time=gt0.timestamp,
        initial_llh=(gt0.latitude, gt0.longitude, gt0.altitude),
        initial_velocity_enu=np.array([gt0.velocity_east, gt0.velocity_north, gt0.velocity_up]),
        initial_quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
    )

    timestamps = []
    positions = []
    errors = []

    for imu, gt in zip(imu_list, gt_list):
        state = ins.update(imu)
        gt_e, gt_n, gt_u = local_frame.to_enu(gt.latitude, gt.longitude, gt.altitude)
        gt_pos = np.array([gt_e, gt_n, gt_u])
        err = float(np.linalg.norm(state.position_enu[:2] - gt_pos[:2]))

        timestamps.append(imu.timestamp)
        positions.append(state.position_enu.copy())
        errors.append(err)

    return {
        "timestamps": np.array(timestamps),
        "positions": np.array(positions),
        "errors": np.array(errors),
    }


def run_filter_pipeline(
    imu_list: List[IMUObservation],
    gnss_list: List[GNSSObservation],
    gt_list: List[GroundTruthObservation],
    local_frame: LocalFrame,
    predictor: Optional[DriftPredictor],
    network: Optional[RoadNetwork],
    enable_ai: bool,
    enable_map: bool,
    outage_start: float = 30.0,
    outage_end: float = 60.0,
) -> Dict[str, Any]:
    """Run integrated ESKF / AI / Map matching pipeline on trajectory."""
    eskf = ErrorStateKalmanFilter()
    gt0 = gt_list[0]
    eskf.initialize(
        initial_time=gt0.timestamp,
        initial_llh=(gt0.latitude, gt0.longitude, gt0.altitude),
        initial_velocity_enu=np.array([gt0.velocity_east, gt0.velocity_north, gt0.velocity_up]),
        initial_quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
    )

    detector = GNSSOutageDetector({
        "max_timestamp_gap_sec": 2.0,
        "outage_horizontal_accuracy_m": 15.0,
        "persistence_count": 2,
    })

    matcher = MapMatcher() if enable_map and network is not None else None

    pipeline = AIESKFPipeline(
        eskf=eskf,
        detector=detector,
        predictor=predictor,
        enable_ai=enable_ai,
        network=network,
        matcher=matcher,
        enable_map_matching=enable_map,
        r_ai_std=(1.5, 1.5, 3.0),
        r_map_std=(2.0, 2.0, 5.0),
        gate_threshold=4.0,
        map_gate_threshold=4.0,
    )

    gnss_map = {round(g.timestamp, 3): g for g in gnss_list}

    timestamps = []
    positions = []
    errors = []
    statuses = []
    road_distances = []

    for imu, gt in zip(imu_list, gt_list):
        t = imu.timestamp
        # Check if 1Hz GNSS observation is available at this time
        t_key = round(t, 3)
        gnss_obs = gnss_map.get(t_key, None)

        # Force GNSS outage during [outage_start, outage_end)
        if outage_start <= t < outage_end:
            gnss_obs = None

        state, status = pipeline.process_sample(imu, gnss_obs)

        gt_e, gt_n, gt_u = local_frame.to_enu(gt.latitude, gt.longitude, gt.altitude)
        gt_pos = np.array([gt_e, gt_n, gt_u])
        err = float(np.linalg.norm(state.position_enu[:2] - gt_pos[:2]))

        timestamps.append(t)
        positions.append(state.position_enu.copy())
        errors.append(err)
        statuses.append(status.value)

        if network is not None:
            r_dist = compute_road_distance(state.position_enu, network)
            road_distances.append(r_dist)
        else:
            road_distances.append(err)

    return {
        "timestamps": np.array(timestamps),
        "positions": np.array(positions),
        "errors": np.array(errors),
        "statuses": statuses,
        "road_distances": np.array(road_distances),
        "pipeline": pipeline,
    }


def main():
    print("=" * 80)
    print("SIH26168 STEP 9 — MAP MATCHING & ROAD NETWORK CONSTRAINT EVALUATION")
    print("=" * 80)

    results_dir = "results/map_matching"
    os.makedirs(results_dir, exist_ok=True)

    # 1. Generate standard 80s synthetic trajectory
    print("\n[1/4] Generating Standard Synthetic Trajectory & Local Road Network...")
    origin_lat, origin_lon, origin_alt = 12.9716, 77.5946, 920.0
    local_frame = LocalFrame(origin_lat, origin_lon, origin_alt)

    imu_list, gnss_list, gt_list = generate_synthetic_trajectory(
        seed=42, duration_sec=80.0, origin_lat=origin_lat, origin_lon=origin_lon, origin_alt=origin_alt
    )
    road_network = build_synthetic_road_network(local_frame)

    ckpt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "ml_training", "best_drift_model.pt")
    if os.path.exists(ckpt_path):
        print(f"Loaded AI DriftPredictor from: {ckpt_path}")
        predictor = DriftPredictor(ckpt_path)
    else:
        print("Warning: AI checkpoint not found.")
        predictor = None

    # 2. Run All 4 Scenarios
    print("\n[2/4] Executing Navigation Evaluation Scenarios (Outage: 30s -> 60s)...")
    
    # Scenario A: Open-Loop INS
    res_a = run_scenario_a_ins(imu_list, gt_list, local_frame)
    # Scenario B: ESKF Outage (AI Off, Map Off)
    res_b = run_filter_pipeline(imu_list, gnss_list, gt_list, local_frame, predictor, road_network, enable_ai=False, enable_map=False)
    # Scenario C: ESKF + AI Outage (Map Off)
    res_c = run_filter_pipeline(imu_list, gnss_list, gt_list, local_frame, predictor, road_network, enable_ai=True, enable_map=False)
    # Scenario D: ESKF + AI + Map Matching Outage
    res_d = run_filter_pipeline(imu_list, gnss_list, gt_list, local_frame, predictor, road_network, enable_ai=True, enable_map=True)

    # 3. Compute Metrics
    t_arr = res_b["timestamps"]
    outage_mask = (t_arr >= 30.0) & (t_arr <= 60.0)
    idx_45s = int(np.argmin(np.abs(t_arr - 45.0)))
    idx_60s = int(np.argmin(np.abs(t_arr - 60.0)))
    idx_80s = len(t_arr) - 1

    scenarios = [
        ("Scenario A (Open-Loop INS)", res_a),
        ("Scenario B (ESKF only)", res_b),
        ("Scenario C (ESKF + AI)", res_c),
        ("Scenario D (ESKF + AI + Map)", res_d),
    ]

    metrics_summary = {}

    print("\n[3/4] SCENARIO COMPARISON BENCHMARK TABLE")
    print("-" * 105)
    print(f"{'System Scenario':<30} | {'Outage RMSE':>11} | {'Err @ 45s':>10} | {'Err @ 60s':>10} | {'Max Outage Err':>14} | {'Road Dist (Avg)':>15}")
    print("-" * 105)

    for name, res in scenarios:
        errs = res["errors"]
        outage_rmse = float(np.sqrt(np.mean(errs[outage_mask] ** 2)))
        err_45s = float(errs[idx_45s])
        err_60s = float(errs[idx_60s])
        max_outage_err = float(np.max(errs[outage_mask]))
        final_err = float(errs[idx_80s])

        road_dists = res.get("road_distances", errs)
        avg_road_dist = float(np.mean(road_dists[outage_mask]))
        max_road_dist = float(np.max(road_dists[outage_mask]))

        metrics_summary[name] = {
            "outage_rmse_m": outage_rmse,
            "error_at_45s_m": err_45s,
            "error_at_60s_m": err_60s,
            "max_outage_error_m": max_outage_err,
            "final_error_m": final_err,
            "avg_road_distance_m": avg_road_dist,
            "max_road_distance_m": max_road_dist,
        }

        print(f"{name:<30} | {outage_rmse:>9.2f} m | {err_45s:>8.2f} m | {err_60s:>8.2f} m | {max_outage_err:>12.2f} m | {avg_road_dist:>13.2f} m")

    print("-" * 105)

    # Map Matching Specific Diagnostics
    pipe_d: AIESKFPipeline = res_d["pipeline"]
    print(f"\nMap Matching Gating Statistics (Scenario D):")
    print(f"  - AI Updates Total     : {pipe_d.total_ai_updates} (Accepted: {pipe_d.ai_accepted_count}, Rejected: {pipe_d.ai_rejected_count}, Rate: {pipe_d.acceptance_rate:.1f}%)")
    print(f"  - Map Updates Total    : {pipe_d.total_map_updates} (Accepted: {pipe_d.map_accepted_count}, Rejected: {pipe_d.map_rejected_count}, Rate: {pipe_d.map_acceptance_rate:.1f}%)")

    map_hist = pipe_d.map_update_history
    if map_hist:
        avg_heading_err = float(np.mean([h["heading_error"] for h in map_hist if h["heading_error"] is not None]))
        avg_conf = float(np.mean([h["confidence"] for h in map_hist if h["confidence"] is not None]))
        print(f"  - Avg Heading Mismatch : {avg_heading_err:.2f}°")
        print(f"  - Avg Match Confidence : {avg_conf:.3f}")

    # Save metrics JSON
    json_path = os.path.join(results_dir, "map_matching_metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)

    # 4. Generate All 7 Diagnostic Plots
    print("\n[4/4] Generating Diagnostic Visualization Plots...")

    # Plot 1: road_network.png
    plt.figure(figsize=(10, 6))
    for seg_id, seg in road_network.segments.items():
        poly = seg.polyline_enu
        style = "--" if "parallel" in seg_id else ("-." if "crossing" in seg_id else "-")
        color = "gray" if "parallel" in seg_id else ("orange" if "crossing" in seg_id else "blue")
        width = 2.5 if "main" in seg_id else 1.5
        plt.plot(poly[:, 0], poly[:, 1], style, color=color, linewidth=width, label=f"{seg.segment_id} ({seg.road_class.value})")
        # Annotate segment name
        mid_pt = poly[len(poly)//2]
        plt.text(mid_pt[0], mid_pt[1] + 3.0, seg.segment_id, fontsize=7, alpha=0.8)

    for node_id, node in road_network.nodes.items():
        plt.scatter(node.point.east_m, node.point.north_m, color="black", s=30, zorder=5)

    plt.title("Step 9 Deterministic Synthetic Road Network (Local ENU)", fontsize=13, fontweight="bold")
    plt.xlabel("East (meters)")
    plt.ylabel("North (meters)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "road_network.png"), dpi=300)
    plt.close()

    # Plot 2: trajectory_map_matching.png
    plt.figure(figsize=(11, 7))
    # Plot Road Network base
    for seg in road_network.segments.values():
        plt.plot(seg.polyline_enu[:, 0], seg.polyline_enu[:, 1], "k--", alpha=0.35, linewidth=1.0)

    # Extract Ground Truth
    gt_e = [local_frame.to_enu(g.latitude, g.longitude, g.altitude)[0] for g in gt_list]
    gt_n = [local_frame.to_enu(g.latitude, g.longitude, g.altitude)[1] for g in gt_list]
    plt.plot(gt_e, gt_n, "g-", linewidth=2.5, label="Ground Truth Path")

    # Plot Trajectories
    plt.plot(res_b["positions"][:, 0], res_b["positions"][:, 1], "r:", linewidth=1.8, label="ESKF Only (No AI/Map)")
    plt.plot(res_c["positions"][:, 0], res_c["positions"][:, 1], "m--", linewidth=1.8, label="ESKF + AI Drift Correction")
    plt.plot(res_d["positions"][:, 0], res_d["positions"][:, 1], "b-", linewidth=2.0, label="ESKF + AI + Map Matching")

    # Mark Outage Start and End
    plt.scatter([gt_e[3000]], [gt_n[3000]], color="red", marker="X", s=90, zorder=6, label="Outage Start (t=30s)")
    plt.scatter([gt_e[6000]], [gt_n[6000]], color="purple", marker="o", s=90, zorder=6, label="Outage End (t=60s)")

    plt.title("2D Trajectory Comparison with Road Network Constraints (30s–60s Outage)", fontsize=13, fontweight="bold")
    plt.xlabel("East (meters)")
    plt.ylabel("North (meters)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "trajectory_map_matching.png"), dpi=300)
    plt.close()

    # Plot 3: position_error_comparison.png
    plt.figure(figsize=(10, 5))
    plt.plot(t_arr, res_b["errors"], "r:", linewidth=1.8, label="Scenario B: ESKF Only (No AI/Map)")
    plt.plot(t_arr, res_c["errors"], "m--", linewidth=1.8, label="Scenario C: ESKF + AI")
    plt.plot(t_arr, res_d["errors"], "b-", linewidth=2.0, label="Scenario D: ESKF + AI + Map Matching")
    plt.axvspan(30.0, 60.0, color="orange", alpha=0.18, label="GNSS Outage Window (30s - 60s)")

    plt.title("Horizontal Position Error Comparison Across System Scenarios", fontsize=13, fontweight="bold")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Horizontal Error (meters)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "position_error_comparison.png"), dpi=300)
    plt.close()

    # Plot 4: distance_to_road.png
    plt.figure(figsize=(10, 5))
    plt.plot(t_arr, res_b["road_distances"], "r:", linewidth=1.5, label="ESKF Only")
    plt.plot(t_arr, res_c["road_distances"], "m--", linewidth=1.5, label="ESKF + AI")
    plt.plot(t_arr, res_d["road_distances"], "b-", linewidth=2.0, label="ESKF + AI + Map Matching")
    plt.axvspan(30.0, 60.0, color="orange", alpha=0.18, label="GNSS Outage Window")

    plt.title("Orthogonal Distance from Estimated Position to Road Geometry", fontsize=13, fontweight="bold")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Distance to Road (meters)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "distance_to_road.png"), dpi=300)
    plt.close()

    # Plot 5: heading_alignment.png
    plt.figure(figsize=(10, 5))
    if map_hist:
        ts_m = [h["timestamp"] for h in map_hist]
        h_errs = [h["heading_error"] for h in map_hist]
        plt.plot(ts_m, h_errs, "c-o", markersize=4, linewidth=1.5, label="Heading Error |ψ_veh - ψ_road|")
        plt.axhline(60.0, color="r", linestyle="--", label="Max Heading Error Gate (60°)")
    plt.axvspan(30.0, 60.0, color="orange", alpha=0.18, label="GNSS Outage Window")

    plt.title("Vehicle Heading Alignment with Matched Road Segment Tangent", fontsize=13, fontweight="bold")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Heading Discrepancy (degrees)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "heading_alignment.png"), dpi=300)
    plt.close()

    # Plot 6: candidate_scores.png
    plt.figure(figsize=(10, 5))
    if map_hist:
        ts_m = [h["timestamp"] for h in map_hist]
        confs = [h["confidence"] for h in map_hist]
        mah_ds = [h["mahalanobis_distance"] for h in map_hist]
        plt.plot(ts_m, confs, "g-s", markersize=4, label="Map Match Candidate Confidence [0, 1]")
        plt.plot(ts_m, [m / 4.0 for m in mah_ds], "m-^", markersize=4, label="Normalized Mahalanobis Distance (d_M / 4.0σ)")
        plt.axhline(0.25, color="gray", linestyle=":", label="Min Confidence Gate (0.25)")
    plt.axvspan(30.0, 60.0, color="orange", alpha=0.18, label="GNSS Outage Window")

    plt.title("Map Matching Candidate Scoring, Confidence & Innovation Metrics", fontsize=13, fontweight="bold")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Normalized Metric Value")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "candidate_scores.png"), dpi=300)
    plt.close()

    # Plot 7: map_update_acceptance.png
    plt.figure(figsize=(10, 5))
    if map_hist:
        ts_m = [h["timestamp"] for h in map_hist]
        accepted = [1 if h["accepted"] else 0 for h in map_hist]
        plt.scatter(ts_m, accepted, c=["green" if a == 1 else "red" for a in accepted], s=60, zorder=5)
        plt.yticks([0, 1], ["Rejected (Gated)", "Accepted (Soft Update)"])
        plt.ylim(-0.3, 1.3)
    plt.axvspan(30.0, 60.0, color="orange", alpha=0.18, label="GNSS Outage Window")

    plt.title("Soft Map Constraint Innovation Gating & Acceptance Timeline", fontsize=13, fontweight="bold")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Update Status")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "map_update_acceptance.png"), dpi=300)
    plt.close()

    print(f"Saved all 7 diagnostic plots to: {os.path.abspath(results_dir)}")
    print("\n" + "=" * 80)
    print("STEP 9 MAP MATCHING EVALUATION COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
