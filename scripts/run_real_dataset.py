"""Real Dataset Ingestion, GNSS Outage Simulation, and Navigation Evaluation Runner (Step 10).

Executes end-to-end ingestion and sensor fusion evaluation on real smartphone,
vehicle logger, or benchmark datasets (such as IO-VNBD).

Usage:
    python scripts/run_real_dataset.py --config config/datasets/example_generic_csv.yaml
"""

import os
import sys
import argparse
import json
import yaml
import math
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.observations import IMUObservation, GNSSObservation, GroundTruthObservation
from src.coordinate_transforms import LocalFrame
from src.data.adapters.schema import (
    DatasetConfig,
    IMUColumnMapping,
    GNSSColumnMapping,
    GroundTruthColumnMapping,
    StaticCalibrationConfig,
    OutageSimulationConfig,
    TimestampUnit,
    AccelUnit,
    GyroUnit,
    AxisTransformConfig,
)
from src.data.adapters.generic_csv import GenericCSVAdapter
from src.data.adapters.io_vnbd import IOVNBDAdapter
from src.evaluation.dataset_quality import (
    generate_dataset_quality_report,
    validate_initial_calibration,
)
from src.calibration.imu_calibration import apply_imu_calibration
from src.navigation.ins import StrapdownINS
from src.navigation.ekf.eskf import ErrorStateKalmanFilter
from src.outage_detection.detector import GNSSOutageDetector, GNSSStatus
from src.ml.inference import DriftPredictor
from src.map_matching.network import RoadNetwork
from src.map_matching.matcher import MapMatcher
from src.map_matching.synthetic_network import build_synthetic_road_network
from src.navigation.ai_fusion import AIESKFPipeline


def load_dataset_config(config_path: str) -> DatasetConfig:
    """Load DatasetConfig from YAML file."""
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    imu_raw = raw.get("imu_mapping", {})
    axis_raw = imu_raw.get("axis_transform", {})
    axis_cfg = AxisTransformConfig(
        x_map=axis_raw.get("x_map", "+x"),
        y_map=axis_raw.get("y_map", "+y"),
        z_map=axis_raw.get("z_map", "+z"),
    )

    imu_mapping = IMUColumnMapping(
        timestamp=imu_raw.get("timestamp", "timestamp"),
        accel_x=imu_raw.get("accel_x", "accel_x"),
        accel_y=imu_raw.get("accel_y", "accel_y"),
        accel_z=imu_raw.get("accel_z", "accel_z"),
        gyro_x=imu_raw.get("gyro_x", "gyro_x"),
        gyro_y=imu_raw.get("gyro_y", "gyro_y"),
        gyro_z=imu_raw.get("gyro_z", "gyro_z"),
        mag_x=imu_raw.get("mag_x", None),
        mag_y=imu_raw.get("mag_y", None),
        mag_z=imu_raw.get("mag_z", None),
        timestamp_unit=TimestampUnit(imu_raw.get("timestamp_unit", "s")),
        accel_unit=AccelUnit(imu_raw.get("accel_unit", "m/s^2")),
        gyro_unit=GyroUnit(imu_raw.get("gyro_unit", "rad/s")),
        axis_transform=axis_cfg,
    )

    gnss_mapping = None
    if "gnss_mapping" in raw and raw["gnss_mapping"]:
        g_raw = raw["gnss_mapping"]
        gnss_mapping = GNSSColumnMapping(
            timestamp=g_raw.get("timestamp", "timestamp"),
            latitude=g_raw.get("latitude", "latitude"),
            longitude=g_raw.get("longitude", "longitude"),
            altitude=g_raw.get("altitude", "altitude"),
            velocity_east=g_raw.get("velocity_east", None),
            velocity_north=g_raw.get("velocity_north", None),
            velocity_up=g_raw.get("velocity_up", None),
            speed=g_raw.get("speed", None),
            heading=g_raw.get("heading", None),
            horizontal_accuracy=g_raw.get("horizontal_accuracy", None),
            vertical_accuracy=g_raw.get("vertical_accuracy", None),
            timestamp_unit=TimestampUnit(g_raw.get("timestamp_unit", "s")),
        )

    gt_mapping = None
    if "ground_truth_mapping" in raw and raw["ground_truth_mapping"]:
        gt_raw = raw["ground_truth_mapping"]
        gt_mapping = GroundTruthColumnMapping(
            timestamp=gt_raw.get("timestamp", "timestamp"),
            latitude=gt_raw.get("latitude", "latitude"),
            longitude=gt_raw.get("longitude", "longitude"),
            altitude=gt_raw.get("altitude", "altitude"),
            velocity_east=gt_raw.get("velocity_east", None),
            velocity_north=gt_raw.get("velocity_north", None),
            velocity_up=gt_raw.get("velocity_up", None),
            speed=gt_raw.get("speed", None),
            heading=gt_raw.get("heading", None),
            roll=gt_raw.get("roll", None),
            pitch=gt_raw.get("pitch", None),
            yaw=gt_raw.get("yaw", None),
            timestamp_unit=TimestampUnit(gt_raw.get("timestamp_unit", "s")),
        )

    calib_raw = raw.get("calibration", {})
    calib_cfg = StaticCalibrationConfig(
        enabled=calib_raw.get("enabled", False),
        start_time_sec=calib_raw.get("start_time_sec", 0.0),
        duration_sec=calib_raw.get("duration_sec", 5.0),
        max_accel_std_mps2=calib_raw.get("max_accel_std_mps2", 0.4),
        max_gyro_std_rads=calib_raw.get("max_gyro_std_rads", 0.05),
    )

    outage_raw = raw.get("outage", {})
    outage_cfg = OutageSimulationConfig(
        enabled=outage_raw.get("enabled", False),
        start_time_sec=outage_raw.get("start_time_sec", 30.0),
        duration_sec=outage_raw.get("duration_sec", 30.0),
    )

    return DatasetConfig(
        dataset_name=raw.get("dataset_name", "dataset"),
        imu_file_path=raw.get("imu_file_path", ""),
        gnss_file_path=raw.get("gnss_file_path", None),
        ground_truth_file_path=raw.get("ground_truth_file_path", None),
        delimiter=raw.get("delimiter", ","),
        skip_rows=raw.get("skip_rows", 0),
        imu_mapping=imu_mapping,
        gnss_mapping=gnss_mapping,
        ground_truth_mapping=gt_mapping,
        calibration=calib_cfg,
        outage=outage_cfg,
        max_sync_tolerance_sec=raw.get("max_sync_tolerance_sec", 0.5),
        origin_lat=raw.get("origin_lat", None),
        origin_lon=raw.get("origin_lon", None),
        origin_alt=raw.get("origin_alt", None),
        metadata=raw.get("metadata", {}),
    )


def run_pipeline(
    imu_list: List[IMUObservation],
    gnss_list: List[GNSSObservation],
    local_frame: LocalFrame,
    predictor: Optional[DriftPredictor] = None,
    network: Optional[RoadNetwork] = None,
    enable_ai: bool = False,
    enable_map: bool = False,
    outage_cfg: Optional[OutageSimulationConfig] = None,
) -> Dict[str, Any]:
    """Execute navigation pipeline with optional AI and Map matching."""
    eskf = ErrorStateKalmanFilter()
    init_imu = imu_list[0]
    init_gnss = gnss_list[0] if gnss_list else None

    init_lat = local_frame.ref_lat
    init_lon = local_frame.ref_lon
    init_alt = local_frame.ref_height

    init_vel = (
        np.array([init_gnss.velocity_east or 0.0, init_gnss.velocity_north or 0.0, init_gnss.velocity_up or 0.0])
        if init_gnss
        else np.zeros(3)
    )

    eskf.initialize(
        initial_time=init_imu.timestamp,
        initial_llh=(init_lat, init_lon, init_alt),
        initial_velocity_enu=init_vel,
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

    # Index GNSS by approximate timestamp (rounded to 2 decimal places)
    gnss_by_time = {round(g.timestamp, 2): g for g in gnss_list}

    outage_start = outage_cfg.start_time_sec if outage_cfg and outage_cfg.enabled else float("inf")
    outage_end = outage_start + outage_cfg.duration_sec if outage_cfg and outage_cfg.enabled else float("inf")

    timestamps = []
    positions = []
    velocities = []
    headings = []
    statuses = []

    for imu in imu_list:
        t = imu.timestamp
        t_key = round(t, 2)
        g_fix = gnss_by_time.get(t_key, None)

        # Artificial outage removal
        if outage_start <= t < outage_end:
            g_fix = None

        state, status = pipeline.process_sample(imu, g_fix)

        timestamps.append(t)
        positions.append(state.position_enu.copy())
        velocities.append(state.velocity_enu.copy())
        headings.append(state.heading_deg())
        statuses.append(status.value)

    return {
        "timestamps": np.array(timestamps),
        "positions": np.array(positions),
        "velocities": np.array(velocities),
        "headings": np.array(headings),
        "statuses": statuses,
        "pipeline": pipeline,
    }


def main():
    parser = argparse.ArgumentParser(description="Real Dataset Ingestion & Evaluation Runner")
    parser.add_argument("--config", type=str, default="config/datasets/example_generic_csv.yaml", help="Dataset config path")
    args = parser.parse_args()

    print("=" * 80)
    print("SIH26168 STEP 10 — REAL DATASET INGESTION & EVALUATION")
    print(f"Configuration: {args.config}")
    print("=" * 80)

    # 1. Load configuration
    dataset_cfg = load_dataset_config(args.config)
    results_dir = os.path.join("results", "real_dataset", dataset_cfg.dataset_name)
    os.makedirs(results_dir, exist_ok=True)

    # 2. Select Adapter
    if "io_vnbd" in dataset_cfg.dataset_name.lower() or "vnbd" in dataset_cfg.dataset_name.lower():
        adapter = IOVNBDAdapter(dataset_cfg)
    else:
        adapter = GenericCSVAdapter(dataset_cfg)

    # Check local availability
    if not os.path.exists(dataset_cfg.imu_file_path):
        print(f"\n[Notice] Dataset file '{dataset_cfg.imu_file_path}' is not present.")
        meta = adapter.get_metadata()
        meta["execution_status"] = "PENDING_DATASET_AVAILABILITY"
        meta["required_files"] = {
            "imu": dataset_cfg.imu_file_path,
            "gnss": dataset_cfg.gnss_file_path,
            "ground_truth": dataset_cfg.ground_truth_file_path,
        }

        with open(os.path.join(results_dir, "run_metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)

        print(f"Saved standby run metadata to: {os.path.join(results_dir, 'run_metadata.json')}")
        print("IO-VNBD benchmark execution pending dataset availability.")
        return

    # 3. Load & Ingest Observations
    print("\n[1/5] Ingesting and Normalizing Sensor Observations...")
    imu_list = adapter.load_imu()
    gnss_list = adapter.load_gnss()
    gt_list = adapter.load_ground_truth()

    print(f"  - IMU Samples Loaded  : {len(imu_list)}")
    print(f"  - GNSS Fixes Loaded   : {len(gnss_list)}")
    print(f"  - Ground Truth Samples: {len(gt_list) if gt_list else 0}")

    # 4. Generate Quality Audit & Calibration Validation
    print("\n[2/5] Running Dataset Quality Audit & Calibration Checks...")
    quality_report = generate_dataset_quality_report(imu_list, gnss_list, gt_list, dataset_cfg.dataset_name)
    with open(os.path.join(results_dir, "dataset_quality.json"), "w") as f:
        json.dump(quality_report, f, indent=2)

    calib_report = validate_initial_calibration(imu_list, dataset_cfg.calibration)
    with open(os.path.join(results_dir, "calibration_report.json"), "w") as f:
        json.dump(calib_report, f, indent=2)

    print(f"  - Overall Dataset Quality Status: {quality_report['overall_status']}")
    print(f"  - Stationary Calibration Status : {calib_report['status']} ({calib_report.get('notice', '')})")

    # Apply initial static bias calibration if valid and enabled
    if calib_report.get("is_stationary", False):
        b_a = np.array(calib_report["accel_bias_mps2"])
        b_g = np.array(calib_report["gyro_bias_rads"])
        imu_list = [apply_imu_calibration(obs, b_a, b_g) for obs in imu_list]
        print("  - Applied estimated initial static sensor biases to IMU stream.")

    # 5. Coordinate Origin & ENU Conversion
    print("\n[3/5] Setting Local Coordinate Origin & WGS84 Geodetic Transforms...")
    if dataset_cfg.origin_lat is not None and dataset_cfg.origin_lon is not None:
        orig_lat = dataset_cfg.origin_lat
        orig_lon = dataset_cfg.origin_lon
        orig_alt = dataset_cfg.origin_alt if dataset_cfg.origin_alt is not None else 0.0
    elif gnss_list:
        orig_lat = gnss_list[0].latitude
        orig_lon = gnss_list[0].longitude
        orig_alt = gnss_list[0].altitude
    elif gt_list:
        orig_lat = gt_list[0].latitude
        orig_lon = gt_list[0].longitude
        orig_alt = gt_list[0].altitude
    else:
        orig_lat, orig_lon, orig_alt = 12.9716, 77.5946, 920.0

    local_frame = LocalFrame(orig_lat, orig_lon, orig_alt)
    print(f"  - Local Frame Origin: Lat={orig_lat:.6f}°, Lon={orig_lon:.6f}°, Alt={orig_alt:.2f}m")

    # 6. Run Navigation Pipelines
    print("\n[4/5] Running Navigation Fusion & GNSS Outage Simulation...")
    ckpt_path = "results/ml_training/best_drift_model.pt"
    predictor = DriftPredictor(ckpt_path) if os.path.exists(ckpt_path) else None
    road_network = build_synthetic_road_network(local_frame)

    # Scenario A: Continuous GNSS ESKF
    res_a = run_pipeline(imu_list, gnss_list, local_frame, predictor=None, enable_ai=False, enable_map=False, outage_cfg=None)
    # Scenario B: Outage INS/ESKF (No AI/Map)
    res_b = run_pipeline(imu_list, gnss_list, local_frame, predictor=None, enable_ai=False, enable_map=False, outage_cfg=dataset_cfg.outage)
    # Scenario C: Outage ESKF + AI
    res_c = run_pipeline(imu_list, gnss_list, local_frame, predictor=predictor, enable_ai=(predictor is not None), enable_map=False, outage_cfg=dataset_cfg.outage)
    # Scenario D: Outage ESKF + AI + Map
    res_d = run_pipeline(imu_list, gnss_list, local_frame, predictor=predictor, network=road_network, enable_ai=(predictor is not None), enable_map=True, outage_cfg=dataset_cfg.outage)

    # 7. Evaluate Metrics & Preliminary SIH Target Check
    print("\n[5/5] Computing Performance Metrics & SIH Target Check...")
    t_arr = res_b["timestamps"]
    has_gt = gt_list is not None and len(gt_list) > 0

    if has_gt:
        gt_map = {round(g.timestamp, 2): g for g in gt_list}
        gt_enu_list = []
        for t in t_arr:
            g = gt_map.get(round(t, 2), gt_list[min(len(gt_list)-1, int(round(t*100)))])
            e, n, u = local_frame.to_enu(g.latitude, g.longitude, g.altitude)
            gt_enu_list.append([e, n, u])
        gt_enu = np.array(gt_enu_list)
    else:
        gt_enu = None

    outage_metrics = {}
    if dataset_cfg.outage.enabled:
        o_start = dataset_cfg.outage.start_time_sec
        o_end = o_start + dataset_cfg.outage.duration_sec
        o_mid = o_start + dataset_cfg.outage.duration_sec / 2.0
        outage_mask = (t_arr >= o_start) & (t_arr <= o_end)

        idx_start = int(np.argmin(np.abs(t_arr - o_start)))
        idx_mid = int(np.argmin(np.abs(t_arr - o_mid)))
        idx_end = int(np.argmin(np.abs(t_arr - o_end)))

        # Distance travelled during outage
        if has_gt and gt_enu is not None:
            gt_diffs = np.diff(gt_enu[outage_mask, :2], axis=0)
            dist_travelled = float(np.sum(np.linalg.norm(gt_diffs, axis=1)))
        else:
            ins_diffs = np.diff(res_b["positions"][outage_mask, :2], axis=0)
            dist_travelled = float(np.sum(np.linalg.norm(ins_diffs, axis=1)))

        dist_travelled = max(1e-3, dist_travelled)

        for sc_name, res in [("Scenario B (ESKF Outage)", res_b), ("Scenario C (ESKF + AI)", res_c), ("Scenario D (ESKF + AI + Map)", res_d)]:
            pos = res["positions"]
            if has_gt and gt_enu is not None:
                errs = np.linalg.norm(pos[:, :2] - gt_enu[:, :2], axis=1)
                err_start = float(errs[idx_start])
                err_mid = float(errs[idx_mid])
                err_end = float(errs[idx_end])
                outage_rmse = float(np.sqrt(np.mean(errs[outage_mask] ** 2)))
                max_err = float(np.max(errs[outage_mask]))
                final_err = float(errs[-1])

                drift_pct = float((err_end / dist_travelled) * 100.0)
                drift_per_100m = float((err_end / dist_travelled) * 100.0)
                target_pass = drift_pct < 10.0
                target_status = "PASS" if target_pass else "FAIL"
            else:
                err_start, err_mid, err_end, outage_rmse, max_err, final_err = None, None, None, None, None, None
                drift_pct, drift_per_100m = None, None
                target_status = "NOT_EVALUABLE (Ground truth not supplied)"

            outage_metrics[sc_name] = {
                "distance_travelled_during_outage_m": dist_travelled,
                "error_at_outage_start_m": err_start,
                "error_at_outage_midpoint_m": err_mid,
                "error_at_outage_end_m": err_end,
                "outage_rmse_m": outage_rmse,
                "max_outage_error_m": max_err,
                "final_error_m": final_err,
                "drift_percentage": drift_pct,
                "drift_per_100m_travelled_m": drift_per_100m,
                "preliminary_sih_target_check": {
                    "target_threshold_pct": 10.0,
                    "measured_drift_pct": drift_pct,
                    "status": target_status,
                    "disclaimer": "Preliminary target check on this recording. Official SIH compliance requires multi-session benchmark validation.",
                },
            }

    with open(os.path.join(results_dir, "outage_metrics.json"), "w") as f:
        json.dump(outage_metrics, f, indent=2)

    # Compile overall experiment metrics
    exp_metrics = {
        "dataset_name": dataset_cfg.dataset_name,
        "scenarios": {
            "Scenario A (Continuous GNSS)": {
                "max_velocity_mps": float(np.max(np.linalg.norm(res_a["velocities"], axis=1))),
                "mean_speed_mps": float(np.mean(np.linalg.norm(res_a["velocities"][:, :2], axis=1))),
            },
            "Scenario B (Outage INS/ESKF)": {
                "final_position_enu": res_b["positions"][-1].tolist(),
            },
            "Scenario C (Outage ESKF + AI)": {
                "ai_updates_accepted": res_c["pipeline"].ai_accepted_count,
                "ai_updates_rejected": res_c["pipeline"].ai_rejected_count,
                "ai_acceptance_rate_pct": res_c["pipeline"].acceptance_rate,
            },
            "Scenario D (Outage ESKF + AI + Map)": {
                "map_updates_accepted": res_d["pipeline"].map_accepted_count,
                "map_updates_rejected": res_d["pipeline"].map_rejected_count,
                "map_acceptance_rate_pct": res_d["pipeline"].map_acceptance_rate,
            },
        },
    }
    with open(os.path.join(results_dir, "experiment_metrics.json"), "w") as f:
        json.dump(exp_metrics, f, indent=2)

    # 8. Generate Visualizations
    print("Generating Visualizations...")
    # Plot 1: 01_raw_imu.png
    plt.figure(figsize=(10, 6))
    ax_val = [obs.accelerometer_x for obs in imu_list]
    ay_val = [obs.accelerometer_y for obs in imu_list]
    az_val = [obs.accelerometer_z for obs in imu_list]
    t_imu = [obs.timestamp for obs in imu_list]
    plt.plot(t_imu, ax_val, label="Accel X", alpha=0.8)
    plt.plot(t_imu, ay_val, label="Accel Y", alpha=0.8)
    plt.plot(t_imu, az_val, label="Accel Z", alpha=0.8)
    plt.title(f"Raw Accelerometer Stream ({dataset_cfg.dataset_name})")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (m/s²)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "01_raw_imu.png"), dpi=200)
    plt.close()

    # Plot 2: 02_sampling_intervals.png
    plt.figure(figsize=(10, 4))
    dts = np.diff(t_imu)
    plt.plot(t_imu[:-1], dts * 1000.0, "b-", linewidth=1.0)
    plt.title("IMU Inter-Sample Time Intervals (dt)")
    plt.xlabel("Time (s)")
    plt.ylabel("Interval dt (ms)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "02_sampling_intervals.png"), dpi=200)
    plt.close()

    # Plot 3: 03_gnss_trajectory.png (if GNSS present)
    if gnss_list:
        plt.figure(figsize=(8, 7))
        g_lons = [g.longitude for g in gnss_list]
        g_lats = [g.latitude for g in gnss_list]
        plt.plot(g_lons, g_lats, "r.-", label="Raw GNSS Fixes")
        plt.title("GNSS Satellite Position Track")
        plt.xlabel("Longitude (°)")
        plt.ylabel("Latitude (°)")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "03_gnss_trajectory.png"), dpi=200)
        plt.close()

    # Plot 4: 04_ground_truth_trajectory.png
    if has_gt and gt_enu is not None:
        plt.figure(figsize=(8, 7))
        plt.plot(gt_enu[:, 0], gt_enu[:, 1], "g-", linewidth=2.0, label="Ground Truth ENU")
        plt.title("Reference Ground Truth ENU Trajectory")
        plt.xlabel("East (m)")
        plt.ylabel("North (m)")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "04_ground_truth_trajectory.png"), dpi=200)
        plt.close()

    # Plot 5: 05_ins_trajectory.png
    plt.figure(figsize=(8, 7))
    plt.plot(res_b["positions"][:, 0], res_b["positions"][:, 1], "r-", label="Scenario B (ESKF Outage)")
    if has_gt and gt_enu is not None:
        plt.plot(gt_enu[:, 0], gt_enu[:, 1], "k--", label="Ground Truth", alpha=0.6)
    plt.title("Scenario B (ESKF Outage - AI Off) Trajectory")
    plt.xlabel("East (m)")
    plt.ylabel("North (m)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "05_ins_trajectory.png"), dpi=200)
    plt.close()

    # Plot 6: 06_eskf_trajectory.png
    plt.figure(figsize=(8, 7))
    plt.plot(res_a["positions"][:, 0], res_a["positions"][:, 1], "g-", label="Continuous GNSS ESKF")
    if has_gt and gt_enu is not None:
        plt.plot(gt_enu[:, 0], gt_enu[:, 1], "k--", label="Ground Truth", alpha=0.6)
    plt.title("Scenario A (Continuous GNSS Baseline) Trajectory")
    plt.xlabel("East (m)")
    plt.ylabel("North (m)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "06_eskf_trajectory.png"), dpi=200)
    plt.close()

    # Plot 7: 07_ai_fusion_trajectory.png
    plt.figure(figsize=(8, 7))
    plt.plot(res_c["positions"][:, 0], res_c["positions"][:, 1], "m-", label="Scenario C (ESKF + AI)")
    plt.plot(res_d["positions"][:, 0], res_d["positions"][:, 1], "b-", label="Scenario D (ESKF + AI + Map)")
    if has_gt and gt_enu is not None:
        plt.plot(gt_enu[:, 0], gt_enu[:, 1], "k--", label="Ground Truth", alpha=0.6)
    plt.title("AI & Map-Assisted Outage Trajectories")
    plt.xlabel("East (m)")
    plt.ylabel("North (m)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "07_ai_fusion_trajectory.png"), dpi=200)
    plt.close()

    # Plot 8: 08_horizontal_error.png
    if has_gt and gt_enu is not None and dataset_cfg.outage.enabled:
        plt.figure(figsize=(10, 5))
        err_b = np.linalg.norm(res_b["positions"][:, :2] - gt_enu[:, :2], axis=1)
        err_c = np.linalg.norm(res_c["positions"][:, :2] - gt_enu[:, :2], axis=1)
        err_d = np.linalg.norm(res_d["positions"][:, :2] - gt_enu[:, :2], axis=1)
        plt.plot(t_arr, err_b, "r:", label="ESKF Outage (AI Off)")
        plt.plot(t_arr, err_c, "m--", label="ESKF + AI")
        plt.plot(t_arr, err_d, "b-", label="ESKF + AI + Map")
        plt.axvspan(dataset_cfg.outage.start_time_sec, dataset_cfg.outage.start_time_sec + dataset_cfg.outage.duration_sec, color="orange", alpha=0.2, label="GNSS Outage Window")
        plt.title("Horizontal Position Error Over Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Horizontal Error (m)")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "08_horizontal_error.png"), dpi=200)
        plt.close()

    # Plot 9: 09_outage_error.png
    if has_gt and gt_enu is not None and dataset_cfg.outage.enabled:
        plt.figure(figsize=(8, 4.5))
        outage_mask = (t_arr >= dataset_cfg.outage.start_time_sec) & (t_arr <= dataset_cfg.outage.start_time_sec + dataset_cfg.outage.duration_sec)
        plt.plot(t_arr[outage_mask], err_b[outage_mask], "r:", label="ESKF Outage (AI Off)")
        plt.plot(t_arr[outage_mask], err_c[outage_mask], "m--", label="ESKF + AI")
        plt.plot(t_arr[outage_mask], err_d[outage_mask], "b-", label="ESKF + AI + Map")
        plt.title("Zoomed Position Drift During GNSS Outage Interval")
        plt.xlabel("Time (s)")
        plt.ylabel("Horizontal Error (m)")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "09_outage_error.png"), dpi=200)
        plt.close()

    # Plot 10: 10_velocity.png
    plt.figure(figsize=(10, 4.5))
    v_mag = np.linalg.norm(res_d["velocities"], axis=1)
    plt.plot(t_arr, v_mag, "b-", label="Estimated 3D Speed")
    plt.title("Estimated Vehicle Speed Profile")
    plt.xlabel("Time (s)")
    plt.ylabel("Speed (m/s)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "10_velocity.png"), dpi=200)
    plt.close()

    # Plot 11: 11_heading.png
    plt.figure(figsize=(10, 4.5))
    plt.plot(t_arr, res_d["headings"], "g-", label="Estimated Heading Azimuth")
    plt.title("Estimated Vehicle Heading Azimuth")
    plt.xlabel("Time (s)")
    plt.ylabel("Heading (°)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "11_heading.png"), dpi=200)
    plt.close()

    # Plot 12: 12_sensor_biases.png
    plt.figure(figsize=(10, 5))
    gx_val = [obs.gyroscope_x for obs in imu_list]
    gy_val = [obs.gyroscope_y for obs in imu_list]
    gz_val = [obs.gyroscope_z for obs in imu_list]
    plt.plot(t_imu, gx_val, label="Gyro X", alpha=0.7)
    plt.plot(t_imu, gy_val, label="Gyro Y", alpha=0.7)
    plt.plot(t_imu, gz_val, label="Gyro Z", alpha=0.7)
    plt.title("Gyroscope Angular Velocity Stream")
    plt.xlabel("Time (s)")
    plt.ylabel("Rate (rad/s)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "12_sensor_biases.png"), dpi=200)
    plt.close()

    # Save Run Metadata
    run_meta = {
        "dataset_name": dataset_cfg.dataset_name,
        "config_file": args.config,
        "adapter_class": adapter.__class__.__name__,
        "origin": {"lat": orig_lat, "lon": orig_lon, "alt": orig_alt},
        "imu_sample_count": len(imu_list),
        "gnss_sample_count": len(gnss_list),
        "ground_truth_available": has_gt,
        "ai_model_loaded": predictor is not None,
        "map_network_loaded": road_network is not None,
        "outage_simulation": dataset_cfg.outage.enabled,
    }
    with open(os.path.join(results_dir, "run_metadata.json"), "w") as f:
        json.dump(run_meta, f, indent=2)

    print(f"\nAll results and plots saved under: {os.path.abspath(results_dir)}")
    print("=" * 80)
    print("STEP 10 REAL DATASET EVALUATION COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
