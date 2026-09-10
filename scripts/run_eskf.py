#!/usr/bin/env python3
"""15-State ESKF GNSS/INS Sensor Fusion Baseline & Outage Experiment Script.

Evaluates and compares three navigation modes on the synthetic trajectory:
1. Open-loop Strapdown INS (IMU only, no GNSS)
2. Full ESKF + GNSS (Continuous GNSS updates)
3. ESKF with GNSS Outage (GNSS 0-30s, Outage 30-60s, Recovery 60-80s)

Saves diagnostic plots to results/eskf/ and prints quantitative baseline metrics.
"""

import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data.loaders import load_imu_csv, load_gnss_csv, load_ground_truth_csv
from src.calibration import estimate_static_bias
from src.navigation.dead_reckoning import InertialDeadReckoning
from src.navigation.ekf import ErrorStateKalmanFilter, ESKFStateConfig
from src.navigation.quaternion import heading_deg_to_yaw_rad, euler_to_quaternion
from src.evaluation import evaluate_trajectory
from src.coordinate_transforms import LocalFrame


def plot_eskf_results(
    eval_ins: dict,
    eval_eskf_outage: dict,
    eval_eskf_full: dict,
    ins_states: list,
    eskf_outage_states: list,
    gt_list: list,
    local_frame: LocalFrame,
    outage_start: float,
    outage_end: float,
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract positions
    gt_enu = np.array([local_frame.to_enu(gt.latitude, gt.longitude, gt.altitude) for gt in gt_list])
    ins_enu = np.array([s.position_enu for s in ins_states])
    eskf_outage_enu = np.array([s.position_enu for s in eskf_outage_states])

    t_ins = np.array([s.timestamp for s in ins_states])
    t_eskf = np.array([s.timestamp for s in eskf_outage_states])

    # 1. Trajectory comparison plot
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.plot(gt_enu[:, 0], gt_enu[:, 1], "g-", label="Ground Truth Reference", linewidth=2)
    ax.plot(ins_enu[:, 0], ins_enu[:, 1], "r--", label="Open-loop INS (Uncorrected)", linewidth=1.5)
    ax.plot(eskf_outage_enu[:, 0], eskf_outage_enu[:, 1], "b-", label="ESKF (Outage 30-60s)", linewidth=2)
    ax.scatter([gt_enu[0, 0]], [gt_enu[0, 1]], color="black", s=60, zorder=5, label="Start Origin")
    ax.set_xlabel("East Position (m)")
    ax.set_ylabel("North Position (m)")
    ax.set_title("Trajectory Comparison: Open-loop INS vs ESKF Fusion")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "trajectory_comparison.png", dpi=150)
    plt.close(fig)

    # 2. Position Error Comparison vs Time
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(eval_ins["timestamps"], eval_ins["horizontal_errors"], "r--", label="Open-loop INS Error", alpha=0.7)
    ax.plot(eval_eskf_outage["timestamps"], eval_eskf_outage["horizontal_errors"], "b-", label="ESKF (Outage 30-60s)", linewidth=2)
    ax.axvspan(outage_start, outage_end, color="orange", alpha=0.2, label="GNSS Outage Window (30-60s)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Horizontal Position Error (m)")
    ax.set_title("Horizontal Drift Error Comparison")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_dir / "position_error_comparison.png", dpi=150)
    plt.close(fig)

    # 3. Horizontal Error (ESKF Zoomed)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(eval_eskf_outage["timestamps"], eval_eskf_outage["horizontal_errors"], "b-", label="ESKF Position Error", linewidth=2)
    ax.axvspan(outage_start, outage_end, color="orange", alpha=0.25, label="GNSS Outage Window")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Horizontal Position Error (m)")
    ax.set_title("ESKF Horizontal Position Error & GNSS Outage Recovery")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_dir / "horizontal_error_vs_time.png", dpi=150)
    plt.close(fig)

    # 4. East Error vs Time
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(eval_ins["timestamps"], eval_ins["east_errors"], "r--", label="INS East Error", alpha=0.7)
    ax.plot(eval_eskf_outage["timestamps"], eval_eskf_outage["east_errors"], "b-", label="ESKF East Error")
    ax.axvspan(outage_start, outage_end, color="orange", alpha=0.2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("East Position Error (m)")
    ax.set_title("East Component Error vs Time")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_dir / "east_error_vs_time.png", dpi=150)
    plt.close(fig)

    # 5. North Error vs Time
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(eval_ins["timestamps"], eval_ins["north_errors"], "r--", label="INS North Error", alpha=0.7)
    ax.plot(eval_eskf_outage["timestamps"], eval_eskf_outage["north_errors"], "b-", label="ESKF North Error")
    ax.axvspan(outage_start, outage_end, color="orange", alpha=0.2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("North Position Error (m)")
    ax.set_title("North Component Error vs Time")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_dir / "north_error_vs_time.png", dpi=150)
    plt.close(fig)

    # 6. Velocity Comparison
    gt_speed = np.array([gt.speed if gt.speed is not None else 0.0 for gt in gt_list])
    gt_times = np.array([gt.timestamp for gt in gt_list])
    eskf_speed = np.array([s.speed for s in eskf_outage_states])

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(gt_times, gt_speed, "g-", label="Ground Truth Speed", linewidth=2)
    ax.plot(t_eskf, eskf_speed, "b-", label="ESKF Estimated Speed", linewidth=1.5)
    ax.axvspan(outage_start, outage_end, color="orange", alpha=0.2, label="GNSS Outage")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed (m/s)")
    ax.set_title("Vehicle Speed Estimation Comparison")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_dir / "velocity_comparison.png", dpi=150)
    plt.close(fig)

    # 7. Heading Comparison
    gt_hdg = np.array([gt.heading if gt.heading is not None else 0.0 for gt in gt_list])
    eskf_hdg = np.array([s.heading_deg() for s in eskf_outage_states])

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(gt_times, gt_hdg, "g-", label="Ground Truth Heading", linewidth=2)
    ax.plot(t_eskf, eskf_hdg, "b-", label="ESKF Estimated Heading", linewidth=1.5)
    ax.axvspan(outage_start, outage_end, color="orange", alpha=0.2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Heading (deg)")
    ax.set_title("Heading Estimation Comparison")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_dir / "heading_comparison.png", dpi=150)
    plt.close(fig)

    print(f"Diagnostic plots saved to: {output_dir}")


def get_error_at_time(eval_dict: dict, target_t: float) -> float:
    """Helper to extract horizontal error at exact timestamp."""
    times = eval_dict["timestamps"]
    idx = int(np.argmin(np.abs(times - target_t)))
    return float(eval_dict["horizontal_errors"][idx])


def main():
    sample_dir = project_root / "data" / "sample"
    imu_path = sample_dir / "imu_sample.csv"
    gnss_path = sample_dir / "gnss_sample.csv"
    gt_path = sample_dir / "ground_truth_sample.csv"

    if not imu_path.exists() or not gnss_path.exists() or not gt_path.exists():
        print("Sample data missing. Please run scripts/generate_sample_data.py first.")
        sys.exit(1)

    print("Loading sample IMU, GNSS, and Ground Truth observations...")
    imu_list = load_imu_csv(imu_path)
    gnss_list = load_gnss_csv(gnss_path)
    gt_list = load_ground_truth_csv(gt_path)

    # Static calibration on first 1.0s
    stationary_segment = [obs for obs in imu_list if obs.timestamp <= 1.0]
    accel_bias, gyro_bias = estimate_static_bias(stationary_segment)

    gt0 = gt_list[0]
    initial_llh = (gt0.latitude, gt0.longitude, gt0.altitude)
    local_frame = LocalFrame(gt0.latitude, gt0.longitude, gt0.altitude)

    v0_enu = np.array(
        [
            gt0.velocity_east if gt0.velocity_east is not None else 0.0,
            gt0.velocity_north if gt0.velocity_north is not None else 0.0,
            gt0.velocity_up if gt0.velocity_up is not None else 0.0,
        ],
        dtype=np.float64,
    )
    initial_heading = gt0.heading if gt0.heading is not None else 90.0
    initial_q = euler_to_quaternion(0.0, 0.0, heading_deg_to_yaw_rad(initial_heading))

    # 1. Run System A: Open-loop Strapdown INS
    print("1. Running System A: Open-loop Strapdown INS...")
    ins_dr = InertialDeadReckoning()
    ins_dr.initialize(
        initial_time=gt0.timestamp,
        initial_llh=initial_llh,
        initial_velocity_enu=v0_enu,
        initial_quaternion=initial_q,
        accel_bias=accel_bias,
        gyro_bias=gyro_bias,
    )
    ins_states = ins_dr.process_imu_stream(imu_list)

    # 2. Run System B: Full ESKF + GNSS (no outage)
    print("2. Running System B: Full ESKF + GNSS (Continuous)...")
    eskf_full = ErrorStateKalmanFilter(config=ESKFStateConfig())
    eskf_full.initialize(
        initial_time=gt0.timestamp,
        initial_llh=initial_llh,
        initial_velocity_enu=v0_enu,
        initial_quaternion=initial_q,
        accel_bias=accel_bias,
        gyro_bias=gyro_bias,
    )
    eskf_full_states = []
    gnss_by_time = {round(g.timestamp, 2): g for g in gnss_list}

    for imu_obs in imu_list:
        state = eskf_full.predict(imu_obs)
        t_key = round(imu_obs.timestamp, 2)
        if t_key in gnss_by_time:
            state = eskf_full.update_gnss(gnss_by_time[t_key])
        eskf_full_states.append(state)

    # 3. Run System C: ESKF during GNSS Outage (0-30s GNSS, 30-60s Outage, 60-80s Recovery)
    outage_start = 30.0
    outage_end = 60.0
    print(f"3. Running System C: ESKF with GNSS Outage ({outage_start}s to {outage_end}s)...")

    eskf_outage = ErrorStateKalmanFilter(config=ESKFStateConfig())
    eskf_outage.initialize(
        initial_time=gt0.timestamp,
        initial_llh=initial_llh,
        initial_velocity_enu=v0_enu,
        initial_quaternion=initial_q,
        accel_bias=accel_bias,
        gyro_bias=gyro_bias,
    )
    eskf_outage_states = []

    for imu_obs in imu_list:
        state = eskf_outage.predict(imu_obs)
        t_curr = imu_obs.timestamp

        # Apply GNSS updates ONLY when OUTSIDE the GNSS outage window
        if not (outage_start <= t_curr < outage_end):
            t_key = round(t_curr, 2)
            if t_key in gnss_by_time:
                state = eskf_outage.update_gnss(gnss_by_time[t_key])

        eskf_outage_states.append(state)

    # Evaluate Trajectories
    eval_ins = evaluate_trajectory(ins_states, gt_list, local_frame)
    eval_eskf_full = evaluate_trajectory(eskf_full_states, gt_list, local_frame)
    eval_eskf_outage = evaluate_trajectory(eskf_outage_states, gt_list, local_frame)

    output_dir = project_root / "results" / "eskf"
    plot_eskf_results(
        eval_ins,
        eval_eskf_outage,
        eval_eskf_full,
        ins_states,
        eskf_outage_states,
        gt_list,
        local_frame,
        outage_start,
        outage_end,
        output_dir,
    )

    # Print baseline comparison report
    err_outage_start = get_error_at_time(eval_eskf_outage, outage_start)
    err_outage_mid = get_error_at_time(eval_eskf_outage, (outage_start + outage_end) / 2.0)
    err_outage_end = get_error_at_time(eval_eskf_outage, outage_end - 0.1)
    err_recovery = get_error_at_time(eval_eskf_outage, outage_end + 2.0)

    print()
    print("──────────────────────────────────────────────────────────")
    print("             GNSS/INS SENSOR FUSION COMPARISON")
    print("──────────────────────────────────────────────────────────")
    print("OPEN-LOOP STRAPDOWN INS (IMU ONLY)")
    print(f"  Horizontal RMSE:       {eval_ins['horizontal_rmse']:.2f} m")
    print(f"  Final position error:  {eval_ins['final_position_error']:.2f} m")
    print(f"  Max horizontal error:  {eval_ins['max_horizontal_error']:.2f} m")
    print()
    print("FULL ESKF (CONTINUOUS GNSS + IMU)")
    print(f"  Horizontal RMSE:       {eval_eskf_full['horizontal_rmse']:.2f} m")
    print(f"  Final position error:  {eval_eskf_full['final_position_error']:.2f} m")
    print(f"  Max horizontal error:  {eval_eskf_full['max_horizontal_error']:.2f} m")
    print()
    print("ESKF EXPERIMENT WITH GNSS OUTAGE")
    print(f"  Outage Start:          {outage_start:.1f} s")
    print(f"  Outage End:            {outage_end:.1f} s")
    print(f"  Outage Duration:       {outage_end - outage_start:.1f} s")
    print(f"  Error at outage start: {err_outage_start:.2f} m")
    print(f"  Error at outage mid:   {err_outage_mid:.2f} m")
    print(f"  Error at outage end:   {err_outage_end:.2f} m")
    print(f"  Error after recovery:  {err_recovery:.2f} m")
    print(f"  Final position error:  {eval_eskf_outage['final_position_error']:.2f} m")
    print(f"  Horizontal RMSE:       {eval_eskf_outage['horizontal_rmse']:.2f} m")
    print("──────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
