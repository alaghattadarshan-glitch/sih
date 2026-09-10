#!/usr/bin/env python3
"""GNSS Outage Dead Reckoning Baseline Experiment Script.

Processes IMU sensor measurements exclusively to propagate vehicle navigation state
(Strapdown INS), evaluates drift against ground truth trajectory, saves diagnostic plots
to results/dead_reckoning/, and outputs quantitative performance metrics.
"""

import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data.loaders import load_imu_csv, load_ground_truth_csv
from src.calibration import estimate_static_bias
from src.navigation.dead_reckoning import InertialDeadReckoning
from src.navigation.quaternion import stationary_attitude_initialization, heading_deg_to_yaw_rad, euler_to_quaternion
from src.evaluation import evaluate_trajectory
from src.coordinate_transforms import LocalFrame


def plot_results(
    eval_res: dict,
    dr_states: list,
    gt_list: list,
    local_frame: LocalFrame,
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract positions
    dr_enu = np.array([s.position_enu for s in dr_states])
    gt_enu = np.array(
        [local_frame.to_enu(gt.latitude, gt.longitude, gt.altitude) for gt in gt_list]
    )

    t_eval = eval_res["timestamps"]

    # 1. Trajectory comparison plot (East vs North)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(gt_enu[:, 0], gt_enu[:, 1], "g-", label="Ground Truth Reference", linewidth=2)
    ax.plot(dr_enu[:, 0], dr_enu[:, 1], "r--", label="Strapdown INS Dead Reckoning", linewidth=1.5)
    ax.scatter([gt_enu[0, 0]], [gt_enu[0, 1]], color="black", s=50, zorder=5, label="Start Position")
    ax.set_xlabel("East Position (m)")
    ax.set_ylabel("North Position (m)")
    ax.set_title("Vehicle Trajectory: Ground Truth vs Dead Reckoning")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_dir / "trajectory_comparison.png", dpi=150)
    plt.close(fig)

    # 2. East Error vs Time
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t_eval, eval_res["east_errors"], "b-")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("East Position Error (m)")
    ax.set_title("East Position Error vs Time")
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig(output_dir / "east_error_vs_time.png", dpi=150)
    plt.close(fig)

    # 3. North Error vs Time
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t_eval, eval_res["north_errors"], "b-")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("North Position Error (m)")
    ax.set_title("North Position Error vs Time")
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig(output_dir / "north_error_vs_time.png", dpi=150)
    plt.close(fig)

    # 4. Horizontal Error vs Time
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t_eval, eval_res["horizontal_errors"], "r-")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Horizontal Position Error (m)")
    ax.set_title("Horizontal Position Drift Error vs Time")
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig(output_dir / "horizontal_error_vs_time.png", dpi=150)
    plt.close(fig)

    # 5. Velocity Comparison
    dr_speed = np.array([s.speed for s in dr_states])
    gt_speed = np.array([gt.speed if gt.speed is not None else 0.0 for gt in gt_list])
    dr_times = np.array([s.timestamp for s in dr_states])
    gt_times = np.array([gt.timestamp for gt in gt_list])

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(gt_times, gt_speed, "g-", label="Ground Truth Speed", linewidth=2)
    ax.plot(dr_times, dr_speed, "r--", label="INS Estimated Speed", linewidth=1.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Ground Speed (m/s)")
    ax.set_title("Speed Comparison vs Time")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_dir / "velocity_comparison.png", dpi=150)
    plt.close(fig)

    # 6. Heading Comparison
    dr_hdg = np.array([s.heading_deg() for s in dr_states])
    gt_hdg = np.array([gt.heading if gt.heading is not None else 0.0 for gt in gt_list])

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(gt_times, gt_hdg, "g-", label="Ground Truth Heading", linewidth=2)
    ax.plot(dr_times, dr_hdg, "r--", label="INS Estimated Heading", linewidth=1.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Heading (deg)")
    ax.set_title("Heading Comparison vs Time")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_dir / "heading_comparison.png", dpi=150)
    plt.close(fig)

    print(f"Diagnostic plots saved to: {output_dir}")


def main():
    sample_dir = project_root / "data" / "sample"
    imu_path = sample_dir / "imu_sample.csv"
    gt_path = sample_dir / "ground_truth_sample.csv"

    if not imu_path.exists() or not gt_path.exists():
        print("Sample data missing. Please run scripts/generate_sample_data.py first.")
        sys.exit(1)

    print("Loading sample IMU and Ground Truth observations...")
    imu_list = load_imu_csv(imu_path)
    gt_list = load_ground_truth_csv(gt_path)

    # Use first 1.0s of stationary IMU data for static bias estimation
    stationary_segment = [obs for obs in imu_list if obs.timestamp <= 1.0]
    accel_bias, gyro_bias = estimate_static_bias(stationary_segment)

    # Initialize navigation engine at t=0 using initial ground truth state
    gt0 = gt_list[0]
    initial_llh = (gt0.latitude, gt0.longitude, gt0.altitude)
    local_frame = LocalFrame(gt0.latitude, gt0.longitude, gt0.altitude)

    v0_east = gt0.velocity_east if gt0.velocity_east is not None else 0.0
    v0_north = gt0.velocity_north if gt0.velocity_north is not None else 0.0
    v0_up = gt0.velocity_up if gt0.velocity_up is not None else 0.0
    v0_enu = np.array([v0_east, v0_north, v0_up], dtype=np.float64)

    initial_heading = gt0.heading if gt0.heading is not None else 90.0
    initial_q = euler_to_quaternion(0.0, 0.0, heading_deg_to_yaw_rad(initial_heading))

    dr_system = InertialDeadReckoning()
    dr_system.initialize(
        initial_time=gt0.timestamp,
        initial_llh=initial_llh,
        initial_velocity_enu=v0_enu,
        initial_quaternion=initial_q,
        accel_bias=accel_bias,
        gyro_bias=gyro_bias,
    )

    print("Executing Inertial Dead Reckoning propagation (IMU data only)...")
    dr_states = dr_system.process_imu_stream(imu_list)

    print("Evaluating dead-reckoned trajectory against Ground Truth...")
    eval_res = evaluate_trajectory(dr_states, gt_list, local_frame)

    output_dir = project_root / "results" / "dead_reckoning"
    plot_results(eval_res, dr_states, gt_list, local_frame, output_dir)

    # Print baseline performance report
    print()
    print("──────────────────────────────────────────────────────────")
    print("                STRAPDOWN INS BASELINE")
    print("──────────────────────────────────────────────────────────")
    print(f"Duration:                 {dr_system.elapsed_time:.2f} s")
    print(f"Initial position:        East = 0.00 m, North = 0.00 m, Up = 0.00 m")
    final_pos = dr_states[-1].position_enu
    print(f"Final estimated position: East = {final_pos[0]:.2f} m, North = {final_pos[1]:.2f} m, Up = {final_pos[2]:.2f} m")
    print(f"Final position error:    {eval_res['final_position_error']:.2f} m")
    print(f"Maximum horizontal error: {eval_res['max_horizontal_error']:.2f} m")
    print(f"Horizontal RMSE:          {eval_res['horizontal_rmse']:.2f} m")
    print(f"Final velocity error:    {eval_res['final_velocity_error']:.3f} m/s")
    print(f"Final heading error:     {eval_res['final_heading_error']:.2f} degrees")
    print("──────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
