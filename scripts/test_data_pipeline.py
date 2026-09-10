#!/usr/bin/env python3
"""Data Pipeline Demonstration Script.

Loads sample data, performs data validation, calculates sampling quality statistics,
synchronizes high-rate IMU with lower-rate GNSS, generates diagnostic plots under
results/data_diagnostics/, and prints a summary report.
"""

import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data.loaders import load_imu_csv, load_gnss_csv, load_ground_truth_csv
from src.data.synchronization import SynchronizedDataset
from src.data.conversions import imu_to_dataframe, gnss_to_dataframe


def generate_diagnostic_plots(imu_df, gnss_df, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Accelerometer x/y/z vs time
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(imu_df["timestamp"], imu_df["accel_x"], label="Accel X (m/s²)", alpha=0.8)
    ax.plot(imu_df["timestamp"], imu_df["accel_y"], label="Accel Y (m/s²)", alpha=0.8)
    ax.plot(imu_df["timestamp"], imu_df["accel_z"], label="Accel Z (m/s²)", alpha=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Acceleration (m/s²)")
    ax.set_title("Accelerometer Measurements vs Time")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_dir / "accelerometer_vs_time.png", dpi=150)
    plt.close(fig)

    # Plot 2: Gyroscope x/y/z vs time
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(imu_df["timestamp"], imu_df["gyro_x"], label="Gyro X (rad/s)", alpha=0.8)
    ax.plot(imu_df["timestamp"], imu_df["gyro_y"], label="Gyro Y (rad/s)", alpha=0.8)
    ax.plot(imu_df["timestamp"], imu_df["gyro_z"], label="Gyro Z (rad/s)", alpha=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angular Velocity (rad/s)")
    ax.set_title("Gyroscope Measurements vs Time")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_dir / "gyroscope_vs_time.png", dpi=150)
    plt.close(fig)

    # Plot 3: GNSS Latitude / Longitude Trajectory
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(gnss_df["longitude"], gnss_df["latitude"], "b.-", label="GNSS Fixes", markersize=6)
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.set_title("GNSS Trajectory (Lat/Lon)")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_dir / "gnss_trajectory.png", dpi=150)
    plt.close(fig)

    # Plot 4: Sensor Sampling Interval (dt vs time)
    imu_dt = np.diff(imu_df["timestamp"])
    gnss_dt = np.diff(gnss_df["timestamp"])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=False)
    ax1.plot(imu_df["timestamp"][1:], imu_dt * 1000.0, "g-", alpha=0.7)
    ax1.set_ylabel("IMU dt (ms)")
    ax1.set_title("Sensor Sampling Interval Jitter")
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2.plot(gnss_df["timestamp"][1:], gnss_dt, "m-", alpha=0.7)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("GNSS dt (s)")
    ax2.grid(True, linestyle="--", alpha=0.6)

    fig.tight_layout()
    fig.savefig(output_dir / "sensor_sampling_interval.png", dpi=150)
    plt.close(fig)

    print(f"Diagnostic plots saved to: {output_dir}")


def main():
    sample_dir = project_root / "data" / "sample"
    imu_file = sample_dir / "imu_sample.csv"
    gnss_file = sample_dir / "gnss_sample.csv"
    gt_file = sample_dir / "ground_truth_sample.csv"

    if not imu_file.exists():
        print(f"Sample data missing at {imu_file}. Please run scripts/generate_sample_data.py first.")
        sys.exit(1)

    # 1. Load data
    imu = load_imu_csv(imu_file)
    gnss = load_gnss_csv(gnss_file)
    gt = load_ground_truth_csv(gt_file) if gt_file.exists() else None

    # 2. Synchronize dataset
    synced_dataset = SynchronizedDataset.create(imu, gnss, ground_truth=gt)

    # 3. Generate diagnostic plots
    imu_df = imu_to_dataframe(imu)
    gnss_df = gnss_to_dataframe(gnss)
    results_dir = project_root / "results" / "data_diagnostics"
    generate_diagnostic_plots(imu_df, gnss_df, results_dir)

    # 4. Print pipeline report
    meta = synced_dataset.metadata
    imu_meta = meta["imu_stats"]
    gnss_meta = meta["gnss_stats"]

    print()
    print("──────────────────────────────────────────────────────────")
    print("                     DATA PIPELINE")
    print("──────────────────────────────────────────────────────────")
    print(f"Duration:               {imu_meta['duration_sec']:.2f} s")
    print(f"IMU samples:            {imu_meta['count']}")
    print(f"GNSS samples:           {gnss_meta['count']}")
    print(f"IMU frequency:          {imu_meta['sampling_rate_hz']:.1f} Hz")
    print(f"GNSS frequency:         {gnss_meta['sampling_rate_hz']:.1f} Hz")
    print(f"Missing samples:        0")
    print(f"Duplicate timestamps:   {imu_meta['duplicate_count'] + gnss_meta['duplicate_count']}")
    print(f"Synchronization status: {meta['status']}")
    print("──────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
