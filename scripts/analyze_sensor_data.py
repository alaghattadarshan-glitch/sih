#!/usr/bin/env python3
"""Sensor Data Quality Analysis & Statistical Profiling Utility.

Inspects raw/sample sensor files or loaded observations and prints a detailed
health and statistical report.
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data.synchronization import calculate_timestamp_statistics, find_timestamp_gaps


def analyze_dataset(
    imu_path: Path, gnss_path: Path, gt_path: Path
):
    print("==========================================================")
    print("        SIH26168 SENSOR DATA QUALITY REPORT")
    print("==========================================================")

    # 1. Load data
    imu_df = pd.read_csv(imu_path) if imu_path.exists() else None
    gnss_df = pd.read_csv(gnss_path) if gnss_path.exists() else None
    gt_df = pd.read_csv(gt_path) if gt_path.exists() else None

    # Summary table
    print("Dataset Files Loaded:")
    print(f"  IMU file:          {imu_path.name if imu_df is not None else 'N/A'}")
    print(f"  GNSS file:         {gnss_path.name if gnss_df is not None else 'N/A'}")
    print(f"  Ground Truth file: {gt_path.name if gt_df is not None else 'N/A'}")
    print()

    # 2. Sample Counts & Duration
    imu_times = imu_df["timestamp"].tolist() if imu_df is not None else []
    gnss_times = gnss_df["timestamp"].tolist() if gnss_df is not None else []
    gt_times = gt_df["timestamp"].tolist() if gt_df is not None else []

    imu_stats = calculate_timestamp_statistics(imu_times)
    gnss_stats = calculate_timestamp_statistics(gnss_times)
    gt_stats = calculate_timestamp_statistics(gt_times)

    print("Sample Counts & Duration:")
    print(f"  IMU samples:           {imu_stats['count']}")
    print(f"  GNSS samples:          {gnss_stats['count']}")
    print(f"  Ground Truth samples:  {gt_stats['count']}")
    print(f"  IMU Time Duration:     {imu_stats['duration_sec']:.2f} s")
    print(f"  GNSS Time Duration:    {gnss_stats['duration_sec']:.2f} s")
    print()

    # Required channel NaN count
    req_imu_cols = ["timestamp", "accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]
    imu_nan_count = imu_df[req_imu_cols].isna().sum().sum() if imu_df is not None else 0

    req_gnss_cols = ["timestamp", "latitude", "longitude", "altitude"]
    gnss_nan_count = gnss_df[req_gnss_cols].isna().sum().sum() if gnss_df is not None else 0

    # 3. Sampling Frequencies & Quality
    print("Sampling Frequencies & Quality:")
    print(f"  IMU Frequency:         {imu_stats['sampling_rate_hz']:.1f} Hz (mean dt = {imu_stats['mean_dt']*1000:.2f} ms)")
    print(f"  GNSS Frequency:        {gnss_stats['sampling_rate_hz']:.1f} Hz (mean dt = {gnss_stats['mean_dt']:.3f} s)")
    print(f"  IMU Duplicate Count:   {imu_stats['duplicate_count']}")
    print(f"  GNSS Duplicate Count:  {gnss_stats['duplicate_count']}")
    print(f"  IMU NaN Count:         {imu_nan_count}")
    print(f"  GNSS NaN Count:        {gnss_nan_count}")
    print()

    # 4. Gaps analysis
    imu_gaps = find_timestamp_gaps(imu_times, max_expected_dt=imu_stats['mean_dt'] * 3.0) if imu_times else []
    gnss_gaps = find_timestamp_gaps(gnss_times, max_expected_dt=gnss_stats['mean_dt'] * 3.0) if gnss_times else []
    print("Timestamp Gaps:")
    print(f"  IMU gaps (>3x dt):     {len(imu_gaps)}")
    print(f"  GNSS gaps (>3x dt):    {len(gnss_gaps)}")
    print()

    # 5. Basic Sensor Statistics
    if imu_df is not None:
        print("IMU Sensor Statistics:")
        accel_norm = np.sqrt(imu_df["accel_x"]**2 + imu_df["accel_y"]**2 + imu_df["accel_z"]**2)
        gyro_norm = np.sqrt(imu_df["gyro_x"]**2 + imu_df["gyro_y"]**2 + imu_df["gyro_z"]**2)
        print(f"  Accel Norm: Mean = {accel_norm.mean():.3f} m/s^2, Std = {accel_norm.std():.3f} m/s^2, Min = {accel_norm.min():.3f}, Max = {accel_norm.max():.3f}")
        print(f"  Gyro Norm:  Mean = {gyro_norm.mean():.4f} rad/s, Std = {gyro_norm.std():.4f} rad/s, Max = {gyro_norm.max():.4f}")
        print()

    if gnss_df is not None:
        print("GNSS Statistics:")
        print(f"  Latitude range:  [{gnss_df['latitude'].min():.6f}°, {gnss_df['latitude'].max():.6f}°]")
        print(f"  Longitude range: [{gnss_df['longitude'].min():.6f}°, {gnss_df['longitude'].max():.6f}°]")
        print(f"  Altitude range:  [{gnss_df['altitude'].min():.2f}m, {gnss_df['altitude'].max():.2f}m]")

    print("==========================================================")


def main():
    parser = argparse.ArgumentParser(description="Analyze sensor dataset quality.")
    parser.add_argument("--imu", type=str, default="data/sample/imu_sample.csv")
    parser.add_argument("--gnss", type=str, default="data/sample/gnss_sample.csv")
    parser.add_argument("--gt", type=str, default="data/sample/ground_truth_sample.csv")
    args = parser.parse_args()

    analyze_dataset(
        project_root / args.imu,
        project_root / args.gnss,
        project_root / args.gt,
    )


if __name__ == "__main__":
    main()
