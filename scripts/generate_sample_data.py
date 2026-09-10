#!/usr/bin/env python3
"""Script to generate deterministic sample synthetic sensor datasets in data/sample/.

Creates:
- data/sample/imu_sample.csv
- data/sample/gnss_sample.csv
- data/sample/ground_truth_sample.csv

Used for offline unit testing and pipeline demonstration.
"""

import sys
from pathlib import Path

# Ensure project root is in path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data.loaders.synthetic import generate_synthetic_trajectory
from src.data.conversions import (
    imu_to_dataframe,
    gnss_to_dataframe,
    ground_truth_to_dataframe,
)


def main():
    sample_dir = project_root / "data" / "sample"
    sample_dir.mkdir(parents=True, exist_ok=True)

    print("Generating synthetic sample trajectory (duration=80s, seed=42)...")
    imu, gnss, gt = generate_synthetic_trajectory(
        seed=42,
        duration_sec=80.0,
        imu_rate_hz=100.0,
        gnss_rate_hz=1.0,
    )

    imu_df = imu_to_dataframe(imu)
    gnss_df = gnss_to_dataframe(gnss)
    gt_df = ground_truth_to_dataframe(gt)

    imu_file = sample_dir / "imu_sample.csv"
    gnss_file = sample_dir / "gnss_sample.csv"
    gt_file = sample_dir / "ground_truth_sample.csv"

    imu_df.to_csv(imu_file, index=False)
    gnss_df.to_csv(gnss_file, index=False)
    gt_df.to_csv(gt_file, index=False)

    print(f"Sample IMU written to:          {imu_file} ({len(imu_df)} rows)")
    print(f"Sample GNSS written to:         {gnss_file} ({len(gnss_df)} rows)")
    print(f"Sample Ground Truth written to: {gt_file} ({len(gt_df)} rows)")
    print("Sample generation complete.")


if __name__ == "__main__":
    main()
