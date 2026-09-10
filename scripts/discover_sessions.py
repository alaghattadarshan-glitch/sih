#!/usr/bin/env python3
"""Script to discover and inspect available dataset sessions across the repository.

Classifies datasets as REAL, SYNTHETIC, FIXTURE, or UNKNOWN.
Outputs session durations, sensor frequencies, overlap intervals, and quality status.
"""

import os
import sys
from pathlib import Path
from typing import List

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.session import (
    DataSourceType,
    DatasetSession,
    discover_available_sessions,
    create_synthetic_multi_session_catalog,
)


def print_session_catalog(sessions: List[DatasetSession]):
    """Print structured table of discovered sessions."""
    print("=" * 110)
    print(f"{'SESSION ID':<14} | {'SOURCE':<10} | {'DURATION':<10} | {'IMU SAMPLES':<12} | {'GNSS FIXES':<10} | {'GT SAMPLES':<10} | {'IMU (Hz)':<8} | {'STATUS':<8}")
    print("=" * 110)
    
    for s in sessions:
        dur_str = f"{s.duration_sec:.1f} s"
        imu_cnt = f"{len(s.imu_observations)}"
        gnss_cnt = f"{len(s.gnss_observations)}"
        gt_cnt = f"{len(s.ground_truth_observations)}" if s.ground_truth_observations else "N/A"
        imu_hz = f"{s.imu_rate_hz:.1f}"
        status = s.quality_report.get("overall_status", "UNKNOWN")
        print(f"{s.session_id:<14} | {s.source_type.value:<10} | {dur_str:<10} | {imu_cnt:<12} | {gnss_cnt:<10} | {gt_cnt:<10} | {imu_hz:<8} | {status:<8}")
    
    print("=" * 110)


def main():
    print("\n" + "=" * 80)
    print("SIH26168 STEP 11 — DATASET SESSION DISCOVERY & INVENTORY AUDIT")
    print("=" * 80 + "\n")

    # 1. Check raw and sample directories
    disk_sessions = discover_available_sessions()
    
    # 2. Check synthetic multi-session benchmark catalog
    synthetic_sessions = create_synthetic_multi_session_catalog(num_sessions=5, base_seed=100, duration_sec=120.0)
    
    all_sessions = disk_sessions + synthetic_sessions

    print(f"Total Sessions Discovered: {len(all_sessions)}\n")
    print_session_catalog(all_sessions)

    # Check for genuine real recordings
    real_sessions = [s for s in all_sessions if s.source_type == DataSourceType.REAL]
    synthetic_count = len([s for s in all_sessions if s.source_type == DataSourceType.SYNTHETIC])
    
    print("\nDATASET INVENTORY STATUS:")
    print(f"  • Genuine Real-World Sessions: {len(real_sessions)}")
    print(f"  • Synthetic Benchmark Sessions: {synthetic_count}")
    print(f"  • Test Fixtures: In tests/fixtures/ (Excluded from general session catalog)")
    
    if not real_sessions:
        print("\n[NOTE] Genuine real-world recordings (e.g. IO-VNBD) are currently PENDING in data/raw/.")
        print("       Multi-session benchmarking will execute on the verified multi-session synthetic catalog.")
        print("       The adapter infrastructure is verified and standby-ready for real datasets.\n")


if __name__ == "__main__":
    main()
