"""End-to-End Replay Integration Tests: Python Reference vs C++ Native Navigation Core."""

import math
import numpy as np
import pytest

from src.data.loaders.synthetic import generate_synthetic_trajectory
from src.navigation.quaternion import euler_to_quaternion
from src.evaluation.edge_parity import EdgeParityEvaluator


def test_python_vs_cpp_full_trajectory_replay_parity():
    """Verify that Python reference and C++ engine produce tightly aligned trajectories."""
    imu_list, gnss_list, gt_list = generate_synthetic_trajectory(seed=42, duration_sec=30.0)

    evaluator = EdgeParityEvaluator()
    t0 = imu_list[0].timestamp
    init_llh = (gt_list[0].latitude, gt_list[0].longitude, gt_list[0].altitude)
    init_vel = np.array([gt_list[0].velocity_east, gt_list[0].velocity_north, gt_list[0].velocity_up])

    roll_rad = math.radians(gt_list[0].roll if gt_list[0].roll is not None else 0.0)
    pitch_rad = math.radians(gt_list[0].pitch if gt_list[0].pitch is not None else 0.0)
    yaw_rad = math.radians(gt_list[0].yaw if gt_list[0].yaw is not None else 0.0)
    init_quat = euler_to_quaternion(roll_rad, pitch_rad, yaw_rad)

    results = evaluator.run_replay_comparison(
        imu_observations=imu_list,
        gnss_observations=gnss_list,
        initial_time=t0,
        initial_llh=init_llh,
        initial_velocity=init_vel,
        initial_quat=init_quat,
    )

    # Tight numerical tolerances
    assert results["parity_status"] == "PASS"
    assert results["pos_rmse"] < 0.05, f"Position RMSE {results['pos_rmse']:.4f} m exceeded tolerance."
    assert results["vel_rmse"] < 0.01, f"Velocity RMSE {results['vel_rmse']:.4f} m/s exceeded tolerance."
    assert results["heading_mean_diff_deg"] < 0.1, f"Heading diff {results['heading_mean_diff_deg']:.4f} deg exceeded tolerance."

    # Verify real-time processing performance (<10 ms = 10,000 us)
    assert results["latency_mean_us"] < 500.0, f"Mean latency {results['latency_mean_us']:.2f} us too high."
