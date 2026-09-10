"""Unit tests for ML Dataset Pipeline (Windowing, Target Generation, PyTorch Dataset, Splitting)."""

import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.observations import IMUObservation, GroundTruthObservation
from src.coordinate_transforms import LocalFrame
from src.ml.datasets.feature_extractor import extract_imu_windows
from src.ml.datasets.target_builder import build_window_targets
from src.ml.datasets.imu_dataset import IMUWindowDataset
from src.ml.datasets.splitter import split_by_trajectory


def make_imu_sequence(n_samples=250, dt=0.01):
    imu_list = []
    for i in range(n_samples):
        t = i * dt
        obs = IMUObservation(
            timestamp=t,
            accelerometer_x=0.1 * np.sin(t),
            accelerometer_y=0.0,
            accelerometer_z=9.81,
            gyroscope_x=0.01,
            gyroscope_y=0.0,
            gyroscope_z=0.0,
        )
        imu_list.append(obs)
    return imu_list


def make_gt_sequence(n_samples=250, dt=0.01, origin_lat=12.9716, origin_lon=77.5946):
    local_frame = LocalFrame(origin_lat, origin_lon, 920.0)
    gt_list = []
    for i in range(n_samples):
        t = i * dt
        # East displacement: 0.5 * t
        east_m = 0.5 * t
        lat, lon, alt = local_frame.from_enu(east_m, 0.0, 0.0)
        gt = GroundTruthObservation(
            timestamp=t,
            latitude=lat,
            longitude=lon,
            altitude=alt,
            velocity_east=0.5,
            velocity_north=0.0,
            velocity_up=0.0,
        )
        gt_list.append(gt)
    return local_frame, gt_list


def test_extract_imu_windows():
    imu_list = make_imu_sequence(n_samples=250)
    windows, time_ranges, feature_names = extract_imu_windows(
        imu_list, window_size_samples=100, stride_samples=50
    )

    # For 250 samples: start at 0, 50, 100, 150 (end=250) -> 4 windows
    assert windows.shape == (4, 100, 9)
    assert len(time_ranges) == 4
    assert "accel_norm" in feature_names
    assert "rel_time" in feature_names


def test_build_window_targets():
    imu_list = make_imu_sequence(n_samples=250)
    local_frame, gt_list = make_gt_sequence(n_samples=250)

    windows, time_ranges, _ = extract_imu_windows(imu_list, window_size_samples=100, stride_samples=50)

    # 1. Displacement target
    disp_targets, target_names = build_window_targets(
        time_ranges, gt_list, local_frame, target_type="displacement_enu"
    )
    assert disp_targets.shape == (4, 3)
    assert target_names == ["delta_p_east", "delta_p_north", "delta_p_up"]
    # 1.0s window at 0.5 m/s -> delta_p_east ~ 0.495m
    assert np.isclose(disp_targets[0, 0], 0.495, atol=0.02)

    # 2. Velocity target
    vel_targets, vel_names = build_window_targets(
        time_ranges, gt_list, local_frame, target_type="velocity_end"
    )
    assert vel_targets.shape == (4, 3)
    assert vel_targets[0, 0] == 0.5


def test_imu_window_dataset_pytorch():
    imu_list = make_imu_sequence(n_samples=250)
    local_frame, gt_list = make_gt_sequence(n_samples=250)

    windows, time_ranges, feature_names = extract_imu_windows(imu_list, window_size_samples=100, stride_samples=50)
    targets, target_names = build_window_targets(time_ranges, gt_list, local_frame, target_type="displacement_enu")

    dataset = IMUWindowDataset(
        features=windows,
        targets=targets,
        time_ranges=time_ranges,
        feature_names=feature_names,
        target_names=target_names,
        trajectory_id="traj_test_1",
    )

    assert len(dataset) == 4
    assert dataset.window_size == 100
    assert dataset.feature_dim == 9
    assert dataset.target_dim == 3

    x_sample, y_sample = dataset[0]
    assert isinstance(x_sample, torch.Tensor)
    assert isinstance(y_sample, torch.Tensor)
    assert x_sample.shape == (100, 9)
    assert y_sample.shape == (3,)

    # Test PyTorch DataLoader integration
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    batch_x, batch_y = next(iter(loader))
    assert batch_x.shape == (2, 100, 9)
    assert batch_y.shape == (2, 3)


def test_dataset_nan_rejection():
    imu_list = make_imu_sequence(n_samples=250)
    imu_list[10].accelerometer_x = np.nan

    with pytest.raises(ValueError, match="Non-finite value detected"):
        extract_imu_windows(imu_list)


def test_trajectory_aware_splitting():
    # Build 5 dummy trajectory datasets
    ds_list = []
    local_frame, gt_list = make_gt_sequence(n_samples=250)

    for i in range(5):
        imu_list = make_imu_sequence(n_samples=250)
        w, tr, fn = extract_imu_windows(imu_list, window_size_samples=100, stride_samples=50)
        t, tn = build_window_targets(tr, gt_list, local_frame)
        ds = IMUWindowDataset(
            features=w,
            targets=t,
            time_ranges=tr,
            feature_names=fn,
            target_names=tn,
            trajectory_id=f"session_{i+1}",
        )
        ds_list.append(ds)

    train_ds, val_ds, test_ds, split_meta = split_by_trajectory(
        ds_list, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42
    )

    train_ids = set(split_meta["train"])
    val_ids = set(split_meta["val"])
    test_ids = set(split_meta["test"])

    # Prove zero leakage across splits
    assert train_ids.isdisjoint(val_ids), "Train and Val splits share trajectory IDs!"
    assert train_ids.isdisjoint(test_ids), "Train and Test splits share trajectory IDs!"
    assert val_ids.isdisjoint(test_ids), "Val and Test splits share trajectory IDs!"

    assert len(train_ids) + len(val_ids) + len(test_ids) == 5
    assert len(train_ds) > 0
    assert len(val_ds) > 0
    assert len(test_ds) > 0
