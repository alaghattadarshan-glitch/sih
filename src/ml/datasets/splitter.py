"""Trajectory-Aware Dataset Splitting Module.

Splits multi-trajectory datasets strictly by trajectory/session IDs to ensure
that overlapping temporal windows from the same continuous trajectory do NOT leak across
train, validation, and test partitions.
"""

from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import torch
from torch.utils.data import ConcatDataset
from src.ml.datasets.imu_dataset import IMUWindowDataset


def split_by_trajectory(
    datasets: List[IMUWindowDataset],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[ConcatDataset, ConcatDataset, ConcatDataset, Dict[str, List[str]]]:
    """Split a collection of trajectory datasets strictly by whole trajectory IDs.

    Args:
        datasets (List[IMUWindowDataset]): List of trajectory datasets.
        train_ratio (float): Fraction of trajectories allocated to training.
        val_ratio (float): Fraction of trajectories allocated to validation.
        test_ratio (float): Fraction of trajectories allocated to testing.
        seed (int): Random seed for reproducible trajectory shuffling.

    Returns:
        Tuple[ConcatDataset, ConcatDataset, ConcatDataset, Dict[str, List[str]]]:
            - Training PyTorch ConcatDataset
            - Validation PyTorch ConcatDataset
            - Testing PyTorch ConcatDataset
            - Dictionary mapping split names ("train", "val", "test") to list of trajectory_ids.

    Raises:
        ValueError: If dataset list is empty, ratios do not sum to 1.0, or duplicate trajectory_ids exist.
    """
    if not datasets:
        raise ValueError("Cannot split an empty list of datasets.")

    if not math_approx_equal(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError(f"Ratios must sum to 1.0 (got {train_ratio + val_ratio + test_ratio:.4f})")

    # Collect trajectory IDs and check for duplicates
    traj_map: Dict[str, IMUWindowDataset] = {}
    for ds in datasets:
        tid = ds.trajectory_id
        if tid in traj_map:
            raise ValueError(f"Duplicate trajectory_id detected: {tid}")
        traj_map[tid] = ds

    traj_ids = sorted(list(traj_map.keys()))
    n_trajs = len(traj_ids)

    if n_trajs < 3:
        # Fallback allocation for small number of trajectories
        # E.g. with 3 trajs: 1 train, 1 val, 1 test
        train_ids = [traj_ids[0]]
        val_ids = [traj_ids[1]] if n_trajs > 1 else [traj_ids[0]]
        test_ids = [traj_ids[2]] if n_trajs > 2 else [traj_ids[0]]
    else:
        rng = np.random.RandomState(seed)
        shuffled_ids = list(traj_ids)
        rng.shuffle(shuffled_ids)

        n_train = max(1, int(round(n_trajs * train_ratio)))
        n_val = max(1, int(round(n_trajs * val_ratio)))
        
        # Ensure at least 1 trajectory per split if n_trajs >= 3
        if n_train + n_val >= n_trajs:
            n_train = max(1, n_trajs - 2)
            n_val = 1

        train_ids = shuffled_ids[:n_train]
        val_ids = shuffled_ids[n_train : n_train + n_val]
        test_ids = shuffled_ids[n_train + n_val :]

        if not test_ids:
            test_ids = [val_ids.pop()]

    # Verify zero leakage across splits
    train_set_ids = set(train_ids)
    val_set_ids = set(val_ids)
    test_set_ids = set(test_ids)

    assert train_set_ids.isdisjoint(val_set_ids), "Leakage detected: overlap between train and val splits!"
    assert train_set_ids.isdisjoint(test_set_ids), "Leakage detected: overlap between train and test splits!"
    assert val_set_ids.isdisjoint(test_set_ids), "Leakage detected: overlap between val and test splits!"

    train_ds_list = [traj_map[tid] for tid in train_ids]
    val_ds_list = [traj_map[tid] for tid in val_ids]
    test_ds_list = [traj_map[tid] for tid in test_ids]

    train_concat = ConcatDataset(train_ds_list)
    val_concat = ConcatDataset(val_ds_list)
    test_concat = ConcatDataset(test_ds_list)

    split_metadata = {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
    }

    return train_concat, val_concat, test_concat, split_metadata


def math_approx_equal(a: float, b: float, tol: float = 1e-4) -> bool:
    """Check approximate equality of floats."""
    return abs(a - b) <= tol
