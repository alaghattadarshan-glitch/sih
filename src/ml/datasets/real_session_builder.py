"""Multi-session dataset builder and Leave-One-Session-Out partitioner for PyTorch ML training.

Ensures strict trajectory-level isolation across train, validation, and test sets,
preventing cross-session information leakage.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.session import DatasetSession
from src.ml.datasets.feature_extractor import extract_imu_windows
from src.ml.datasets.target_builder import build_window_targets
from src.ml.datasets.imu_dataset import IMUWindowDataset


class MultiSessionDatasetBuilder:
    """Extracts windows and builds PyTorch datasets from lists of DatasetSession objects."""

    def __init__(
        self,
        window_size_samples: int = 100,
        stride_samples: int = 50,
        imu_rate_hz: float = 100.0,
    ):
        self.window_size_samples = window_size_samples
        self.stride_samples = stride_samples
        self.imu_rate_hz = imu_rate_hz

    def build_dataset_from_sessions(
        self,
        sessions: List[DatasetSession],
        feature_mean: Optional[np.ndarray] = None,
        feature_std: Optional[np.ndarray] = None,
    ) -> Tuple[IMUWindowDataset, np.ndarray, np.ndarray]:
        """Extract windows and targets from a list of sessions, returning a consolidated IMUWindowDataset.
        
        If feature_mean and feature_std are not provided, they are calculated from these sessions
        (e.g., when building the training split).
        """
        all_windows = []
        all_targets = []
        all_time_ranges = []
        all_traj_ids = []
        feature_names = [
            "accel_x", "accel_y", "accel_z",
            "gyro_x", "gyro_y", "gyro_z",
            "accel_norm", "gyro_norm"
        ]
        target_names = ["delta_p_east", "delta_p_north", "delta_p_up"]

        for session in sessions:
            if not session.imu_observations:
                continue
            if not session.ground_truth_observations:
                continue
            if session.local_frame is None:
                continue

            windows, time_ranges, f_names = extract_imu_windows(
                session.imu_observations,
                window_size_samples=self.window_size_samples,
                stride_samples=self.stride_samples,
                feature_columns=feature_names,
            )

            targets, t_names = build_window_targets(
                time_ranges,
                session.ground_truth_observations,
                session.local_frame,
            )

            if len(windows) > 0 and len(targets) > 0 and len(windows) == len(targets):
                all_windows.append(windows)
                all_targets.append(targets)
                all_time_ranges.extend(time_ranges)
                all_traj_ids.extend([session.session_id] * len(windows))
                feature_names = f_names
                target_names = t_names

        if not all_windows:
            empty_windows = np.zeros((0, self.window_size_samples, 8), dtype=np.float32)
            empty_targets = np.zeros((0, 3), dtype=np.float32)
            dataset = IMUWindowDataset(
                features=empty_windows,
                targets=empty_targets,
                time_ranges=[],
                feature_names=feature_names,
                target_names=target_names,
                trajectory_id="empty",
                feature_mean=feature_mean,
                feature_std=feature_std,
            )
            return dataset, np.zeros(8, dtype=np.float32), np.ones(8, dtype=np.float32)

        concat_windows = np.concatenate(all_windows, axis=0)
        concat_targets = np.concatenate(all_targets, axis=0)

        # Calculate normalization parameters if not provided
        if feature_mean is None or feature_std is None:
            feature_mean = np.mean(concat_windows, axis=(0, 1)).astype(np.float32)
            feature_std = np.std(concat_windows, axis=(0, 1)).astype(np.float32)
            # Prevent division by zero
            feature_std = np.where(feature_std < 1e-6, 1.0, feature_std)

        dataset = IMUWindowDataset(
            features=concat_windows,
            targets=concat_targets,
            time_ranges=all_time_ranges,
            feature_names=feature_names,
            target_names=target_names,
            trajectory_id="multi_session",
            feature_mean=feature_mean,
            feature_std=feature_std,
        )

        return dataset, feature_mean, feature_std

    def leave_one_session_out_splits(
        self,
        sessions: List[DatasetSession],
    ) -> List[Dict[str, Any]]:
        """Generate Leave-One-Session-Out (LOSO) cross-validation folds.
        
        For N sessions:
        - Test: Session i
        - Val: Session (i+1)%N
        - Train: Remaining N-2 sessions
        """
        num_sessions = len(sessions)
        if num_sessions < 3:
            raise ValueError(f"LOSO cross-validation requires at least 3 sessions, found {num_sessions}")

        folds = []
        for i in range(num_sessions):
            test_session = sessions[i]
            val_idx = (i + 1) % num_sessions
            val_session = sessions[val_idx]
            train_sessions = [sessions[j] for j in range(num_sessions) if j != i and j != val_idx]

            # Build train dataset and derive normalization
            train_ds, mean, std = self.build_dataset_from_sessions(train_sessions)
            val_ds, _, _ = self.build_dataset_from_sessions(
                [val_session], feature_mean=mean, feature_std=std
            )
            test_ds, _, _ = self.build_dataset_from_sessions(
                [test_session], feature_mean=mean, feature_std=std
            )

            folds.append({
                "fold": i,
                "test_session_id": test_session.session_id,
                "val_session_id": val_session.session_id,
                "train_session_ids": [s.session_id for s in train_sessions],
                "train_dataset": train_ds,
                "val_dataset": val_ds,
                "test_dataset": test_ds,
                "feature_mean": mean,
                "feature_std": std,
            })

        return folds
