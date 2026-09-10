"""PyTorch Compatible IMU Window Dataset.

Wraps extracted IMU feature windows and supervised motion targets into a PyTorch Dataset
suitable for PyTorch DataLoader, batching, and normalization.
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import torch
from torch.utils.data import Dataset


class IMUWindowDataset(Dataset):
    """PyTorch Dataset for windowed IMU features and ground-truth drift targets."""

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        time_ranges: List[Tuple[float, float]],
        feature_names: List[str],
        target_names: List[str],
        trajectory_id: str = "traj_0",
        feature_mean: Optional[np.ndarray] = None,
        feature_std: Optional[np.ndarray] = None,
    ):
        """Initialize PyTorch IMU Window Dataset.

        Args:
            features (np.ndarray): Array of shape [N_windows, L_window, D_features].
            targets (np.ndarray): Array of shape [N_windows, D_targets].
            time_ranges (List[Tuple[float, float]]): List of (t_start, t_end) timestamps.
            feature_names (List[str]): Names of feature columns.
            target_names (List[str]): Names of target columns.
            trajectory_id (str): Unique trajectory/session identifier.
            feature_mean (Optional[np.ndarray]): Mean array of shape [D_features] for normalization.
            feature_std (Optional[np.ndarray]): Std array of shape [D_features] for normalization.

        Raises:
            ValueError: If dimensions mismatch or NaNs/Infs are present.
        """
        if features.ndim != 3:
            raise ValueError(f"Expected 3D features array [N, L, D], got shape {features.shape}")
        if targets.ndim != 2:
            raise ValueError(f"Expected 2D targets array [N, D], got shape {targets.shape}")
        if features.shape[0] != targets.shape[0]:
            raise ValueError(
                f"Number of feature windows ({features.shape[0]}) does not match number of targets ({targets.shape[0]})."
            )
        if len(time_ranges) != features.shape[0]:
            raise ValueError(
                f"Number of time ranges ({len(time_ranges)}) does not match number of windows ({features.shape[0]})."
            )

        # Quality checks: NaN/Inf validation
        if not np.all(np.isfinite(features)):
            raise ValueError("Detected non-finite (NaN/Inf) values in features array.")
        if not np.all(np.isfinite(targets)):
            raise ValueError("Detected non-finite (NaN/Inf) values in targets array.")

        self.features_raw = features.astype(np.float32)
        self.targets_raw = targets.astype(np.float32)
        self.time_ranges = time_ranges
        self.feature_names = list(feature_names)
        self.target_names = list(target_names)
        self.trajectory_id = trajectory_id

        # Compute or apply normalization parameters
        d_feat = self.features_raw.shape[2]
        if feature_mean is not None:
            self.feature_mean = np.array(feature_mean, dtype=np.float32)
        else:
            # Calculate feature mean across windows and sequence length
            self.feature_mean = np.mean(self.features_raw, axis=(0, 1))

        if feature_std is not None:
            self.feature_std = np.array(feature_std, dtype=np.float32)
        else:
            self.feature_std = np.std(self.features_raw, axis=(0, 1))
            # Protect against division by zero for constant features
            self.feature_std[self.feature_std < 1e-8] = 1.0

        # Normalize features
        norm_features = (self.features_raw - self.feature_mean) / self.feature_std

        # Convert to PyTorch Tensors
        self.x_tensor = torch.from_numpy(norm_features)
        self.y_tensor = torch.from_numpy(self.targets_raw)

    def __len__(self) -> int:
        """Return total number of windows."""
        return self.x_tensor.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fetch normalized feature tensor and target tensor for window index.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Feature tensor of shape [L_window, D_features].
                - Target tensor of shape [D_targets].
        """
        return self.x_tensor[idx], self.y_tensor[idx]

    @property
    def window_size(self) -> int:
        """Number of time steps L_window per sample."""
        return self.x_tensor.shape[1]

    @property
    def feature_dim(self) -> int:
        """Number of input features D_features."""
        return self.x_tensor.shape[2]

    @property
    def target_dim(self) -> int:
        """Number of output target dimensions D_targets."""
        return self.y_tensor.shape[1]
