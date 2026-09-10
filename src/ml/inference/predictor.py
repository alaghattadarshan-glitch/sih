"""Inference Utility for AI Inertial Drift Prediction Model.

Loads trained model weights and feature normalization parameters from disk
and provides a clean single-window prediction interface for real-time navigation pipelines.
"""

import json
import os
from typing import Optional, Union, Dict, Any
import numpy as np
import torch

from src.ml.models.tcn_drift import TCNDriftModel


class DriftPredictor:
    """Inference predictor wrapping trained TCNDriftModel with feature normalization."""

    def __init__(
        self,
        checkpoint_path: str,
        config_path: Optional[str] = None,
        device: str = "cpu",
    ):
        """Initialize inference engine.

        Args:
            checkpoint_path (str): File path to trained PyTorch checkpoint .pt file.
            config_path (Optional[str]): Path to model config JSON file (optional if inside checkpoint).
            device (str): Inference device ("cpu" or "cuda").

        Raises:
            FileNotFoundError: If checkpoint file does not exist.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")

        self.device = device
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Load architecture parameters
        if "model_config" in checkpoint:
            cfg = checkpoint["model_config"]
        elif config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                cfg = json.load(f)
        else:
            cfg = {
                "in_channels": 8,
                "output_dim": 3,
                "num_channels": [32, 64, 128],
                "kernel_size": 3,
                "dropout": 0.1,
            }

        self.model = TCNDriftModel(
            in_channels=cfg.get("in_channels", 8),
            output_dim=cfg.get("output_dim", 3),
            num_channels=cfg.get("num_channels", [32, 64, 128]),
            kernel_size=cfg.get("kernel_size", 3),
            dropout=cfg.get("dropout", 0.1),
        )

        # Load model weights
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.to(device)
        self.model.eval()

        # Load normalization parameters
        if "feature_mean" in checkpoint and "feature_std" in checkpoint:
            self.feature_mean = np.array(checkpoint["feature_mean"], dtype=np.float32)
            self.feature_std = np.array(checkpoint["feature_std"], dtype=np.float32)
        else:
            # Default fallback identity scaling
            self.feature_mean = np.zeros(cfg.get("in_channels", 8), dtype=np.float32)
            self.feature_std = np.ones(cfg.get("in_channels", 8), dtype=np.float32)

    def predict_window(self, imu_window: np.ndarray) -> np.ndarray:
        """Predict 3D displacement vector [delta_p_east, delta_p_north, delta_p_up] for a single window.

        Args:
            imu_window (np.ndarray): Unnormalized raw IMU window matrix of shape [100, 8] or [1, 100, 8].

        Returns:
            np.ndarray: Predicted 1D displacement vector [3] in meters.
        """
        arr = np.array(imu_window, dtype=np.float32)
        if arr.ndim == 2:
            arr = np.expand_dims(arr, axis=0)  # [1, L, D]
        if arr.ndim != 3:
            raise ValueError(f"Expected 2D [L, D] or 3D [1, L, D] window, got shape {arr.shape}")

        # Apply training normalization
        norm_window = (arr - self.feature_mean) / self.feature_std
        x_tensor = torch.from_numpy(norm_window).to(self.device)

        with torch.no_grad():
            pred_tensor = self.model(x_tensor)
            pred_np = pred_tensor.cpu().numpy().squeeze(0)  # [3]

        return pred_np.astype(np.float32)

    def predict(self, imu_window: np.ndarray) -> np.ndarray:
        """Alias for predict_window."""
        return self.predict_window(imu_window)
