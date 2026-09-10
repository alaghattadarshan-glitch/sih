"""1D CNN / Temporal Convolutional Network (TCN) for Inertial Drift Prediction.

Predicts 3D displacement vectors [delta_p_east, delta_p_north, delta_p_up] from
sliding high-rate IMU feature windows [batch, sequence_length, num_features].
"""

from typing import List, Dict, Any, Optional
import torch
import torch.nn as nn


class ConvBlock1D(nn.Module):
    """1D Convolutional Residual Block with Batch Normalization and Dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Residual shortcut if channel dimensions change
        self.shortcut = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        return out + residual


class TCNDriftModel(nn.Module):
    """Temporal Convolutional Network for IMU Inertial Drift Correction."""

    def __init__(
        self,
        in_channels: int = 8,
        output_dim: int = 3,
        num_channels: Optional[List[int]] = None,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        """Initialize TCN Drift Correction Model.

        Args:
            in_channels (int): Input feature dimension per time step (default 8).
            output_dim (int): Target output dimension (default 3: [east, north, up]).
            num_channels (Optional[List[int]]): Conv channel layers list (default [32, 64, 128]).
            kernel_size (int): 1D Conv kernel size (default 3).
            dropout (float): Dropout probability (default 0.1).
        """
        super().__init__()
        self.in_channels = in_channels
        self.output_dim = output_dim
        self.num_channels = num_channels if num_channels is not None else [32, 64, 128]
        self.kernel_size = kernel_size
        self.dropout_rate = dropout

        layers = []
        curr_in = in_channels
        for c in self.num_channels:
            layers.append(
                ConvBlock1D(
                    in_channels=curr_in,
                    out_channels=c,
                    kernel_size=kernel_size,
                    dropout=dropout,
                )
            )
            curr_in = c

        self.conv_net = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Final Linear Regression Head
        last_channels = self.num_channels[-1] if self.num_channels else in_channels
        self.fc_head = nn.Sequential(
            nn.Linear(last_channels, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_dim),
        )

        self._reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Feature tensor of shape [batch, L_window, D_features]
                              or channel-first [batch, D_features, L_window].

        Returns:
            torch.Tensor: Predicted displacement tensor of shape [batch, output_dim].
        """
        # If input is [batch, L_window, D_features], transpose to [batch, D_features, L_window]
        if x.ndim == 3 and x.shape[1] != self.in_channels and x.shape[2] == self.in_channels:
            x = x.transpose(1, 2)

        feat = self.conv_net(x)  # [batch, C_last, L_window]
        pooled = self.global_pool(feat).squeeze(-1)  # [batch, C_last]
        out = self.fc_head(pooled)  # [batch, output_dim]
        return out

    def _reset_parameters(self):
        """Kaiming initialization for conviction weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def get_config(self) -> Dict[str, Any]:
        """Return model architecture hyperparameter dictionary."""
        return {
            "model_type": "tcn",
            "in_channels": self.in_channels,
            "output_dim": self.output_dim,
            "num_channels": self.num_channels,
            "kernel_size": self.kernel_size,
            "dropout": self.dropout_rate,
        }
