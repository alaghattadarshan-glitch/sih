#!/usr/bin/env python3
"""Edge Model Exporter for SIH26168.

Exports the PyTorch TCN inertial drift model to ONNX format, generates frozen
normalization configurations, writes metadata, and verifies numerical parity
between PyTorch and ONNX Runtime across deterministic test windows.
"""

import argparse
import json
import os
import sys
from typing import Dict, Any

import numpy as np
import torch
import onnx
import onnxruntime as ort

# Add repository root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.ml.models.tcn_drift import TCNDriftModel


def export_edge_model(
    checkpoint_path: str = "results/ml_training/best_drift_model.pt",
    output_dir: str = "results/edge_model",
    opset_version: int = 18,
    num_test_windows: int = 100,
) -> Dict[str, Any]:
    """Export PyTorch checkpoint to ONNX and frozen metadata.

    Args:
        checkpoint_path: Path to PyTorch model checkpoint.
        output_dir: Output directory for edge model artifacts.
        opset_version: ONNX opset version (default 18).
        num_test_windows: Number of deterministic verification windows.

    Returns:
        Dict[str, Any]: Parity verification summary.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    cfg = checkpoint.get("model_config", {})
    feature_mean = checkpoint.get("feature_mean", [])
    feature_std = checkpoint.get("feature_std", [])
    feature_names = checkpoint.get(
        "feature_names",
        [
            "accel_x",
            "accel_y",
            "accel_z",
            "gyro_x",
            "gyro_y",
            "gyro_z",
            "accel_norm",
            "gyro_norm",
        ],
    )
    target_names = checkpoint.get("target_names", ["delta_p_east", "delta_p_north", "delta_p_up"])

    # Instantiate PyTorch Model
    model = TCNDriftModel(
        in_channels=cfg.get("in_channels", 8),
        output_dim=cfg.get("output_dim", 3),
        num_channels=cfg.get("num_channels", [32, 64, 128]),
        kernel_size=cfg.get("kernel_size", 3),
        dropout=cfg.get("dropout", 0.1),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Total parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model instantiated with {total_params:,} parameters.")

    # 1. Export ONNX
    onnx_path = os.path.join(output_dir, "model.onnx")
    dummy_input = torch.randn(1, 100, 8, dtype=torch.float32)

    print(f"Exporting ONNX model to: {onnx_path} (opset {opset_version})...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["imu_features"],
        output_names=["predicted_displacement"],
        dynamic_axes={
            "imu_features": {0: "batch_size"},
            "predicted_displacement": {0: "batch_size"},
        },
        opset_version=opset_version,
    )

    # Verify ONNX model proto
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    onnx_size_bytes = os.path.getsize(onnx_path)
    print(f"ONNX model verified successfully (Size: {onnx_size_bytes / 1024:.2f} KB).")

    # 2. Save Normalization Configuration
    norm_dict = {
        "version": "1.0.0",
        "feature_names": feature_names,
        "feature_mean": [float(m) for m in feature_mean],
        "feature_std": [float(s) for s in feature_std],
        "target_names": target_names,
    }
    norm_path = os.path.join(output_dir, "normalization.json")
    with open(norm_path, "w") as f:
        json.dump(norm_dict, f, indent=2)
    print(f"Saved normalization configuration to: {norm_path}")

    # 3. Save Model Metadata
    metadata_dict = {
        "model_name": "TCNDriftModel",
        "version": "1.0.0",
        "input_name": "imu_features",
        "input_shape": [1, 100, 8],
        "output_name": "predicted_displacement",
        "output_shape": [1, 3],
        "feature_dimension": 8,
        "sequence_length": 100,
        "sampling_rate_hz": 100.0,
        "total_parameters": total_params,
        "onnx_opset_version": opset_version,
        "onnx_size_bytes": onnx_size_bytes,
        "training_checkpoint": os.path.basename(checkpoint_path),
        "best_val_loss": checkpoint.get("best_val_loss", None),
    }
    meta_path = os.path.join(output_dir, "model_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata_dict, f, indent=2)
    print(f"Saved model metadata to: {meta_path}")

    # 4. Numerical Parity Verification
    print(f"Running numerical parity check across {num_test_windows} deterministic windows...")
    np.random.seed(42)
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name

    max_abs_diff = 0.0
    abs_diffs = []

    for i in range(num_test_windows):
        test_data = np.random.randn(1, 100, 8).astype(np.float32)
        torch_tensor = torch.from_numpy(test_data)

        with torch.no_grad():
            torch_pred = model(torch_tensor).numpy()

        ort_pred = session.run(None, {input_name: test_data})[0]
        diff = float(np.max(np.abs(torch_pred - ort_pred)))
        abs_diffs.append(diff)
        if diff > max_abs_diff:
            max_abs_diff = diff

    rmse_diff = float(np.sqrt(np.mean(np.array(abs_diffs) ** 2)))
    mae_diff = float(np.mean(abs_diffs))

    parity_summary = {
        "num_windows": num_test_windows,
        "max_absolute_difference": max_abs_diff,
        "mae_difference": mae_diff,
        "rmse_difference": rmse_diff,
        "parity_status": "PASS" if max_abs_diff < 1e-4 else "FAIL",
    }
    print(f"Parity Results: Max Diff = {max_abs_diff:.6e}, RMSE = {rmse_diff:.6e} -> {parity_summary['parity_status']}")

    parity_path = os.path.join(output_dir, "parity_report.json")
    with open(parity_path, "w") as f:
        json.dump(parity_summary, f, indent=2)

    return parity_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PyTorch TCN to ONNX Edge Model")
    parser.add_argument("--checkpoint", type=str, default="results/ml_training/best_drift_model.pt")
    parser.add_argument("--output", type=str, default="results/edge_model")
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args()

    export_edge_model(
        checkpoint_path=args.checkpoint,
        output_dir=args.output,
        opset_version=args.opset,
    )
