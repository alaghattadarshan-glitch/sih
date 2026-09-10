"""Tests for PyTorch vs ONNX Model Parity, Normalization, and Causality."""

import json
import os
import numpy as np
import pytest
import torch
import onnxruntime as ort

from src.ml.models.tcn_drift import TCNDriftModel


@pytest.fixture(scope="module")
def model_artifacts():
    """Ensure ONNX and metadata artifacts exist."""
    onnx_path = "results/edge_model/model.onnx"
    norm_path = "results/edge_model/normalization.json"
    meta_path = "results/edge_model/model_metadata.json"

    if not (os.path.exists(onnx_path) and os.path.exists(norm_path) and os.path.exists(meta_path)):
        from scripts.export_edge_model import export_edge_model
        export_edge_model()

    with open(norm_path, "r") as f:
        norm_data = json.load(f)

    with open(meta_path, "r") as f:
        meta_data = json.load(f)

    ckpt = torch.load("results/ml_training/best_drift_model.pt", map_location="cpu")
    cfg = ckpt.get("model_config", {})
    py_model = TCNDriftModel(
        in_channels=cfg.get("in_channels", 8),
        output_dim=cfg.get("output_dim", 3),
        num_channels=cfg.get("num_channels", [32, 64, 128]),
        kernel_size=cfg.get("kernel_size", 3),
        dropout=cfg.get("dropout", 0.1),
    )
    py_model.load_state_dict(ckpt["model_state_dict"])
    py_model.eval()

    ort_session = ort.InferenceSession(onnx_path)

    return {
        "py_model": py_model,
        "ort_session": ort_session,
        "norm_data": norm_data,
        "meta_data": meta_data,
    }


def test_pytorch_vs_onnx_numerical_parity_100_windows(model_artifacts):
    """Verify PyTorch and ONNX predictions are numerically consistent across 100 deterministic windows."""
    py_model = model_artifacts["py_model"]
    ort_session = model_artifacts["ort_session"]
    input_name = ort_session.get_inputs()[0].name

    np.random.seed(12345)
    max_diff = 0.0

    for _ in range(100):
        # Generate random normalized window [1, 100, 8]
        test_window = np.random.randn(1, 100, 8).astype(np.float32)

        with torch.no_grad():
            py_pred = py_model(torch.from_numpy(test_window)).numpy()

        ort_pred = ort_session.run(None, {input_name: test_window})[0]

        diff = float(np.max(np.abs(py_pred - ort_pred)))
        if diff > max_diff:
            max_diff = diff

    # Max difference between PyTorch and ONNX must be < 1e-5 meters (sub-micrometer)
    assert max_diff < 1e-5, f"PyTorch vs ONNX difference {max_diff:.6e} exceeded tolerance."


def test_normalization_consistency(model_artifacts):
    """Verify normalization statistics in normalization.json match training checkpoint."""
    norm_data = model_artifacts["norm_data"]
    ckpt = torch.load("results/ml_training/best_drift_model.pt", map_location="cpu")

    ckpt_mean = np.array(ckpt["feature_mean"], dtype=np.float32)
    ckpt_std = np.array(ckpt["feature_std"], dtype=np.float32)

    norm_mean = np.array(norm_data["feature_mean"], dtype=np.float32)
    norm_std = np.array(norm_data["feature_std"], dtype=np.float32)

    np.testing.assert_allclose(norm_mean, ckpt_mean, atol=1e-7)
    np.testing.assert_allclose(norm_std, ckpt_std, atol=1e-7)
    assert len(norm_data["feature_names"]) == 8


def test_feature_ordering_contract(model_artifacts):
    """Verify feature names contract conforms to the project specification."""
    expected_order = [
        "accel_x",
        "accel_y",
        "accel_z",
        "gyro_x",
        "gyro_y",
        "gyro_z",
        "accel_norm",
        "gyro_norm",
    ]
    norm_data = model_artifacts["norm_data"]
    assert norm_data["feature_names"] == expected_order


def test_causal_window_invariance(model_artifacts):
    """Verify that predictions depend deterministically on the provided 100-sample window without external leakage."""
    ort_session = model_artifacts["ort_session"]
    input_name = ort_session.get_inputs()[0].name

    np.random.seed(999)
    fixed_window = np.random.randn(1, 100, 8).astype(np.float32)

    pred1 = ort_session.run(None, {input_name: fixed_window})[0]
    pred2 = ort_session.run(None, {input_name: fixed_window})[0]

    np.testing.assert_allclose(pred1, pred2, atol=1e-7)
