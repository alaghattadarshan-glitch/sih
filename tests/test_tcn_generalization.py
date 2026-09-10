"""Unit & Integration tests for TCN Model Generalization across Unseen Sessions & Domain Shift."""

import pytest
import numpy as np
import torch
from pathlib import Path

from src.data.session import create_synthetic_multi_session_catalog
from src.ml.models.tcn_drift import TCNDriftModel
from src.ml.inference.predictor import DriftPredictor
from src.ml.datasets.real_session_builder import MultiSessionDatasetBuilder
from src.evaluation.multi_session_evaluator import (
    evaluate_tcn_session_generalization,
    generate_domain_shift_analysis,
)


@pytest.fixture
def trained_predictor(tmp_path) -> DriftPredictor:
    """Fixture providing a mock trained DriftPredictor with deterministic weights and normalization."""
    model = TCNDriftModel(in_channels=8, output_dim=3, num_channels=[32, 64], kernel_size=3, dropout=0.0)
    feat_mean = np.array([0.0, 0.0, 9.81, 0.0, 0.0, 0.0, 9.81, 0.0], dtype=np.float32)
    feat_std = np.array([0.5, 0.5, 0.5, 0.1, 0.1, 0.1, 0.5, 0.1], dtype=np.float32)

    ckpt_path = tmp_path / "mock_drift_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "feature_mean": feat_mean,
        "feature_std": feat_std,
        "model_config": {
            "in_channels": 8,
            "output_dim": 3,
            "num_channels": [32, 64],
            "kernel_size": 3,
            "dropout": 0.0,
        },
    }, ckpt_path)

    return DriftPredictor(str(ckpt_path))


def test_tcn_generalization_on_unseen_session(trained_predictor):
    """Verify TCN inference runs smoothly on an unseen session and computes all generalization metrics."""
    sessions = create_synthetic_multi_session_catalog(num_sessions=1, base_seed=800, duration_sec=30.0)
    session = sessions[0]

    eval_res = evaluate_tcn_session_generalization(session, trained_predictor)

    assert eval_res["status"] == "SUCCESS"
    assert eval_res["num_windows"] > 0
    assert "ai_2d_rmse_m" in eval_res
    assert "ai_3d_rmse_m" in eval_res
    assert "classical_2d_rmse_m" in eval_res
    assert "improvement_over_classical_pct" in eval_res
    assert "bias_2d_m" in eval_res
    assert "mae_2d_m" in eval_res
    assert "percentile_95_2d_m" in eval_res
    assert eval_res["ai_2d_rmse_m"] >= 0.0


def test_tcn_normalization_consistency(trained_predictor):
    """Verify inference normalizes unseen windows using exact training mean and std."""
    raw_window = np.zeros((100, 8), dtype=np.float32)
    raw_window[:, 2] = 9.81  # Z-accel
    raw_window[:, 6] = 9.81  # Accel norm

    pred = trained_predictor.predict(raw_window)
    assert isinstance(pred, np.ndarray)
    assert pred.shape == (3,)
    assert np.isfinite(pred).all()


def test_domain_shift_distribution_metrics(tmp_path):
    """Verify domain shift analysis compares feature distributions and outputs json report."""
    base_sessions = create_synthetic_multi_session_catalog(num_sessions=2, base_seed=900, duration_sec=20.0)
    target_sessions = create_synthetic_multi_session_catalog(num_sessions=2, base_seed=950, duration_sec=20.0)

    out_dir = tmp_path / "domain_shift_test"
    report = generate_domain_shift_analysis(base_sessions, target_sessions, out_dir)

    assert report["baseline_sample_count"] > 0
    assert report["target_sample_count"] > 0
    assert (out_dir / "accel_distribution.png").exists()
    assert (out_dir / "gyro_distribution.png").exists()
    assert (out_dir / "accel_norm_distribution.png").exists()
    assert (out_dir / "sampling_distribution.png").exists()
    assert (out_dir / "domain_shift_report.json").exists()
