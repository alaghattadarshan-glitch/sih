"""Unit tests for ML Neural Network Architecture, Classical Baseline, Trainer, and Inference."""

import os
import pytest
import numpy as np
import torch

from src.ml.models.tcn_drift import TCNDriftModel
from src.ml.baseline import compute_classical_imu_displacement, compute_dataset_classical_baseline
from src.ml.training.trainer import ModelTrainer, set_seed
from src.ml.inference import DriftPredictor
from src.ml.evaluation.metrics import compute_regression_metrics, evaluate_comparative_performance
from src.ml.datasets.imu_dataset import IMUWindowDataset


def test_tcn_model_input_output_shapes():
    """Verify TCNDriftModel handles input [batch, 100, 8] and outputs [batch, 3]."""
    model = TCNDriftModel(in_channels=8, output_dim=3)
    x = torch.randn(16, 100, 8)
    out = model(x)

    assert out.shape == (16, 3)
    assert not torch.isnan(out).any()


def test_tcn_model_transpose_support():
    """Verify TCNDriftModel handles channel-first input [batch, 8, 100]."""
    model = TCNDriftModel(in_channels=8, output_dim=3)
    x_channel_first = torch.randn(8, 8, 100)
    out = model(x_channel_first)

    assert out.shape == (8, 3)


def test_tcn_model_config_export():
    """Verify get_config exposes architecture parameters."""
    model = TCNDriftModel(in_channels=8, output_dim=3, num_channels=[16, 32], dropout=0.2)
    cfg = model.get_config()

    assert cfg["model_type"] == "tcn"
    assert cfg["in_channels"] == 8
    assert cfg["output_dim"] == 3
    assert cfg["num_channels"] == [16, 32]
    assert cfg["dropout"] == 0.2


def test_classical_imu_baseline_calculation():
    """Verify classical IMU displacement integration for stationary and motion windows."""
    # Stationary window (accel_z = 9.81)
    stationary_window = np.zeros((100, 8), dtype=np.float32)
    stationary_window[:, 2] = 9.81

    disp_stationary = compute_classical_imu_displacement(stationary_window, dt=0.01)
    assert disp_stationary.shape == (3,)
    assert np.allclose(disp_stationary, 0.0, atol=1e-5)

    # Constant forward accel_x = 1.0 m/s^2 for 1.0s window -> disp = 0.5 * 1.0 * (1.0)^2 = 0.5m
    motion_window = np.zeros((100, 8), dtype=np.float32)
    motion_window[:, 0] = 1.0
    motion_window[:, 2] = 9.81

    disp_motion = compute_classical_imu_displacement(motion_window, dt=0.01)
    assert np.isclose(disp_motion[0], 0.5, atol=0.02)


def test_model_trainer_step_and_early_stopping():
    """Verify ModelTrainer executes training step and reduces training loss."""
    set_seed(42)
    model = TCNDriftModel(in_channels=8, output_dim=3, num_channels=[16, 32])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    trainer = ModelTrainer(model=model, optimizer=optimizer, patience=5)

    # Dummy dataset
    x = torch.randn(64, 100, 8)
    y = torch.randn(64, 3)
    dataset = torch.utils.data.TensorDataset(x, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=16)

    initial_loss = trainer.validate_epoch(loader)
    history = trainer.fit(loader, loader, epochs=10)

    final_loss = trainer.validate_epoch(loader)
    assert final_loss < initial_loss
    assert len(history["train_loss"]) > 0


def test_checkpoint_save_and_inference_predictor(tmp_path):
    """Verify saving checkpoint to disk and loading it via DriftPredictor."""
    set_seed(42)
    model = TCNDriftModel(in_channels=8, output_dim=3)
    ckpt_path = os.path.join(tmp_path, "test_model.pt")

    feature_mean = np.full(8, 0.5, dtype=np.float32)
    feature_std = np.full(8, 2.0, dtype=np.float32)

    torch.save({
        "model_config": model.get_config(),
        "model_state_dict": model.state_dict(),
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
    }, ckpt_path)

    predictor = DriftPredictor(ckpt_path)
    sample_window = np.random.randn(100, 8).astype(np.float32)
    pred = predictor.predict_window(sample_window)

    assert pred.shape == (3,)
    assert not np.isnan(pred).any()


def test_normalization_anti_leakage_protection():
    """Verify that feature normalization parameters are computed strictly from training split."""
    train_features = np.ones((50, 100, 8), dtype=np.float32) * 5.0  # mean = 5.0
    val_features = np.ones((20, 100, 8), dtype=np.float32) * 100.0  # distinct mean

    # Compute mean/std strictly on training set
    train_mean = np.mean(train_features, axis=(0, 1))
    train_std = np.std(train_features, axis=(0, 1))
    train_std[train_std < 1e-8] = 1.0

    assert np.allclose(train_mean, 5.0)

    # Instantiate datasets for Train and Val splits using TRAIN statistics
    train_ds = IMUWindowDataset(
        features=train_features,
        targets=np.zeros((50, 3)),
        time_ranges=[(0.0, 1.0)] * 50,
        feature_names=["f"] * 8,
        target_names=["t"] * 3,
        trajectory_id="train_1",
        feature_mean=train_mean,
        feature_std=train_std,
    )

    val_ds = IMUWindowDataset(
        features=val_features,
        targets=np.zeros((20, 3)),
        time_ranges=[(0.0, 1.0)] * 20,
        feature_names=["f"] * 8,
        target_names=["t"] * 3,
        trajectory_id="val_1",
        feature_mean=train_mean,  # Applied UNCHANGED
        feature_std=train_std,
    )

    # Prove train_mean in val_ds matches train_mean exactly, and wasn't contaminated by val_features
    assert np.array_equal(val_ds.feature_mean, train_ds.feature_mean)
    assert not np.allclose(val_ds.feature_mean, 100.0)


def test_evaluation_metrics():
    """Verify calculation of MAE, RMSE, and Horizontal RMSE."""
    y_true = np.array([[1.0, 2.0, 0.0], [3.0, 4.0, 0.0]], dtype=np.float32)
    y_pred = np.array([[1.0, 2.0, 0.0], [3.0, 4.0, 0.0]], dtype=np.float32)

    perfect_metrics = compute_regression_metrics(y_true, y_pred)
    assert perfect_metrics["rmse_horizontal"] == 0.0
    assert perfect_metrics["mae_east"] == 0.0

    y_err = np.array([[2.0, 2.0, 0.0], [4.0, 4.0, 0.0]], dtype=np.float32)
    err_metrics = compute_regression_metrics(y_true, y_err)
    # diff East = [1, 1] -> mean sq_err = 1.0 -> rmse_horiz = 1.0
    assert np.isclose(err_metrics["rmse_horizontal"], 1.0, atol=1e-3)


def test_comparative_performance_evaluation():
    """Verify evaluation breakdown by classical vs AI and GNSS status stratification."""
    y_true = np.array([[1.0, 1.0, 0.0], [2.0, 2.0, 0.0]], dtype=np.float32)
    y_classical = np.array([[2.0, 2.0, 0.0], [4.0, 4.0, 0.0]], dtype=np.float32)
    y_ai = np.array([[1.1, 1.1, 0.0], [2.1, 2.1, 0.0]], dtype=np.float32)
    statuses = ["GOOD", "OUTAGE"]

    report = evaluate_comparative_performance(y_true, y_classical, y_ai, gnss_statuses=statuses)
    assert report["ai_outperforms_baseline"] is True
    assert "stratified_by_gnss_status" in report
    assert "GOOD" in report["stratified_by_gnss_status"]
    assert "OUTAGE" in report["stratified_by_gnss_status"]
