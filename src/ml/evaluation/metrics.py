"""Evaluation Metrics & Comparative Analysis Module.

Computes 3D regression metrics (MAE East/North/Up, RMSE East/North/Up, Horizontal RMSE)
and performs stratified performance comparisons between Ground Truth, Classical IMU Baseline,
and AI Neural Network Predictions across GNSS operational statuses.
"""

from typing import Dict, Any, List, Optional
import numpy as np


def compute_regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, prefix: str = ""
) -> Dict[str, float]:
    """Compute 3D displacement regression metrics.

    Args:
        y_true (np.ndarray): Ground truth target array [N, 3] (East, North, Up).
        y_pred (np.ndarray): Predicted target array [N, 3] (East, North, Up).
        prefix (str): Optional key prefix.

    Returns:
        Dict[str, float]: Dictionary of computed MAE and RMSE metrics.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")

    diff = y_true - y_pred  # [N, 3]
    abs_err = np.abs(diff)
    sq_err = diff**2

    mae_east = float(np.mean(abs_err[:, 0]))
    mae_north = float(np.mean(abs_err[:, 1]))
    mae_up = float(np.mean(abs_err[:, 2]))

    rmse_east = float(np.sqrt(np.mean(sq_err[:, 0])))
    rmse_north = float(np.sqrt(np.mean(sq_err[:, 1])))
    rmse_up = float(np.sqrt(np.mean(sq_err[:, 2])))

    # Horizontal RMSE: sqrt(mean((err_E^2 + err_N^2)))
    horizontal_sq_err = sq_err[:, 0] + sq_err[:, 1]
    rmse_horizontal = float(np.sqrt(np.mean(horizontal_sq_err)))

    # Overall 3D RMSE
    rmse_3d = float(np.sqrt(np.mean(np.sum(sq_err, axis=1))))

    p = f"{prefix}_" if prefix else ""
    return {
        f"{p}mae_east": mae_east,
        f"{p}mae_north": mae_north,
        f"{p}mae_up": mae_up,
        f"{p}rmse_east": rmse_east,
        f"{p}rmse_north": rmse_north,
        f"{p}rmse_up": rmse_up,
        f"{p}rmse_horizontal": rmse_horizontal,
        f"{p}rmse_3d": rmse_3d,
    }


def evaluate_comparative_performance(
    y_true: np.ndarray,
    y_classical: np.ndarray,
    y_ai: np.ndarray,
    gnss_statuses: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Perform comprehensive comparative evaluation between Classical IMU Baseline and AI Model.

    Args:
        y_true (np.ndarray): Ground truth target array [N, 3].
        y_classical (np.ndarray): Classical IMU baseline predictions [N, 3].
        y_ai (np.ndarray): AI model predictions [N, 3].
        gnss_statuses (Optional[List[str]]): List of GNSS status strings for window stratification.

    Returns:
        Dict[str, Any]: Nested metrics dictionary.
    """
    classical_metrics = compute_regression_metrics(y_true, y_classical, prefix="classical")
    ai_metrics = compute_regression_metrics(y_true, y_ai, prefix="ai")

    # Relative improvement in horizontal RMSE (%)
    c_horiz = classical_metrics["classical_rmse_horizontal"]
    ai_horiz = ai_metrics["ai_rmse_horizontal"]
    horiz_improvement_pct = ((c_horiz - ai_horiz) / max(1e-8, c_horiz)) * 100.0

    report = {
        "classical_baseline": classical_metrics,
        "ai_model": ai_metrics,
        "horizontal_rmse_improvement_pct": float(horiz_improvement_pct),
        "ai_outperforms_baseline": bool(ai_horiz < c_horiz),
    }

    # Stratified breakdown by GNSS status if provided
    if gnss_statuses is not None and len(gnss_statuses) == len(y_true):
        statuses_array = np.array(gnss_statuses)
        stratified = {}
        unique_statuses = sorted(list(set(gnss_statuses)))

        for st in unique_statuses:
            mask = statuses_array == st
            if np.sum(mask) > 0:
                sub_gt = y_true[mask]
                sub_c = y_classical[mask]
                sub_ai = y_ai[mask]
                stratified[st] = {
                    "count": int(np.sum(mask)),
                    "classical": compute_regression_metrics(sub_gt, sub_c),
                    "ai": compute_regression_metrics(sub_gt, sub_ai),
                }
        report["stratified_by_gnss_status"] = stratified

    return report
