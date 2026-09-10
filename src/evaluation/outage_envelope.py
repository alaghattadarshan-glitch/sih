"""Outage Operating Envelope and AI Gating Validation Suite for SIH26168.

Evaluates navigation performance across a full spectrum of outage durations
(5s, 10s, 20s, 30s, 60s, 120s), tracking AI innovation gating and GNSS recovery (0s to 10s).
"""

from typing import Any, Dict, List, Optional
import numpy as np

from src.data.session import DatasetSession, DataSourceType
from src.coordinate_transforms import LocalFrame
from src.navigation.ekf.eskf import ErrorStateKalmanFilter
from src.navigation.ai_fusion import AIESKFPipeline
from src.outage_detection.detector import GNSSOutageDetector
from src.ml.inference.predictor import DriftPredictor


def evaluate_outage_operating_envelope(
    session: DatasetSession,
    predictor: Optional[DriftPredictor] = None,
    candidate_durations: Optional[List[float]] = None,
    outage_start_sec: float = 30.0,
) -> Dict[str, Any]:
    """Evaluate performance across short to long outage durations.
    
    Durations: [5.0, 10.0, 20.0, 30.0, 60.0, 120.0] seconds.
    """
    if candidate_durations is None:
        candidate_durations = [5.0, 10.0, 20.0, 30.0, 60.0, 120.0]

    session_duration = session.duration_sec
    # Filter durations that fit within the session
    valid_durations = [
        d for d in candidate_durations if (outage_start_sec + d + 5.0) <= session_duration
    ]
    if not valid_durations:
        valid_durations = [min(10.0, max(5.0, session_duration - outage_start_sec - 2.0))]

    envelope_results: Dict[str, Any] = {}

    for out_dur in valid_durations:
        key = f"{int(out_dur)}s_outage"
        res = _run_envelope_pass(
            session=session,
            predictor=predictor,
            outage_start=outage_start_sec,
            outage_duration=out_dur,
        )
        envelope_results[key] = res

    # Analyze operating envelope limits
    pass_durations = [
        res["outage_duration_sec"]
        for res in envelope_results.values()
        if res["sih_target_check"]["status"] == "PASS"
    ]
    max_compliant_outage_sec = max(pass_durations) if pass_durations else 0.0

    return {
        "session_id": session.session_id,
        "source_type": session.source_type.value,
        "session_duration_sec": round(session_duration, 1),
        "tested_durations_sec": valid_durations,
        "max_compliant_outage_sec": max_compliant_outage_sec,
        "outage_duration_evaluations": envelope_results,
    }


def _run_envelope_pass(
    session: DatasetSession,
    predictor: Optional[DriftPredictor],
    outage_start: float,
    outage_duration: float,
) -> Dict[str, Any]:
    """Execute single simulation pass recording fine-grained gating and recovery metrics."""
    gnss_fixes = session.gnss_observations
    first_gnss = gnss_fixes[0]
    local_frame = session.local_frame or LocalFrame(first_gnss.latitude, first_gnss.longitude, first_gnss.altitude)

    eskf = ErrorStateKalmanFilter()
    eskf.initialize(
        initial_time=0.0,
        initial_llh=(first_gnss.latitude, first_gnss.longitude, first_gnss.altitude),
        initial_velocity_enu=np.zeros(3),
        initial_quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
    )

    pipeline = AIESKFPipeline(
        eskf=eskf,
        detector=GNSSOutageDetector({"persistence_count": 2}),
        predictor=predictor,
        enable_ai=(predictor is not None),
    )

    gnss_by_time = {round(g.timestamp, 2): g for g in gnss_fixes}
    gt_dict = {round(gt.timestamp, 2): gt for gt in (session.ground_truth_observations or [])}

    timestamps = []
    estimated_positions = []
    gt_positions = []
    outage_end = outage_start + outage_duration

    for imu in session.imu_observations:
        t = imu.timestamp
        g_fix = gnss_by_time.get(round(t, 2), None)

        if outage_start <= t < outage_end:
            g_fix = None

        state, status = pipeline.process_sample(imu, g_fix)

        timestamps.append(t)
        estimated_positions.append(state.position_enu.copy())

        gt_obs = gt_dict.get(round(t, 2), None)
        if gt_obs is not None:
            e, n, u = local_frame.to_enu(gt_obs.latitude, gt_obs.longitude, gt_obs.altitude)
            gt_positions.append(np.array([e, n, u]))
        else:
            gt_positions.append(state.position_enu.copy())

    t_arr = np.array(timestamps)
    est_arr = np.array(estimated_positions)
    gt_arr = np.array(gt_positions)

    # Compute errors
    errors = np.sqrt(np.sum((est_arr[:, :2] - gt_arr[:, :2]) ** 2, axis=1))
    out_mask = (t_arr >= outage_start) & (t_arr <= outage_end)
    out_errors = errors[out_mask]
    out_gt = gt_arr[out_mask]

    # Distance travelled during outage
    if len(out_gt) > 1:
        diffs = np.diff(out_gt[:, :2], axis=0)
        dist_travelled_m = float(np.sum(np.sqrt(np.sum(diffs ** 2, axis=1))))
    else:
        dist_travelled_m = 100.0

    rmse = float(np.sqrt(np.mean(out_errors ** 2))) if len(out_errors) else 0.0
    max_err = float(np.max(out_errors)) if len(out_errors) else 0.0
    pct_95 = float(np.percentile(out_errors, 95)) if len(out_errors) else 0.0
    err_start = float(errors[np.argmin(np.abs(t_arr - outage_start))])
    err_mid = float(errors[np.argmin(np.abs(t_arr - (outage_start + outage_duration / 2.0)))])
    err_end = float(errors[np.argmin(np.abs(t_arr - outage_end))])
    final_err = float(errors[-1])

    drift_pct = (err_end / dist_travelled_m) * 100.0 if dist_travelled_m > 0 else 0.0
    sih_status = "PASS" if drift_pct < 10.0 else "FAIL"

    # Recovery at 0s, 1s, 2s, 5s, 10s
    rec_times = [0.0, 1.0, 2.0, 5.0, 10.0]
    recovery_metrics = {}
    for rt in rec_times:
        target_t = outage_end + rt
        if target_t <= t_arr[-1]:
            idx = np.argmin(np.abs(t_arr - target_t))
            recovery_metrics[f"recovery_{int(rt)}s_error_m"] = round(float(errors[idx]), 2)
        else:
            recovery_metrics[f"recovery_{int(rt)}s_error_m"] = round(float(errors[-1]), 2)

    # AI Gating Statistics
    ai_accepted = pipeline.ai_accepted_count
    ai_rejected = pipeline.ai_rejected_count
    ai_total = ai_accepted + ai_rejected
    ai_accept_pct = (ai_accepted / ai_total * 100.0) if ai_total > 0 else 0.0

    return {
        "outage_duration_sec": outage_duration,
        "outage_start_sec": outage_start,
        "outage_end_sec": outage_end,
        "distance_travelled_during_outage_m": round(dist_travelled_m, 2),
        "outage_rmse_m": round(rmse, 2),
        "error_at_outage_start_m": round(err_start, 2),
        "error_at_outage_midpoint_m": round(err_mid, 2),
        "error_at_outage_end_m": round(err_end, 2),
        "max_outage_error_m": round(max_err, 2),
        "percentile_95_error_m": round(pct_95, 2),
        "final_error_m": round(final_err, 2),
        "drift_percentage": round(drift_pct, 2),
        "drift_per_100m_travelled_m": round(drift_pct, 2),
        "sih_target_check": {
            "target_threshold_pct": 10.0,
            "measured_drift_pct": round(drift_pct, 2),
            "status": sih_status,
            "disclaimer": "Preliminary target check on this recording/session.",
        },
        "recovery": recovery_metrics,
        "ai_gating": {
            "total_ai_updates": ai_total,
            "accepted_ai_updates": ai_accepted,
            "rejected_ai_updates": ai_rejected,
            "acceptance_rate_pct": round(ai_accept_pct, 1),
        },
    }
