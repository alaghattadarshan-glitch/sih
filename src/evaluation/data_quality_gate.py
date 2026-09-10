"""Data Quality Gate for Android Real-World and Synthetic Sensor Logs.

Audits multi-sensor datasets against rigorous criteria:
- Monotonic timestamps
- Duplicate timestamps
- Sensor gaps and missing data
- NaNs and Infs
- GNSS validity (bounds, accuracy, status)
- Unrealistic acceleration and angular velocity bounds
- Trajectory and outage duration sufficiency

Assigns PASS / WARN / FAIL status to each criterion and generates
a structured JSON quality report.
"""

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

from src.data.observations import IMUObservation, GNSSObservation, GroundTruthObservation


@dataclass
class QualityCriterionResult:
    """Individual quality criterion evaluation result."""
    name: str
    status: str  # "PASS", "WARN", "FAIL"
    value: Any
    threshold: str
    message: str


@dataclass
class DataQualityReport:
    """Aggregated data quality gate report."""
    dataset_name: str
    overall_status: str  # "PASS", "WARN", "FAIL"
    criteria: Dict[str, QualityCriterionResult] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "overall_status": self.overall_status,
            "criteria": {
                name: {
                    "status": crit.status,
                    "value": crit.value,
                    "threshold": crit.threshold,
                    "message": crit.message,
                }
                for name, crit in self.criteria.items()
            },
            "summary": self.summary,
        }

    def save_json(self, output_path: str | Path) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


class DataQualityGate:
    """Rigorous data quality evaluation gate."""

    def __init__(
        self,
        max_gap_seconds: float = 0.1,
        min_accel_norm: float = 5.0,
        max_accel_norm: float = 25.0,
        max_gyro_norm: float = 15.0,
        min_trajectory_duration: float = 5.0,
        min_outage_duration: float = 5.0,
    ):
        self.max_gap_seconds = max_gap_seconds
        self.min_accel_norm = min_accel_norm
        self.max_accel_norm = max_accel_norm
        self.max_gyro_norm = max_gyro_norm
        self.min_trajectory_duration = min_trajectory_duration
        self.min_outage_duration = min_outage_duration

    def evaluate(
        self,
        imu_list: List[IMUObservation],
        gnss_list: Optional[List[GNSSObservation]] = None,
        dataset_name: str = "real_android_session",
        simulated_outage_duration: Optional[float] = None,
    ) -> DataQualityReport:
        """Run all data quality audits on provided observation streams."""
        criteria: Dict[str, QualityCriterionResult] = {}
        statuses: List[str] = []

        # 1. IMU Monotonicity
        if not imu_list:
            criteria["imu_monotonicity"] = QualityCriterionResult(
                name="imu_monotonicity",
                status="FAIL",
                value=0,
                threshold="0 non-monotonic steps",
                message="IMU observation list is empty.",
            )
            statuses.append("FAIL")
        else:
            t_imu = np.array([obs.timestamp for obs in imu_list], dtype=np.float64)
            dts_imu = np.diff(t_imu)
            non_monotonic_imu = int(np.sum(dts_imu < 0.0))
            crit_status = "PASS" if non_monotonic_imu == 0 else "FAIL"
            criteria["imu_monotonicity"] = QualityCriterionResult(
                name="imu_monotonicity",
                status=crit_status,
                value=non_monotonic_imu,
                threshold="== 0",
                message=f"Found {non_monotonic_imu} non-monotonic IMU timestamp transitions.",
            )
            statuses.append(crit_status)

        # 2. Duplicate Timestamps
        if imu_list:
            duplicates_imu = int(np.sum(dts_imu == 0.0))
            crit_status = "PASS" if duplicates_imu == 0 else "WARN"
            criteria["imu_duplicates"] = QualityCriterionResult(
                name="imu_duplicates",
                status=crit_status,
                value=duplicates_imu,
                threshold="== 0",
                message=f"Found {duplicates_imu} duplicate IMU timestamps.",
            )
            statuses.append(crit_status)

        # 3. IMU Sensor Gaps
        if imu_list and len(dts_imu) > 0:
            large_gaps = int(np.sum(dts_imu > self.max_gap_seconds))
            max_dt = float(np.max(dts_imu))
            if large_gaps == 0:
                crit_status = "PASS"
            elif large_gaps <= 2 and max_dt < self.max_gap_seconds * 2:
                crit_status = "WARN"
            else:
                crit_status = "FAIL"

            criteria["imu_sensor_gaps"] = QualityCriterionResult(
                name="imu_sensor_gaps",
                status=crit_status,
                value={"gaps_count": large_gaps, "max_gap_sec": max_dt},
                threshold=f"<= {self.max_gap_seconds} s",
                message=f"Found {large_gaps} IMU intervals exceeding {self.max_gap_seconds}s (max: {max_dt:.4f}s).",
            )
            statuses.append(crit_status)

        # 4. NaNs and Infs in IMU
        if imu_list:
            ax = np.array([obs.accelerometer_x for obs in imu_list])
            ay = np.array([obs.accelerometer_y for obs in imu_list])
            az = np.array([obs.accelerometer_z for obs in imu_list])
            gx = np.array([obs.gyroscope_x for obs in imu_list])
            gy = np.array([obs.gyroscope_y for obs in imu_list])
            gz = np.array([obs.gyroscope_z for obs in imu_list])

            nan_inf_count = int(
                np.sum(np.isnan(ax) | np.isinf(ax))
                + np.sum(np.isnan(ay) | np.isinf(ay))
                + np.sum(np.isnan(az) | np.isinf(az))
                + np.sum(np.isnan(gx) | np.isinf(gx))
                + np.sum(np.isnan(gy) | np.isinf(gy))
                + np.sum(np.isnan(gz) | np.isinf(gz))
            )
            crit_status = "PASS" if nan_inf_count == 0 else "FAIL"
            criteria["imu_nan_inf"] = QualityCriterionResult(
                name="imu_nan_inf",
                status=crit_status,
                value=nan_inf_count,
                threshold="== 0",
                message=f"Found {nan_inf_count} NaN/Inf values across IMU channels.",
            )
            statuses.append(crit_status)

            # 5. Unrealistic Acceleration
            accel_norm = np.sqrt(ax**2 + ay**2 + az**2)
            min_norm = float(np.min(accel_norm))
            max_norm = float(np.max(accel_norm))
            mean_norm = float(np.mean(accel_norm))

            if min_norm < 1.0 or max_norm > 50.0 or (mean_norm < self.min_accel_norm or mean_norm > self.max_accel_norm):
                crit_status = "FAIL"
            elif min_norm < self.min_accel_norm * 0.7 or max_norm > self.max_accel_norm * 1.5:
                crit_status = "WARN"
            else:
                crit_status = "PASS"

            criteria["accel_validity"] = QualityCriterionResult(
                name="accel_validity",
                status=crit_status,
                value={"mean_norm": mean_norm, "min_norm": min_norm, "max_norm": max_norm},
                threshold=f"norm in [{self.min_accel_norm}, {self.max_accel_norm}] m/s²",
                message=f"Accel norm mean={mean_norm:.2f}, min={min_norm:.2f}, max={max_norm:.2f} m/s².",
            )
            statuses.append(crit_status)

            # 6. Unrealistic Gyroscope
            gyro_norm = np.sqrt(gx**2 + gy**2 + gz**2)
            max_g = float(np.max(gyro_norm))
            mean_g = float(np.mean(gyro_norm))

            if max_g > self.max_gyro_norm:
                crit_status = "FAIL"
            elif max_g > self.max_gyro_norm * 0.5:
                crit_status = "WARN"
            else:
                crit_status = "PASS"

            criteria["gyro_validity"] = QualityCriterionResult(
                name="gyro_validity",
                status=crit_status,
                value={"mean_norm": mean_g, "max_norm": max_g},
                threshold=f"max norm <= {self.max_gyro_norm} rad/s",
                message=f"Gyro norm mean={mean_g:.4f}, max={max_g:.4f} rad/s.",
            )
            statuses.append(crit_status)

        # 7. GNSS Validity
        if gnss_list and len(gnss_list) > 0:
            lats = np.array([g.latitude for g in gnss_list])
            lons = np.array([g.longitude for g in gnss_list])
            alts = np.array([g.altitude for g in gnss_list])
            accs = np.array([g.horizontal_accuracy for g in gnss_list])

            invalid_lat = np.sum((lats < -90.0) | (lats > 90.0) | np.isnan(lats))
            invalid_lon = np.sum((lons < -180.0) | (lons > 180.0) | np.isnan(lons))
            invalid_acc = np.sum((accs <= 0.0) | np.isnan(accs) | (accs > 500.0))

            if invalid_lat > 0 or invalid_lon > 0 or invalid_acc > len(gnss_list) * 0.1:
                crit_status = "FAIL"
            elif invalid_acc > 0:
                crit_status = "WARN"
            else:
                crit_status = "PASS"

            criteria["gnss_validity"] = QualityCriterionResult(
                name="gnss_validity",
                status=crit_status,
                value={
                    "sample_count": len(gnss_list),
                    "invalid_lat_count": int(invalid_lat),
                    "invalid_lon_count": int(invalid_lon),
                    "invalid_acc_count": int(invalid_acc),
                    "mean_accuracy_m": float(np.mean(accs)),
                },
                threshold="valid coordinates and horizontal_accuracy > 0",
                message=f"GNSS samples: {len(gnss_list)}, mean accuracy: {float(np.mean(accs)):.2f}m.",
            )
            statuses.append(crit_status)
        else:
            criteria["gnss_validity"] = QualityCriterionResult(
                name="gnss_validity",
                status="WARN",
                value=0,
                threshold=">= 1 sample",
                message="No GNSS observations present.",
            )
            statuses.append("WARN")

        # 8. Trajectory Duration Sufficiency
        duration = float(t_imu[-1] - t_imu[0]) if imu_list and len(imu_list) > 1 else 0.0
        if duration < self.min_trajectory_duration:
            crit_status = "FAIL"
        else:
            crit_status = "PASS"

        criteria["trajectory_duration"] = QualityCriterionResult(
            name="trajectory_duration",
            status=crit_status,
            value=duration,
            threshold=f">= {self.min_trajectory_duration} s",
            message=f"Recorded trajectory duration is {duration:.2f} s.",
        )
        statuses.append(crit_status)

        # 9. Outage Duration Sufficiency
        if simulated_outage_duration is not None:
            if simulated_outage_duration < self.min_outage_duration:
                crit_status = "WARN"
            else:
                crit_status = "PASS"
            criteria["outage_duration"] = QualityCriterionResult(
                name="outage_duration",
                status=crit_status,
                value=simulated_outage_duration,
                threshold=f">= {self.min_outage_duration} s",
                message=f"Outage duration evaluated is {simulated_outage_duration:.2f} s.",
            )
            statuses.append(crit_status)

        # Overall Status
        if "FAIL" in statuses:
            overall_status = "FAIL"
        elif "WARN" in statuses:
            overall_status = "WARN"
        else:
            overall_status = "PASS"

        summary = {
            "total_criteria": len(criteria),
            "pass_count": sum(1 for c in criteria.values() if c.status == "PASS"),
            "warn_count": sum(1 for c in criteria.values() if c.status == "WARN"),
            "fail_count": sum(1 for c in criteria.values() if c.status == "FAIL"),
            "trajectory_duration_sec": duration,
            "imu_sample_count": len(imu_list),
            "gnss_sample_count": len(gnss_list) if gnss_list else 0,
        }

        return DataQualityReport(
            dataset_name=dataset_name,
            overall_status=overall_status,
            criteria=criteria,
            summary=summary,
        )
