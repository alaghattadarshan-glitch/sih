"""Sensor Rate, Timestamp Jitter, and Quality Distribution Analyzer.

Computes separate sample rates, timestamp intervals (Δt), jitter percentiles,
and gap statistics for accelerometer, gyroscope, and GNSS observations.
Produces publication-quality diagnostic plots.
"""

import os
from typing import Dict, Any, List, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


class SensorRateAnalyzer:
    """Computes exact empirical sampling rate and jitter metrics from timestamps."""

    @staticmethod
    def analyze_timestamps(
        timestamps: List[float],
        nominal_hz: float = 100.0,
        gap_threshold_sec: float = 0.05,
    ) -> Dict[str, Any]:
        """Compute comprehensive sampling rate and jitter statistics.

        Args:
            timestamps: Monotonic sequence of epoch timestamps in seconds.
            nominal_hz: Expected target frequency.
            gap_threshold_sec: Threshold defining a timestamp gap.

        Returns:
            Dict[str, Any]: Detailed frequency and jitter statistics.
        """
        if len(timestamps) < 2:
            return {
                "sample_count": len(timestamps),
                "duration_sec": 0.0,
                "effective_hz": 0.0,
                "mean_dt_ms": 0.0,
                "median_dt_ms": 0.0,
                "std_dt_ms": 0.0,
                "p95_dt_ms": 0.0,
                "p99_dt_ms": 0.0,
                "min_dt_ms": 0.0,
                "max_dt_ms": 0.0,
                "timestamp_gaps_count": 0,
                "duplicate_timestamps_count": 0,
            }

        ts_arr = np.array(timestamps, dtype=np.float64)
        duration = float(ts_arr[-1] - ts_arr[0])
        dts = np.diff(ts_arr)

        duplicate_count = int(np.sum(dts <= 0.0))
        valid_dts = dts[dts > 0.0]

        if len(valid_dts) == 0:
            return {
                "sample_count": len(timestamps),
                "duration_sec": duration,
                "effective_hz": 0.0,
                "mean_dt_ms": 0.0,
                "median_dt_ms": 0.0,
                "std_dt_ms": 0.0,
                "p95_dt_ms": 0.0,
                "p99_dt_ms": 0.0,
                "min_dt_ms": 0.0,
                "max_dt_ms": 0.0,
                "timestamp_gaps_count": 0,
                "duplicate_timestamps_count": duplicate_count,
            }

        dts_ms = valid_dts * 1000.0
        gaps_count = int(np.sum(valid_dts > gap_threshold_sec))
        effective_hz = (len(timestamps) - 1) / duration if duration > 0 else nominal_hz

        return {
            "sample_count": len(timestamps),
            "duration_sec": duration,
            "effective_hz": float(effective_hz),
            "mean_dt_ms": float(np.mean(dts_ms)),
            "median_dt_ms": float(np.median(dts_ms)),
            "std_dt_ms": float(np.std(dts_ms)),
            "p95_dt_ms": float(np.percentile(dts_ms, 95)),
            "p99_dt_ms": float(np.percentile(dts_ms, 99)),
            "min_dt_ms": float(np.min(dts_ms)),
            "max_dt_ms": float(np.max(dts_ms)),
            "timestamp_gaps_count": gaps_count,
            "duplicate_timestamps_count": duplicate_count,
        }

    @classmethod
    def analyze_session(
        cls,
        imu_list: List[Any],
        gnss_list: Optional[List[Any]] = None,
        output_dir: Optional[str | Any] = None,
    ) -> Dict[str, Any]:
        """Analyze full multi-sensor session and generate diagnostic plots."""
        imu_times = [obs.timestamp for obs in imu_list] if imu_list else []
        gnss_times = [obs.timestamp for obs in gnss_list] if gnss_list else []

        imu_stats = cls.analyze_timestamps(imu_times, nominal_hz=100.0, gap_threshold_sec=0.05)
        gnss_stats = cls.analyze_timestamps(gnss_times, nominal_hz=1.0, gap_threshold_sec=2.0) if gnss_times else {}

        if output_dir:
            cls.generate_rate_plots(imu_times, gnss_times, str(output_dir))

        return {
            "sensors": {
                "imu": imu_stats,
                "gnss": gnss_stats,
            },
            "overall_healthy": bool(imu_stats.get("effective_hz", 0.0) > 80.0),
        }

    @staticmethod
    def generate_rate_plots(
        imu_timestamps: List[float],
        gnss_timestamps: List[float],
        output_dir: str,
    ):
        """Generate diagnostic sensor rate, jitter, and gap plots."""
        os.makedirs(output_dir, exist_ok=True)

        imu_ts = np.array(imu_timestamps, dtype=np.float64)
        imu_dts_ms = np.diff(imu_ts) * 1000.0

        # 1. sensor_rates.png
        # Compute rolling 1-second sample rate
        window_size = 1.0
        t_start = imu_ts[0]
        t_end = imu_ts[-1]
        t_eval = np.arange(t_start, t_end, 0.5)
        rolling_hz = []
        for t in t_eval:
            count = np.sum((imu_ts >= t) & (imu_ts < t + window_size))
            rolling_hz.append(count / window_size)

        plt.figure(figsize=(8, 4))
        plt.plot(t_eval, rolling_hz, color="teal", lw=2, label="IMU Effective Rate")
        plt.axhline(100.0, color="crimson", linestyle="--", label="Target Rate (100 Hz)")
        plt.title("Empirical IMU Sampling Frequency vs Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Sample Rate (Hz)")
        plt.ylim(80, 120)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "sensor_rates.png"), dpi=150)
        plt.close()

        # 2. timestamp_jitter.png
        plt.figure(figsize=(7, 4))
        plt.hist(imu_dts_ms, bins=50, color="royalblue", edgecolor="black", alpha=0.8)
        plt.axvline(10.0, color="red", linestyle="--", label="Nominal Δt (10 ms)")
        plt.title("IMU Inter-Sample Interval (Δt) Distribution")
        plt.xlabel("Δt (ms)")
        plt.ylabel("Sample Count")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "timestamp_jitter.png"), dpi=150)
        plt.close()

        # 3. sensor_gaps.png
        plt.figure(figsize=(8, 4))
        plt.plot(imu_ts[1:], imu_dts_ms, color="darkorange", lw=1, alpha=0.8)
        plt.axhline(50.0, color="red", linestyle="--", label="Gap Threshold (50 ms)")
        plt.title("Sensor Interval Gaps & Outliers")
        plt.xlabel("Time (s)")
        plt.ylabel("Interval Δt (ms)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "sensor_gaps.png"), dpi=150)
        plt.close()
