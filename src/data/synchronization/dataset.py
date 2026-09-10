"""Synchronized Dataset Abstraction.

Encapsulates IMU, GNSS, and Ground Truth observations alongside time synchronization
mappings and dataset health metadata for consumption by future navigation algorithms.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any

from src.data.observations import IMUObservation, GNSSObservation, GroundTruthObservation
from src.data.synchronization.sync import (
    calculate_timestamp_statistics,
    synchronize_imu_gnss,
)


@dataclass
class SynchronizedDataset:
    """Synchronized Multi-Sensor Dataset container.

    Attributes:
        imu: List of IMU observations.
        gnss: List of GNSS observations.
        ground_truth: Optional list of Ground Truth observations.
        sync_indices: List of (imu_index, matched_gnss_index, dt_sec).
        metadata: Dataset statistics and synchronization parameters.
    """

    imu: List[IMUObservation]
    gnss: List[GNSSObservation]
    ground_truth: Optional[List[GroundTruthObservation]] = None
    sync_indices: List[Tuple[int, Optional[int], float]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        imu: List[IMUObservation],
        gnss: List[GNSSObservation],
        ground_truth: Optional[List[GroundTruthObservation]] = None,
        max_tolerance_sec: float = 0.5,
    ) -> "SynchronizedDataset":
        """Factory method to create and synchronize a dataset.

        Args:
            imu (List[IMUObservation]): Raw IMU observations.
            gnss (List[GNSSObservation]): Raw GNSS observations.
            ground_truth (Optional[List[GroundTruthObservation]]): Reference ground truth observations.
            max_tolerance_sec (float): Matching tolerance threshold in seconds.

        Returns:
            SynchronizedDataset: Initialized and synchronized dataset object.
        """
        sync_results = synchronize_imu_gnss(imu, gnss, max_tolerance_sec=max_tolerance_sec)

        imu_times = [obs.timestamp for obs in imu]
        gnss_times = [obs.timestamp for obs in gnss]
        gt_times = [obs.timestamp for obs in ground_truth] if ground_truth else []

        imu_stats = calculate_timestamp_statistics(imu_times)
        gnss_stats = calculate_timestamp_statistics(gnss_times)
        gt_stats = calculate_timestamp_statistics(gt_times) if ground_truth else {}

        matched_count = sum(1 for _, g_idx, _ in sync_results if g_idx is not None)

        metadata = {
            "imu_stats": imu_stats,
            "gnss_stats": gnss_stats,
            "ground_truth_stats": gt_stats,
            "sync_matched_count": matched_count,
            "sync_match_ratio": matched_count / len(imu) if imu else 0.0,
            "max_tolerance_sec": max_tolerance_sec,
            "status": "PASS" if matched_count > 0 or len(gnss) == 0 else "WARNING",
        }

        return cls(
            imu=imu,
            gnss=gnss,
            ground_truth=ground_truth,
            sync_indices=sync_results,
            metadata=metadata,
        )
