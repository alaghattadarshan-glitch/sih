"""Base abstract dataset adapter for real-world and benchmark dataset ingestion.

Provides canonical unit normalization, axis transformations, validation,
and conversion into strongly-typed project observations.
"""

from abc import ABC, abstractmethod
import math
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd

from src.data.observations import IMUObservation, GNSSObservation, GroundTruthObservation
from src.data.synchronization.dataset import SynchronizedDataset
from src.data.adapters.schema import (
    DatasetConfig,
    TimestampUnit,
    AccelUnit,
    GyroUnit,
    AxisTransformConfig,
)


class DatasetAdapter(ABC):
    """Abstract base adapter converting raw dataset formats into canonical observations."""

    def __init__(self, config: DatasetConfig):
        """Initialize adapter with dataset configuration.

        Args:
            config (DatasetConfig): Ingestion and mapping configuration.
        """
        self.config = config

    @abstractmethod
    def load_imu(self) -> List[IMUObservation]:
        """Load and normalize IMU observations into canonical units and body frame.

        Returns:
            List[IMUObservation]: Canonical IMU observation samples.
        """
        pass

    @abstractmethod
    def load_gnss(self) -> List[GNSSObservation]:
        """Load and normalize GNSS satellite positioning observations.

        Returns:
            List[GNSSObservation]: Canonical GNSS observation samples.
        """
        pass

    @abstractmethod
    def load_ground_truth(self) -> Optional[List[GroundTruthObservation]]:
        """Load reference ground truth trajectory if available.

        Returns:
            Optional[List[GroundTruthObservation]]: Ground truth samples or None.
        """
        pass

    def load_dataset(self) -> SynchronizedDataset:
        """Load, normalize, and time-synchronize all available sensor modalities.

        Returns:
            SynchronizedDataset: Synchronized multi-sensor dataset container.
        """
        imu_list = self.load_imu()
        gnss_list = self.load_gnss()
        gt_list = self.load_ground_truth()

        sync_dataset = SynchronizedDataset.create(
            imu=imu_list,
            gnss=gnss_list,
            ground_truth=gt_list,
            max_tolerance_sec=self.config.max_sync_tolerance_sec,
        )
        sync_dataset.metadata.update(self.get_metadata())
        return sync_dataset

    def get_metadata(self) -> Dict[str, Any]:
        """Return dataset metadata summary.

        Returns:
            Dict[str, Any]: Metadata dictionary.
        """
        return {
            "dataset_name": self.config.dataset_name,
            "adapter_class": self.__class__.__name__,
            "imu_file": self.config.imu_file_path,
            "gnss_file": self.config.gnss_file_path,
            "ground_truth_file": self.config.ground_truth_file_path,
        }

    @staticmethod
    def normalize_timestamp(val: float, unit: TimestampUnit) -> float:
        """Convert timestamp to canonical seconds.

        Args:
            val (float): Raw timestamp value.
            unit (TimestampUnit): Input timestamp unit.

        Returns:
            float: Timestamp in seconds.
        """
        if unit == TimestampUnit.SECONDS:
            return float(val)
        elif unit == TimestampUnit.MILLISECONDS:
            return float(val) * 1e-3
        elif unit == TimestampUnit.MICROSECONDS:
            return float(val) * 1e-6
        elif unit == TimestampUnit.NANOSECONDS:
            return float(val) * 1e-9
        else:
            raise ValueError(f"Unsupported timestamp unit '{unit}'.")

    @staticmethod
    def normalize_accel(val: float, unit: AccelUnit) -> float:
        """Convert acceleration to canonical m/s^2.

        Args:
            val (float): Raw acceleration value.
            unit (AccelUnit): Input acceleration unit.

        Returns:
            float: Acceleration in m/s^2.
        """
        if unit == AccelUnit.MPS2:
            return float(val)
        elif unit == AccelUnit.G:
            return float(val) * 9.80665
        else:
            raise ValueError(f"Unsupported accelerometer unit '{unit}'.")

    @staticmethod
    def normalize_gyro(val: float, unit: GyroUnit) -> float:
        """Convert angular velocity to canonical rad/s.

        Args:
            val (float): Raw gyroscope value.
            unit (GyroUnit): Input gyroscope unit.

        Returns:
            float: Angular velocity in rad/s.
        """
        if unit == GyroUnit.RAD_PER_SEC:
            return float(val)
        elif unit == GyroUnit.DEG_PER_SEC:
            return float(val) * (math.pi / 180.0)
        else:
            raise ValueError(f"Unsupported gyroscope unit '{unit}'.")

    @staticmethod
    def apply_axis_transform(vec3: np.ndarray, R_transform: np.ndarray) -> np.ndarray:
        """Apply 3x3 orthogonal permutation matrix to 3D vector.

        Args:
            vec3 (np.ndarray): 3D input vector [x, y, z].
            R_transform (np.ndarray): 3x3 transformation matrix.

        Returns:
            np.ndarray: Transformed 3D vector.
        """
        return np.asarray(R_transform @ vec3, dtype=np.float64)
