"""Schema, Enums, and Configuration Models for Dataset Adapters.

Defines supported units, axis transformation models, column mapping schemas,
and dataset ingestion configuration specifications.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


class TimestampUnit(str, Enum):
    """Supported timestamp units for automatic conversion to seconds."""
    SECONDS = "s"
    MILLISECONDS = "ms"
    MICROSECONDS = "us"
    NANOSECONDS = "ns"


class AccelUnit(str, Enum):
    """Supported accelerometer measurement units for conversion to m/s^2."""
    MPS2 = "m/s^2"
    G = "g"


class GyroUnit(str, Enum):
    """Supported gyroscope measurement units for conversion to rad/s."""
    RAD_PER_SEC = "rad/s"
    DEG_PER_SEC = "deg/s"


@dataclass
class AxisTransformConfig:
    """Configurable orthogonal axis permutation and sign inversion transform.

    Transforms raw device frame coordinates into the canonical body frame:
        v_body = R_device_to_body @ v_device
    """
    x_map: str = "+x"  # e.g., "+x", "-x", "+y", "-y", "+z", "-z"
    y_map: str = "+y"
    z_map: str = "+z"

    def build_matrix(self) -> np.ndarray:
        """Build and validate 3x3 orthogonal signed permutation matrix.

        Returns:
            np.ndarray: 3x3 transformation matrix.

        Raises:
            ValueError: If axis mapping is invalid or non-orthogonal.
        """
        mapping = [self.x_map.strip().lower(), self.y_map.strip().lower(), self.z_map.strip().lower()]
        R = np.zeros((3, 3), dtype=np.float64)

        axis_dict = {"x": 0, "y": 1, "z": 2}
        used_source_axes = set()

        for body_idx, map_str in enumerate(mapping):
            if len(map_str) < 2 or map_str[0] not in ("+", "-") or map_str[1:] not in axis_dict:
                raise ValueError(
                    f"Invalid axis map specifier '{map_str}'. Must be formatted as '[+/-][x/y/z]' (e.g. '+x', '-y')."
                )

            sign = 1.0 if map_str[0] == "+" else -1.0
            src_axis = map_str[1:]
            src_idx = axis_dict[src_axis]

            if src_idx in used_source_axes:
                raise ValueError(
                    f"Duplicate source axis '{src_axis}' used in axis mapping: {mapping}."
                )
            used_source_axes.add(src_idx)

            R[body_idx, src_idx] = sign

        # Validate orthogonality: R @ R.T == I
        if not np.allclose(R @ R.T, np.eye(3), atol=1e-6):
            raise ValueError(f"Resulting axis transformation matrix is non-orthogonal:\n{R}")

        # Validate determinant is +/- 1
        det = np.linalg.det(R)
        if not (np.isclose(det, 1.0) or np.isclose(det, -1.0)):
            raise ValueError(f"Invalid transformation matrix determinant {det:.3f}.")

        return R


@dataclass
class IMUColumnMapping:
    """Column name mappings and unit definitions for IMU data."""
    timestamp: str = "timestamp"
    accel_x: str = "accel_x"
    accel_y: str = "accel_y"
    accel_z: str = "accel_z"
    gyro_x: str = "gyro_x"
    gyro_y: str = "gyro_y"
    gyro_z: str = "gyro_z"
    mag_x: Optional[str] = None
    mag_y: Optional[str] = None
    mag_z: Optional[str] = None
    timestamp_unit: TimestampUnit = TimestampUnit.SECONDS
    accel_unit: AccelUnit = AccelUnit.MPS2
    gyro_unit: GyroUnit = GyroUnit.RAD_PER_SEC
    axis_transform: AxisTransformConfig = field(default_factory=AxisTransformConfig)


@dataclass
class GNSSColumnMapping:
    """Column name mappings and unit definitions for GNSS data."""
    timestamp: str = "timestamp"
    latitude: str = "latitude"
    longitude: str = "longitude"
    altitude: str = "altitude"
    velocity_east: Optional[str] = None
    velocity_north: Optional[str] = None
    velocity_up: Optional[str] = None
    speed: Optional[str] = None
    heading: Optional[str] = None
    horizontal_accuracy: Optional[str] = None
    vertical_accuracy: Optional[str] = None
    timestamp_unit: TimestampUnit = TimestampUnit.SECONDS


@dataclass
class GroundTruthColumnMapping:
    """Column name mappings and unit definitions for Ground Truth reference data."""
    timestamp: str = "timestamp"
    latitude: str = "latitude"
    longitude: str = "longitude"
    altitude: str = "altitude"
    velocity_east: Optional[str] = None
    velocity_north: Optional[str] = None
    velocity_up: Optional[str] = None
    speed: Optional[str] = None
    heading: Optional[str] = None
    roll: Optional[str] = None
    pitch: Optional[str] = None
    yaw: Optional[str] = None
    timestamp_unit: TimestampUnit = TimestampUnit.SECONDS


@dataclass
class StaticCalibrationConfig:
    """Configuration for initial stationary sensor bias estimation."""
    enabled: bool = False
    start_time_sec: float = 0.0
    duration_sec: float = 5.0
    max_accel_std_mps2: float = 0.4
    max_gyro_std_rads: float = 0.05


@dataclass
class OutageSimulationConfig:
    """Configuration for artificial GNSS outage simulation."""
    enabled: bool = False
    start_time_sec: float = 30.0
    duration_sec: float = 30.0


@dataclass
class DatasetConfig:
    """Master configuration structure for real dataset ingestion and evaluation."""
    dataset_name: str = "generic_dataset"
    imu_file_path: str = ""
    gnss_file_path: Optional[str] = None
    ground_truth_file_path: Optional[str] = None
    delimiter: str = ","
    skip_rows: int = 0
    imu_mapping: IMUColumnMapping = field(default_factory=IMUColumnMapping)
    gnss_mapping: Optional[GNSSColumnMapping] = None
    ground_truth_mapping: Optional[GroundTruthColumnMapping] = None
    calibration: StaticCalibrationConfig = field(default_factory=StaticCalibrationConfig)
    outage: OutageSimulationConfig = field(default_factory=OutageSimulationConfig)
    max_sync_tolerance_sec: float = 0.5
    origin_lat: Optional[float] = None
    origin_lon: Optional[float] = None
    origin_alt: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
