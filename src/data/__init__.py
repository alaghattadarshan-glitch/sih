"""Data handling, observations, loaders, synchronization, and preprocessing package."""

from src.data.observations import IMUObservation, GNSSObservation, GroundTruthObservation
from src.data.validation import (
    validate_imu_observation,
    validate_gnss_observation,
    validate_ground_truth_observation,
    validate_timestamp_sequence,
    DataValidationError,
)
from src.data.conversions import (
    imu_to_dataframe,
    gnss_to_dataframe,
    ground_truth_to_dataframe,
    dataframe_to_imu,
    dataframe_to_gnss,
    dataframe_to_ground_truth,
)
from src.data.synchronization import (
    estimate_sampling_rate,
    find_timestamp_gaps,
    calculate_timestamp_statistics,
    synchronize_imu_gnss,
    SynchronizedDataset,
)

__all__ = [
    "IMUObservation",
    "GNSSObservation",
    "GroundTruthObservation",
    "validate_imu_observation",
    "validate_gnss_observation",
    "validate_ground_truth_observation",
    "validate_timestamp_sequence",
    "DataValidationError",
    "imu_to_dataframe",
    "gnss_to_dataframe",
    "ground_truth_to_dataframe",
    "dataframe_to_imu",
    "dataframe_to_gnss",
    "dataframe_to_ground_truth",
    "estimate_sampling_rate",
    "find_timestamp_gaps",
    "calculate_timestamp_statistics",
    "synchronize_imu_gnss",
    "SynchronizedDataset",
]
