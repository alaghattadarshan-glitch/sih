"""Generic Configurable CSV Data Loader Module.

Provides loaders for IMU, GNSS, and Ground Truth CSV files with configurable
column mapping and explicit unit conversions (e.g. gyroscope deg/s to rad/s).
"""

import math
from pathlib import Path
from typing import List, Optional, Dict, Union
import pandas as pd
import yaml

from src.data.observations import IMUObservation, GNSSObservation, GroundTruthObservation
from src.data.conversions import dataframe_to_imu, dataframe_to_gnss, dataframe_to_ground_truth
from src.data.validation import validate_imu_observation, validate_gnss_observation, validate_ground_truth_observation


def load_config_column_mapping(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Dict[str, str]]:
    """Load column mappings from config.yaml if available."""
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent.parent.parent / "config" / "config.yaml"
    
    if Path(config_path).exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            if "data" in config and "column_mapping" in config["data"]:
                return config["data"]["column_mapping"]
    return {}


def load_imu_csv(
    filepath: Union[str, Path],
    column_map: Optional[Dict[str, str]] = None,
    gyro_unit: str = "rad/s",
) -> List[IMUObservation]:
    """Load IMU observations from a CSV file.

    Args:
        filepath (Union[str, Path]): Path to CSV file.
        column_map (Optional[Dict[str, str]]): Mapping of standard column name -> CSV header.
        gyro_unit (str): Unit of gyroscope data in CSV ('rad/s' or 'deg/s'). Default 'rad/s'.

    Returns:
        List[IMUObservation]: Validated list of IMU observations.
    """
    df = pd.read_csv(filepath)

    if gyro_unit.lower() in ("deg/s", "degrees/s", "deg_per_sec"):
        # Convert gyroscope columns from deg/s to rad/s
        gyro_cols = ["gyro_x", "gyro_y", "gyro_z"]
        if column_map:
            gyro_cols = [column_map.get(col, col) for col in gyro_cols]
        for col in gyro_cols:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: math.radians(float(x)) if pd.notna(x) else x)

    observations = dataframe_to_imu(df, col_map=column_map)

    # Validate all loaded observations
    for obs in observations:
        validate_imu_observation(obs)

    return observations


def load_gnss_csv(
    filepath: Union[str, Path],
    column_map: Optional[Dict[str, str]] = None,
) -> List[GNSSObservation]:
    """Load GNSS observations from a CSV file.

    Args:
        filepath (Union[str, Path]): Path to CSV file.
        column_map (Optional[Dict[str, str]]): Mapping of standard column name -> CSV header.

    Returns:
        List[GNSSObservation]: Validated list of GNSS observations.
    """
    df = pd.read_csv(filepath)
    observations = dataframe_to_gnss(df, col_map=column_map)

    for obs in observations:
        validate_gnss_observation(obs)

    return observations


def load_ground_truth_csv(
    filepath: Union[str, Path],
    column_map: Optional[Dict[str, str]] = None,
) -> List[GroundTruthObservation]:
    """Load Ground Truth observations from a CSV file.

    Args:
        filepath (Union[str, Path]): Path to CSV file.
        column_map (Optional[Dict[str, str]]): Mapping of standard column name -> CSV header.

    Returns:
        List[GroundTruthObservation]: Validated list of Ground Truth observations.
    """
    df = pd.read_csv(filepath)
    observations = dataframe_to_ground_truth(df, col_map=column_map)

    for obs in observations:
        validate_ground_truth_observation(obs)

    return observations
