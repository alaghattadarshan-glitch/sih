"""Tabular Pandas DataFrame Conversions.

Provides utilities to convert list of dataclass observations to/from
Pandas DataFrames with standardized column names and configurable column mapping.
"""

from typing import List, Optional, Dict
import pandas as pd

from src.data.observations import IMUObservation, GNSSObservation, GroundTruthObservation


def imu_to_dataframe(observations: List[IMUObservation]) -> pd.DataFrame:
    """Convert list of IMUObservation instances to Pandas DataFrame."""
    data = [
        {
            "timestamp": obs.timestamp,
            "accel_x": obs.accelerometer_x,
            "accel_y": obs.accelerometer_y,
            "accel_z": obs.accelerometer_z,
            "gyro_x": obs.gyroscope_x,
            "gyro_y": obs.gyroscope_y,
            "gyro_z": obs.gyroscope_z,
            "mag_x": obs.magnetometer_x,
            "mag_y": obs.magnetometer_y,
            "mag_z": obs.magnetometer_z,
        }
        for obs in observations
    ]
    return pd.DataFrame(data)


def gnss_to_dataframe(observations: List[GNSSObservation]) -> pd.DataFrame:
    """Convert list of GNSSObservation instances to Pandas DataFrame."""
    data = [
        {
            "timestamp": obs.timestamp,
            "latitude": obs.latitude,
            "longitude": obs.longitude,
            "altitude": obs.altitude,
            "velocity_east": obs.velocity_east,
            "velocity_north": obs.velocity_north,
            "velocity_up": obs.velocity_up,
            "speed": obs.speed,
            "heading": obs.heading,
            "horizontal_accuracy": obs.horizontal_accuracy,
            "vertical_accuracy": obs.vertical_accuracy,
        }
        for obs in observations
    ]
    return pd.DataFrame(data)


def ground_truth_to_dataframe(observations: List[GroundTruthObservation]) -> pd.DataFrame:
    """Convert list of GroundTruthObservation instances to Pandas DataFrame."""
    data = [
        {
            "timestamp": obs.timestamp,
            "latitude": obs.latitude,
            "longitude": obs.longitude,
            "altitude": obs.altitude,
            "velocity_east": obs.velocity_east,
            "velocity_north": obs.velocity_north,
            "velocity_up": obs.velocity_up,
            "speed": obs.speed,
            "heading": obs.heading,
            "roll": obs.roll,
            "pitch": obs.pitch,
            "yaw": obs.yaw,
        }
        for obs in observations
    ]
    return pd.DataFrame(data)


def dataframe_to_imu(
    df: pd.DataFrame, col_map: Optional[Dict[str, str]] = None
) -> List[IMUObservation]:
    """Convert Pandas DataFrame to list of IMUObservation instances.

    Args:
        df (pd.DataFrame): Input dataframe.
        col_map (Optional[Dict[str, str]]): Custom mapping from dataframe column name to expected standard key.

    Returns:
        List[IMUObservation]: List of IMU observation objects.
    """
    mapping = {
        "timestamp": "timestamp",
        "accel_x": "accel_x",
        "accel_y": "accel_y",
        "accel_z": "accel_z",
        "gyro_x": "gyro_x",
        "gyro_y": "gyro_y",
        "gyro_z": "gyro_z",
        "mag_x": "mag_x",
        "mag_y": "mag_y",
        "mag_z": "mag_z",
    }
    if col_map:
        mapping.update(col_map)

    # Invert mapping so we lookup df column from standard name
    inv_map = {v: k for k, v in mapping.items()}

    observations = []
    for _, row in df.iterrows():
        obs = IMUObservation(
            timestamp=float(row[inv_map["timestamp"]]),
            accelerometer_x=float(row[inv_map["accel_x"]]),
            accelerometer_y=float(row[inv_map["accel_y"]]),
            accelerometer_z=float(row[inv_map["accel_z"]]),
            gyroscope_x=float(row[inv_map["gyro_x"]]),
            gyroscope_y=float(row[inv_map["gyro_y"]]),
            gyroscope_z=float(row[inv_map["gyro_z"]]),
            magnetometer_x=(
                float(row[inv_map["mag_x"]])
                if inv_map.get("mag_x") in row and pd.notna(row[inv_map["mag_x"]])
                else None
            ),
            magnetometer_y=(
                float(row[inv_map["mag_y"]])
                if inv_map.get("mag_y") in row and pd.notna(row[inv_map["mag_y"]])
                else None
            ),
            magnetometer_z=(
                float(row[inv_map["mag_z"]])
                if inv_map.get("mag_z") in row and pd.notna(row[inv_map["mag_z"]])
                else None
            ),
        )
        observations.append(obs)
    return observations


def dataframe_to_gnss(
    df: pd.DataFrame, col_map: Optional[Dict[str, str]] = None
) -> List[GNSSObservation]:
    """Convert Pandas DataFrame to list of GNSSObservation instances."""
    mapping = {
        "timestamp": "timestamp",
        "latitude": "latitude",
        "longitude": "longitude",
        "altitude": "altitude",
    }
    if col_map:
        mapping.update(col_map)

    inv_map = {v: k for k, v in mapping.items()}

    observations = []
    for _, row in df.iterrows():
        obs = GNSSObservation(
            timestamp=float(row[inv_map["timestamp"]]),
            latitude=float(row[inv_map["latitude"]]),
            longitude=float(row[inv_map["longitude"]]),
            altitude=float(row[inv_map["altitude"]]),
            velocity_east=(
                float(row[col_map["velocity_east"]])
                if col_map and "velocity_east" in col_map and col_map["velocity_east"] in row
                else float(row["velocity_east"]) if "velocity_east" in row and pd.notna(row["velocity_east"]) else None
            ),
            velocity_north=(
                float(row[col_map["velocity_north"]])
                if col_map and "velocity_north" in col_map and col_map["velocity_north"] in row
                else float(row["velocity_north"]) if "velocity_north" in row and pd.notna(row["velocity_north"]) else None
            ),
            velocity_up=(
                float(row[col_map["velocity_up"]])
                if col_map and "velocity_up" in col_map and col_map["velocity_up"] in row
                else float(row["velocity_up"]) if "velocity_up" in row and pd.notna(row["velocity_up"]) else None
            ),
            speed=(
                float(row["speed"]) if "speed" in row and pd.notna(row["speed"]) else None
            ),
            heading=(
                float(row["heading"]) if "heading" in row and pd.notna(row["heading"]) else None
            ),
            horizontal_accuracy=(
                float(row["horizontal_accuracy"])
                if "horizontal_accuracy" in row and pd.notna(row["horizontal_accuracy"])
                else None
            ),
            vertical_accuracy=(
                float(row["vertical_accuracy"])
                if "vertical_accuracy" in row and pd.notna(row["vertical_accuracy"])
                else None
            ),
        )
        observations.append(obs)
    return observations


def dataframe_to_ground_truth(
    df: pd.DataFrame, col_map: Optional[Dict[str, str]] = None
) -> List[GroundTruthObservation]:
    """Convert Pandas DataFrame to list of GroundTruthObservation instances."""
    mapping = {
        "timestamp": "timestamp",
        "latitude": "latitude",
        "longitude": "longitude",
        "altitude": "altitude",
    }
    if col_map:
        mapping.update(col_map)

    inv_map = {v: k for k, v in mapping.items()}

    observations = []
    for _, row in df.iterrows():
        obs = GroundTruthObservation(
            timestamp=float(row[inv_map["timestamp"]]),
            latitude=float(row[inv_map["latitude"]]),
            longitude=float(row[inv_map["longitude"]]),
            altitude=float(row[inv_map["altitude"]]),
            velocity_east=(
                float(row["velocity_east"])
                if "velocity_east" in row and pd.notna(row["velocity_east"])
                else None
            ),
            velocity_north=(
                float(row["velocity_north"])
                if "velocity_north" in row and pd.notna(row["velocity_north"])
                else None
            ),
            velocity_up=(
                float(row["velocity_up"])
                if "velocity_up" in row and pd.notna(row["velocity_up"])
                else None
            ),
            speed=float(row["speed"]) if "speed" in row and pd.notna(row["speed"]) else None,
            heading=(
                float(row["heading"]) if "heading" in row and pd.notna(row["heading"]) else None
            ),
            roll=float(row["roll"]) if "roll" in row and pd.notna(row["roll"]) else None,
            pitch=float(row["pitch"]) if "pitch" in row and pd.notna(row["pitch"]) else None,
            yaw=float(row["yaw"]) if "yaw" in row and pd.notna(row["yaw"]) else None,
        )
        observations.append(obs)
    return observations
