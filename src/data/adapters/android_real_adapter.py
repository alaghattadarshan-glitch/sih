"""Android Real-World Session Data Adapter & Parser.

Parses versioned real-world sensor logs recorded by Android RealSensorRecorder
(session_metadata.json, imu.csv, gnss.csv, navigation_state.csv) into canonical
observation objects ready for offline Python navigation replay and parity verification.
"""

import json
import os
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

from src.data.observations import IMUObservation, GNSSObservation


class AndroidRealSession:
    """Encapsulates a parsed real-world Android sensor recording session."""

    def __init__(
        self,
        session_dir: str,
        metadata: Dict[str, Any],
        imu_observations: List[IMUObservation],
        gnss_observations: List[GNSSObservation],
        nav_df: Optional[pd.DataFrame] = None,
    ):
        self.session_dir = session_dir
        self.metadata = metadata
        self.imu_observations = imu_observations
        self.gnss_observations = gnss_observations
        self.nav_df = nav_df

    @property
    def session_id(self) -> str:
        return self.metadata.get("session_id", "unknown")

    @property
    def duration_sec(self) -> float:
        if not self.imu_observations:
            return 0.0
        return self.imu_observations[-1].timestamp - self.imu_observations[0].timestamp

    @property
    def imu_count(self) -> int:
        return len(self.imu_observations)

    @property
    def gnss_count(self) -> int:
        return len(self.gnss_observations)


class AndroidRealDataAdapter:
    """Loader and validator for Android real-world sensor recordings."""

    @staticmethod
    def load_session(session_dir: str) -> AndroidRealSession:
        """Load a session directory containing real Android CSV recordings.

        Args:
            session_dir (str): Path to directory containing session_metadata.json, imu.csv, gnss.csv.

        Returns:
            AndroidRealSession: Parsed session object.
        """
        if not os.path.isdir(session_dir):
            raise FileNotFoundError(f"Session directory not found: {session_dir}")

        meta_path = os.path.join(session_dir, "session_metadata.json")
        imu_path = os.path.join(session_dir, "imu.csv")
        gnss_path = os.path.join(session_dir, "gnss.csv")
        nav_path = os.path.join(session_dir, "navigation_state.csv")

        metadata: Dict[str, Any] = {}
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                metadata = json.load(f)

        imu_observations: List[IMUObservation] = []
        if os.path.exists(imu_path):
            imu_df = pd.read_csv(imu_path)
            for _, row in imu_df.iterrows():
                imu_observations.append(
                    IMUObservation(
                        timestamp=float(row["timestamp"]),
                        accelerometer_x=float(row["accel_x"]),
                        accelerometer_y=float(row["accel_y"]),
                        accelerometer_z=float(row["accel_z"]),
                        gyroscope_x=float(row["gyro_x"]),
                        gyroscope_y=float(row["gyro_y"]),
                        gyroscope_z=float(row["gyro_z"]),
                    )
                )

        gnss_observations: List[GNSSObservation] = []
        if os.path.exists(gnss_path):
            gnss_df = pd.read_csv(gnss_path)
            for _, row in gnss_df.iterrows():
                gnss_observations.append(
                    GNSSObservation(
                        timestamp=float(row["timestamp"]),
                        latitude=float(row["latitude"]),
                        longitude=float(row["longitude"]),
                        altitude=float(row["altitude"]),
                        horizontal_accuracy=float(row["horizontal_accuracy"]) if "horizontal_accuracy" in row and not pd.isna(row["horizontal_accuracy"]) else 2.5,
                        speed=float(row["speed"]) if "speed" in row and not pd.isna(row["speed"]) else None,
                        heading=float(row["bearing"]) if "bearing" in row and not pd.isna(row["bearing"]) else None,
                    )
                )

        nav_df = pd.read_csv(nav_path) if os.path.exists(nav_path) else None

        return AndroidRealSession(
            session_dir=session_dir,
            metadata=metadata,
            imu_observations=imu_observations,
            gnss_observations=gnss_observations,
            nav_df=nav_df,
        )
