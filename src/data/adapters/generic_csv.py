"""Generic CSV dataset adapter with configurable column mappings and unit normalization.

Allows ingestion of smartphone recordings (Sensor Logger, Android), vehicle loggers,
and custom multi-sensor CSV datasets.
"""

import os
import math
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd

from src.data.observations import IMUObservation, GNSSObservation, GroundTruthObservation
from src.data.adapters.base import DatasetAdapter
from src.data.adapters.schema import DatasetConfig


class GenericCSVAdapter(DatasetAdapter):
    """Configurable CSV adapter mapping arbitrary dataset columns into canonical observations."""

    def load_imu(self) -> List[IMUObservation]:
        """Load, validate, normalize units, and apply axis transformation for IMU CSV data.

        Returns:
            List[IMUObservation]: Canonical IMU observations.

        Raises:
            FileNotFoundError: If imu_file_path does not exist.
            ValueError: If required columns are missing or invalid.
        """
        path = self.config.imu_file_path
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"IMU data file not found at '{path}'.")

        mapping = self.config.imu_mapping
        df = pd.read_csv(path, delimiter=self.config.delimiter, skiprows=self.config.skip_rows)

        # Check required columns
        req_cols = [mapping.timestamp, mapping.accel_x, mapping.accel_y, mapping.accel_z, mapping.gyro_x, mapping.gyro_y, mapping.gyro_z]
        missing = [c for c in req_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required IMU columns in {path}: {missing}. Available columns: {list(df.columns)}")

        # Build axis transformation matrix
        R_axis = mapping.axis_transform.build_matrix()

        # Extract numpy arrays
        raw_t = df[mapping.timestamp].to_numpy(dtype=np.float64)
        raw_ax = df[mapping.accel_x].to_numpy(dtype=np.float64)
        raw_ay = df[mapping.accel_y].to_numpy(dtype=np.float64)
        raw_az = df[mapping.accel_z].to_numpy(dtype=np.float64)
        raw_gx = df[mapping.gyro_x].to_numpy(dtype=np.float64)
        raw_gy = df[mapping.gyro_y].to_numpy(dtype=np.float64)
        raw_gz = df[mapping.gyro_z].to_numpy(dtype=np.float64)

        # Check Magnetometer if configured
        has_mag = (
            mapping.mag_x in df.columns
            and mapping.mag_y in df.columns
            and mapping.mag_z in df.columns
            if mapping.mag_x and mapping.mag_y and mapping.mag_z
            else False
        )
        raw_mx = df[mapping.mag_x].to_numpy(dtype=np.float64) if has_mag else None
        raw_my = df[mapping.mag_y].to_numpy(dtype=np.float64) if has_mag else None
        raw_mz = df[mapping.mag_z].to_numpy(dtype=np.float64) if has_mag else None

        # Normalize units
        t_sec = np.array([self.normalize_timestamp(t, mapping.timestamp_unit) for t in raw_t], dtype=np.float64)
        ax_mps2 = np.array([self.normalize_accel(a, mapping.accel_unit) for a in raw_ax], dtype=np.float64)
        ay_mps2 = np.array([self.normalize_accel(a, mapping.accel_unit) for a in raw_ay], dtype=np.float64)
        az_mps2 = np.array([self.normalize_accel(a, mapping.accel_unit) for a in raw_az], dtype=np.float64)
        gx_rads = np.array([self.normalize_gyro(g, mapping.gyro_unit) for g in raw_gx], dtype=np.float64)
        gy_rads = np.array([self.normalize_gyro(g, mapping.gyro_unit) for g in raw_gy], dtype=np.float64)
        gz_rads = np.array([self.normalize_gyro(g, mapping.gyro_unit) for g in raw_gz], dtype=np.float64)

        accel_mat = np.column_stack([ax_mps2, ay_mps2, az_mps2])  # (N, 3)
        gyro_mat = np.column_stack([gx_rads, gy_rads, gz_rads])   # (N, 3)

        # Apply axis transformation: v_body = R @ v_device
        accel_body = (R_axis @ accel_mat.T).T
        gyro_body = (R_axis @ gyro_mat.T).T

        imu_observations: List[IMUObservation] = []
        for i in range(len(t_sec)):
            t = float(t_sec[i])
            if math.isnan(t) or math.isinf(t):
                continue

            mx = float(raw_mx[i]) if has_mag and raw_mx is not None else None
            my = float(raw_my[i]) if has_mag and raw_my is not None else None
            mz = float(raw_mz[i]) if has_mag and raw_mz is not None else None

            obs = IMUObservation(
                timestamp=t,
                accelerometer_x=float(accel_body[i, 0]),
                accelerometer_y=float(accel_body[i, 1]),
                accelerometer_z=float(accel_body[i, 2]),
                gyroscope_x=float(gyro_body[i, 0]),
                gyroscope_y=float(gyro_body[i, 1]),
                gyroscope_z=float(gyro_body[i, 2]),
                magnetometer_x=mx,
                magnetometer_y=my,
                magnetometer_z=mz,
            )
            imu_observations.append(obs)

        return imu_observations

    def load_gnss(self) -> List[GNSSObservation]:
        """Load and normalize GNSS satellite fixes.

        Returns:
            List[GNSSObservation]: Canonical GNSS observations.
        """
        path = self.config.gnss_file_path
        if not path or not os.path.exists(path):
            return []

        mapping = self.config.gnss_mapping
        if mapping is None:
            return []

        df = pd.read_csv(path, delimiter=self.config.delimiter, skiprows=self.config.skip_rows)
        req_cols = [mapping.timestamp, mapping.latitude, mapping.longitude, mapping.altitude]
        missing = [c for c in req_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required GNSS columns in {path}: {missing}. Available: {list(df.columns)}")

        raw_t = df[mapping.timestamp].to_numpy(dtype=np.float64)
        raw_lat = df[mapping.latitude].to_numpy(dtype=np.float64)
        raw_lon = df[mapping.longitude].to_numpy(dtype=np.float64)
        raw_alt = df[mapping.altitude].to_numpy(dtype=np.float64)

        ve = df[mapping.velocity_east].to_numpy(dtype=np.float64) if mapping.velocity_east in df.columns else None
        vn = df[mapping.velocity_north].to_numpy(dtype=np.float64) if mapping.velocity_north in df.columns else None
        vu = df[mapping.velocity_up].to_numpy(dtype=np.float64) if mapping.velocity_up in df.columns else None
        speed = df[mapping.speed].to_numpy(dtype=np.float64) if mapping.speed in df.columns else None
        heading = df[mapping.heading].to_numpy(dtype=np.float64) if mapping.heading in df.columns else None
        h_acc = df[mapping.horizontal_accuracy].to_numpy(dtype=np.float64) if mapping.horizontal_accuracy in df.columns else None
        v_acc = df[mapping.vertical_accuracy].to_numpy(dtype=np.float64) if mapping.vertical_accuracy in df.columns else None

        t_sec = np.array([self.normalize_timestamp(t, mapping.timestamp_unit) for t in raw_t], dtype=np.float64)

        gnss_observations: List[GNSSObservation] = []
        for i in range(len(t_sec)):
            t = float(t_sec[i])
            lat = float(raw_lat[i])
            lon = float(raw_lon[i])
            alt = float(raw_alt[i])

            if any(math.isnan(v) or math.isinf(v) for v in (t, lat, lon, alt)):
                continue

            obs = GNSSObservation(
                timestamp=t,
                latitude=lat,
                longitude=lon,
                altitude=alt,
                velocity_east=float(ve[i]) if ve is not None and not math.isnan(ve[i]) else None,
                velocity_north=float(vn[i]) if vn is not None and not math.isnan(vn[i]) else None,
                velocity_up=float(vu[i]) if vu is not None and not math.isnan(vu[i]) else None,
                speed=float(speed[i]) if speed is not None and not math.isnan(speed[i]) else None,
                heading=float(heading[i]) if heading is not None and not math.isnan(heading[i]) else None,
                horizontal_accuracy=float(h_acc[i]) if h_acc is not None and not math.isnan(h_acc[i]) else None,
                vertical_accuracy=float(v_acc[i]) if v_acc is not None and not math.isnan(v_acc[i]) else None,
            )
            gnss_observations.append(obs)

        return gnss_observations

    def load_ground_truth(self) -> Optional[List[GroundTruthObservation]]:
        """Load reference ground truth trajectory if configured.

        Returns:
            Optional[List[GroundTruthObservation]]: Reference observations.
        """
        path = self.config.ground_truth_file_path
        if not path or not os.path.exists(path):
            return None

        mapping = self.config.ground_truth_mapping
        if mapping is None:
            return None

        df = pd.read_csv(path, delimiter=self.config.delimiter, skiprows=self.config.skip_rows)
        req_cols = [mapping.timestamp, mapping.latitude, mapping.longitude, mapping.altitude]
        missing = [c for c in req_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required Ground Truth columns in {path}: {missing}")

        raw_t = df[mapping.timestamp].to_numpy(dtype=np.float64)
        raw_lat = df[mapping.latitude].to_numpy(dtype=np.float64)
        raw_lon = df[mapping.longitude].to_numpy(dtype=np.float64)
        raw_alt = df[mapping.altitude].to_numpy(dtype=np.float64)

        ve = df[mapping.velocity_east].to_numpy(dtype=np.float64) if mapping.velocity_east in df.columns else None
        vn = df[mapping.velocity_north].to_numpy(dtype=np.float64) if mapping.velocity_north in df.columns else None
        vu = df[mapping.velocity_up].to_numpy(dtype=np.float64) if mapping.velocity_up in df.columns else None
        speed = df[mapping.speed].to_numpy(dtype=np.float64) if mapping.speed in df.columns else None
        heading = df[mapping.heading].to_numpy(dtype=np.float64) if mapping.heading in df.columns else None
        roll = df[mapping.roll].to_numpy(dtype=np.float64) if mapping.roll in df.columns else None
        pitch = df[mapping.pitch].to_numpy(dtype=np.float64) if mapping.pitch in df.columns else None
        yaw = df[mapping.yaw].to_numpy(dtype=np.float64) if mapping.yaw in df.columns else None

        t_sec = np.array([self.normalize_timestamp(t, mapping.timestamp_unit) for t in raw_t], dtype=np.float64)

        gt_observations: List[GroundTruthObservation] = []
        for i in range(len(t_sec)):
            t = float(t_sec[i])
            lat = float(raw_lat[i])
            lon = float(raw_lon[i])
            alt = float(raw_alt[i])

            if any(math.isnan(v) or math.isinf(v) for v in (t, lat, lon, alt)):
                continue

            obs = GroundTruthObservation(
                timestamp=t,
                latitude=lat,
                longitude=lon,
                altitude=alt,
                velocity_east=float(ve[i]) if ve is not None and not math.isnan(ve[i]) else None,
                velocity_north=float(vn[i]) if vn is not None and not math.isnan(vn[i]) else None,
                velocity_up=float(vu[i]) if vu is not None and not math.isnan(vu[i]) else None,
                speed=float(speed[i]) if speed is not None and not math.isnan(speed[i]) else None,
                heading=float(heading[i]) if heading is not None and not math.isnan(heading[i]) else None,
                roll=float(roll[i]) if roll is not None and not math.isnan(roll[i]) else None,
                pitch=float(pitch[i]) if pitch is not None and not math.isnan(pitch[i]) else None,
                yaw=float(yaw[i]) if yaw is not None and not math.isnan(yaw[i]) else None,
            )
            gt_observations.append(obs)

        return gt_observations
