"""Core Sensor Observation Data Structures.

This module defines strongly-typed dataclasses for IMU, GNSS, and Ground Truth
observations. Strict unit conventions are enforced and documented.

Unit Conventions:
- Timestamp: seconds (UTC or relative epoch)
- Acceleration: m/s^2
- Angular Velocity: rad/s
- Magnetometer: microteslas (uT)
- Latitude / Longitude: degrees [-90.0, +90.0] and [-180.0, +180.0]
- Altitude: meters above WGS84 ellipsoid
- Linear Velocity: m/s (East, North, Up or X, Y, Z)
- Heading / Orientation: degrees [0.0, 360.0)
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class IMUObservation:
    """Represents a single Inertial Measurement Unit (IMU) sample.

    Attributes:
        timestamp (float): Sample epoch timestamp in seconds.
        accelerometer_x (float): Accelerometer X-axis reading in m/s^2.
        accelerometer_y (float): Accelerometer Y-axis reading in m/s^2.
        accelerometer_z (float): Accelerometer Z-axis reading in m/s^2.
        gyroscope_x (float): Gyroscope X-axis rate of turn in rad/s.
        gyroscope_y (float): Gyroscope Y-axis rate of turn in rad/s.
        gyroscope_z (float): Gyroscope Z-axis rate of turn in rad/s.
        magnetometer_x (Optional[float]): Magnetometer X-axis in uT (optional).
        magnetometer_y (Optional[float]): Magnetometer Y-axis in uT (optional).
        magnetometer_z (Optional[float]): Magnetometer Z-axis in uT (optional).
    """

    timestamp: float
    accelerometer_x: float
    accelerometer_y: float
    accelerometer_z: float
    gyroscope_x: float
    gyroscope_y: float
    gyroscope_z: float
    magnetometer_x: Optional[float] = None
    magnetometer_y: Optional[float] = None
    magnetometer_z: Optional[float] = None


@dataclass
class GNSSObservation:
    """Represents a single GNSS satellite positioning fix.

    Attributes:
        timestamp (float): Fix epoch timestamp in seconds.
        latitude (float): Geodetic latitude in degrees [-90.0, +90.0].
        longitude (float): Geodetic longitude in degrees [-180.0, +180.0].
        altitude (float): WGS84 ellipsoidal height in meters.
        velocity_east (Optional[float]): Eastward velocity in m/s.
        velocity_north (Optional[float]): Northward velocity in m/s.
        velocity_up (Optional[float]): Upward velocity in m/s.
        speed (Optional[float]): Ground speed in m/s.
        heading (Optional[float]): Ground track heading in degrees [0.0, 360.0).
        horizontal_accuracy (Optional[float]): Estimated 1-sigma horizontal position accuracy in meters.
        vertical_accuracy (Optional[float]): Estimated 1-sigma vertical accuracy in meters.
    """

    timestamp: float
    latitude: float
    longitude: float
    altitude: float
    velocity_east: Optional[float] = None
    velocity_north: Optional[float] = None
    velocity_up: Optional[float] = None
    speed: Optional[float] = None
    heading: Optional[float] = None
    horizontal_accuracy: Optional[float] = None
    vertical_accuracy: Optional[float] = None


@dataclass
class GroundTruthObservation:
    """Represents reference ground-truth trajectory state for evaluation.

    Attributes:
        timestamp (float): Epoch timestamp in seconds.
        latitude (float): Geodetic latitude in degrees.
        longitude (float): Geodetic longitude in degrees.
        altitude (float): Ellipsoidal height in meters.
        velocity_east (Optional[float]): Ground-truth Eastward velocity in m/s.
        velocity_north (Optional[float]): Ground-truth Northward velocity in m/s.
        velocity_up (Optional[float]): Ground-truth Upward velocity in m/s.
        speed (Optional[float]): True speed in m/s.
        heading (Optional[float]): True heading in degrees.
        roll (Optional[float]): True roll angle in degrees.
        pitch (Optional[float]): True pitch angle in degrees.
        yaw (Optional[float]): True yaw angle in degrees.
    """

    timestamp: float
    latitude: float
    longitude: float
    altitude: float
    velocity_east: Optional[float] = None
    velocity_north: Optional[float] = None
    velocity_up: Optional[float] = None
    speed: Optional[float] = None
    heading: Optional[float] = None
    roll: Optional[float] = None
    pitch: Optional[float] = None
    yaw: Optional[float] = None
