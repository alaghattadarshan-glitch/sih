"""Geodetic (LLH) and Earth-Centered Earth-Fixed (ECEF) Transformation Module.

This module provides bidirectional transformations between WGS84 Geodetic
coordinates (Latitude, Longitude, Height) and Cartesian ECEF (X, Y, Z) coordinates.

Units Convention:
- Geodetic API Input/Output: Latitude & Longitude in DEGREES, Height in METERS.
- Internal Trigonometric Equations: Angle arguments in RADIANS.
- ECEF API Input/Output: X, Y, Z coordinates in METERS.
"""

import math
from typing import Tuple
from src.coordinate_transforms.wgs84 import WGS84, WGS84Ellipsoid


def normalize_longitude(lon_deg: float) -> float:
    """Normalize longitude to [-180.0, +180.0] without introducing floating point noise."""
    if -180.0 <= lon_deg <= 180.0:
        return lon_deg
    lon = ((lon_deg + 180.0) % 360.0) - 180.0
    return 180.0 if lon == -180.0 else lon


def llh_to_ecef(
    lat_deg: float,
    lon_deg: float,
    h_m: float,
    ellipsoid: WGS84Ellipsoid = WGS84,
) -> Tuple[float, float, float]:
    """Convert Geodetic LLH (Lat, Lon, Height) to ECEF (X, Y, Z) coordinates.

    Args:
        lat_deg (float): Geodetic latitude in degrees [-90.0, +90.0].
        lon_deg (float): Geodetic longitude in degrees [-180.0, +180.0].
        h_m (float): Height above WGS84 ellipsoid in meters.
        ellipsoid (WGS84Ellipsoid): Reference ellipsoid parameters.

    Returns:
        Tuple[float, float, float]: ECEF coordinates (x, y, z) in meters.

    Raises:
        ValueError: If latitude is outside [-90.0, +90.0].
    """
    if not (-90.0 <= lat_deg <= 90.0):
        raise ValueError(f"Latitude must be within [-90.0, +90.0], got {lat_deg}")

    lon_deg = normalize_longitude(lon_deg)

    phi = math.radians(lat_deg)
    lam = math.radians(lon_deg)

    sin_phi = math.sin(phi)
    cos_phi = math.cos(phi)
    sin_lam = math.sin(lam)
    cos_lam = math.cos(lam)

    n = ellipsoid.prime_vertical_radius(phi)

    x = (n + h_m) * cos_phi * cos_lam
    y = (n + h_m) * cos_phi * sin_lam
    z = (n * (1.0 - ellipsoid.e2) + h_m) * sin_phi

    return x, y, z


def ecef_to_llh(
    x: float,
    y: float,
    z: float,
    ellipsoid: WGS84Ellipsoid = WGS84,
    max_iter: int = 10,
    tol: float = 1e-12,
) -> Tuple[float, float, float]:
    """Convert ECEF (X, Y, Z) coordinates to Geodetic LLH (Lat, Lon, Height).

    Uses Bowring's non-iterative initial formulation combined with a high-precision
    convergent loop to guarantee sub-millimeter precision across equatorial,
    mid-latitude, polar, and high-altitude locations.

    Args:
        x (float): ECEF X coordinate in meters.
        y (float): ECEF Y coordinate in meters.
        z (float): ECEF Z coordinate in meters.
        ellipsoid (WGS84Ellipsoid): Reference ellipsoid parameters.
        max_iter (int): Maximum convergence iterations.
        tol (float): Convergence tolerance in radians.

    Returns:
        Tuple[float, float, float]: Geodetic coordinates (lat_deg, lon_deg, h_m).
    """
    p = math.hypot(x, y)

    # Singular case: Origin (0, 0, 0)
    if p < 1e-12 and abs(z) < 1e-12:
        return 0.0, 0.0, -ellipsoid.b

    # Singular case: Polar axes (X=0, Y=0)
    if p < 1e-12:
        lon_deg = 0.0
        if z >= 0.0:
            lat_deg = 90.0
            h_m = z - ellipsoid.b
        else:
            lat_deg = -90.0
            h_m = -z - ellipsoid.b
        return lat_deg, lon_deg, h_m

    lon_rad = math.atan2(y, x)

    # Initial Bowring approximation
    a = ellipsoid.a
    b = ellipsoid.b
    e2 = ellipsoid.e2
    ep2 = ellipsoid.ep2

    theta = math.atan2(z * a, p * b)
    lat_rad = math.atan2(
        z + ep2 * b * (math.sin(theta) ** 3),
        p - e2 * a * (math.cos(theta) ** 3),
    )

    # Refinement iteration loop with strict convergence checks
    for _ in range(max_iter):
        sin_lat = math.sin(lat_rad)
        cos_lat = math.cos(lat_rad)
        
        # Recalculate parametric latitude theta
        theta_new = math.atan2(b * sin_lat, a * cos_lat)
        lat_new = math.atan2(
            z + ep2 * b * (math.sin(theta_new) ** 3),
            p - e2 * a * (math.cos(theta_new) ** 3),
        )

        if abs(lat_new - lat_rad) < tol:
            lat_rad = lat_new
            break
        lat_rad = lat_new

    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)
    n = ellipsoid.prime_vertical_radius(lat_rad)

    # Height computation selecting numerically stable branch based on latitude
    if abs(cos_lat) > 0.1:
        h_m = (p / cos_lat) - n
    else:
        h_m = (abs(z) / abs(sin_lat)) - n * (1.0 - e2)

    lat_deg = math.degrees(lat_rad)
    lon_deg = normalize_longitude(math.degrees(lon_rad))

    return lat_deg, lon_deg, h_m
