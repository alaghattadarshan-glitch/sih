"""Local East-North-Up (ENU) Tangent Plane Coordinate Transformations.

This module provides transformations between Earth-Centered Earth-Fixed (ECEF) /
Geodetic (LLH) coordinates and a local Cartesian ENU tangent frame centered at
a specified reference origin.

Units Convention:
- Geographic coordinates (lat, lon): DEGREES
- Reference height & ENU coordinates (east, north, up): METERS
- ECEF coordinates (x, y, z): METERS
"""

import math
from typing import Tuple
from src.coordinate_transforms.wgs84 import WGS84, WGS84Ellipsoid
from src.coordinate_transforms.geodetic import llh_to_ecef, ecef_to_llh


def rotation_matrix_ecef_to_enu(lat_deg: float, lon_deg: float) -> Tuple[
    Tuple[float, float, float],
    Tuple[float, float, float],
    Tuple[float, float, float],
]:
    """Compute the 3x3 rotation matrix from ECEF to ENU frame at (lat, lon).

    Rotation Matrix R:
        [ -sin(lon),             cos(lon),            0       ]
        [ -sin(lat)*cos(lon),   -sin(lat)*sin(lon),   cos(lat)]
        [  cos(lat)*cos(lon),    cos(lat)*sin(lon),   sin(lat)]

    Args:
        lat_deg (float): Reference latitude in degrees.
        lon_deg (float): Reference longitude in degrees.

    Returns:
        Tuple of 3 tuples representing the 3x3 rotation matrix rows.
    """
    phi = math.radians(lat_deg)
    lam = math.radians(lon_deg)

    sin_phi = math.sin(phi)
    cos_phi = math.cos(phi)
    sin_lam = math.sin(lam)
    cos_lam = math.cos(lam)

    r1 = (-sin_lam, cos_lam, 0.0)
    r2 = (-sin_phi * cos_lam, -sin_phi * sin_lam, cos_phi)
    r3 = (cos_phi * cos_lam, cos_phi * sin_lam, sin_phi)

    return r1, r2, r3


def ecef_to_enu(
    x: float,
    y: float,
    z: float,
    ref_lat_deg: float,
    ref_lon_deg: float,
    ref_h_m: float,
    ellipsoid: WGS84Ellipsoid = WGS84,
) -> Tuple[float, float, float]:
    """Transform ECEF (X, Y, Z) point to ENU (East, North, Up) relative to reference origin.

    Args:
        x (float): Target ECEF X in meters.
        y (float): Target ECEF Y in meters.
        z (float): Target ECEF Z in meters.
        ref_lat_deg (float): Reference origin latitude in degrees.
        ref_lon_deg (float): Reference origin longitude in degrees.
        ref_h_m (float): Reference origin height in meters.
        ellipsoid (WGS84Ellipsoid): Reference ellipsoid model.

    Returns:
        Tuple[float, float, float]: ENU coordinates (east, north, up) in meters.
    """
    x0, y0, z0 = llh_to_ecef(ref_lat_deg, ref_lon_deg, ref_h_m, ellipsoid)

    dx = x - x0
    dy = y - y0
    dz = z - z0

    r1, r2, r3 = rotation_matrix_ecef_to_enu(ref_lat_deg, ref_lon_deg)

    east = r1[0] * dx + r1[1] * dy + r1[2] * dz
    north = r2[0] * dx + r2[1] * dy + r2[2] * dz
    up = r3[0] * dx + r3[1] * dy + r3[2] * dz

    return east, north, up


def enu_to_ecef(
    east: float,
    north: float,
    up: float,
    ref_lat_deg: float,
    ref_lon_deg: float,
    ref_h_m: float,
    ellipsoid: WGS84Ellipsoid = WGS84,
) -> Tuple[float, float, float]:
    """Transform ENU (East, North, Up) point back to ECEF (X, Y, Z).

    Uses the transpose of the ECEF-to-ENU rotation matrix (since R is orthogonal).

    Args:
        east (float): Local East displacement in meters.
        north (float): Local North displacement in meters.
        up (float): Local Up displacement in meters.
        ref_lat_deg (float): Reference origin latitude in degrees.
        ref_lon_deg (float): Reference origin longitude in degrees.
        ref_h_m (float): Reference origin height in meters.
        ellipsoid (WGS84Ellipsoid): Reference ellipsoid model.

    Returns:
        Tuple[float, float, float]: Target ECEF coordinates (x, y, z) in meters.
    """
    x0, y0, z0 = llh_to_ecef(ref_lat_deg, ref_lon_deg, ref_h_m, ellipsoid)
    r1, r2, r3 = rotation_matrix_ecef_to_enu(ref_lat_deg, ref_lon_deg)

    # Multiply ENU vector by R^T
    dx = r1[0] * east + r2[0] * north + r3[0] * up
    dy = r1[1] * east + r2[1] * north + r3[1] * up
    dz = r1[2] * east + r2[2] * north + r3[2] * up

    return x0 + dx, y0 + dy, z0 + dz


def llh_to_enu(
    lat_deg: float,
    lon_deg: float,
    h_m: float,
    ref_lat_deg: float,
    ref_lon_deg: float,
    ref_h_m: float,
    ellipsoid: WGS84Ellipsoid = WGS84,
) -> Tuple[float, float, float]:
    """Direct convenience transformation: Geodetic LLH -> Local ENU.

    Args:
        lat_deg (float): Target latitude in degrees.
        lon_deg (float): Target longitude in degrees.
        h_m (float): Target height in meters.
        ref_lat_deg (float): Reference origin latitude in degrees.
        ref_lon_deg (float): Reference origin longitude in degrees.
        ref_h_m (float): Reference origin height in meters.
        ellipsoid (WGS84Ellipsoid): Reference ellipsoid model.

    Returns:
        Tuple[float, float, float]: ENU displacement (east, north, up) in meters.
    """
    x, y, z = llh_to_ecef(lat_deg, lon_deg, h_m, ellipsoid)
    return ecef_to_enu(x, y, z, ref_lat_deg, ref_lon_deg, ref_h_m, ellipsoid)


def enu_to_llh(
    east: float,
    north: float,
    up: float,
    ref_lat_deg: float,
    ref_lon_deg: float,
    ref_h_m: float,
    ellipsoid: WGS84Ellipsoid = WGS84,
) -> Tuple[float, float, float]:
    """Direct convenience transformation: Local ENU -> Geodetic LLH.

    Args:
        east (float): Local East displacement in meters.
        north (float): Local North displacement in meters.
        up (float): Local Up displacement in meters.
        ref_lat_deg (float): Reference origin latitude in degrees.
        ref_lon_deg (float): Reference origin longitude in degrees.
        ref_h_m (float): Reference origin height in meters.
        ellipsoid (WGS84Ellipsoid): Reference ellipsoid model.

    Returns:
        Tuple[float, float, float]: Target Geodetic LLH (lat_deg, lon_deg, h_m).
    """
    x, y, z = enu_to_ecef(east, north, up, ref_lat_deg, ref_lon_deg, ref_h_m, ellipsoid)
    return ecef_to_llh(x, y, z, ellipsoid)
