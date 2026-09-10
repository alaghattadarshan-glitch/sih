"""Local Navigation Frame Abstraction.

Provides a clean object-oriented interface for a local East-North-Up (ENU)
tangent plane centered at a specific geographic reference origin.
"""

from typing import Tuple
from src.coordinate_transforms.wgs84 import WGS84, WGS84Ellipsoid
from src.coordinate_transforms.geodetic import llh_to_ecef
from src.coordinate_transforms.enu import (
    ecef_to_enu,
    enu_to_ecef,
    llh_to_enu,
    enu_to_llh,
)


def normalize_longitude(lon_deg: float) -> float:
    """Normalize longitude to [-180.0, +180.0] without introducing floating point noise."""
    if -180.0 <= lon_deg <= 180.0:
        return lon_deg
    lon = ((lon_deg + 180.0) % 360.0) - 180.0
    return 180.0 if lon == -180.0 else lon


class LocalFrame:
    """Represents a local tangent plane (ENU) centered at a reference geographic origin."""

    def __init__(
        self,
        latitude: float,
        longitude: float,
        height: float = 0.0,
        ellipsoid: WGS84Ellipsoid = WGS84,
    ):
        """Initialize local navigation frame at reference origin.

        Args:
            latitude (float): Origin reference latitude in degrees [-90.0, +90.0].
            longitude (float): Origin reference longitude in degrees [-180.0, +180.0].
            height (float): Origin reference ellipsoid height in meters. Default 0.0m.
            ellipsoid (WGS84Ellipsoid): WGS84 Reference ellipsoid parameters.
        """
        if not (-90.0 <= latitude <= 90.0):
            raise ValueError(f"Latitude must be within [-90.0, +90.0], got {latitude}")

        self._lat = latitude
        self._lon = normalize_longitude(longitude)
        self._height = height
        self._ellipsoid = ellipsoid
        self._ecef_origin = llh_to_ecef(self._lat, self._lon, self._height, self._ellipsoid)

    @property
    def ref_lat(self) -> float:
        """Reference origin latitude in degrees."""
        return self._lat

    @property
    def ref_lon(self) -> float:
        """Reference origin longitude in degrees."""
        return self._lon

    @property
    def ref_height(self) -> float:
        """Reference origin height in meters."""
        return self._height

    @property
    def ref_ecef(self) -> Tuple[float, float, float]:
        """Reference origin ECEF coordinates (X, Y, Z) in meters."""
        return self._ecef_origin

    def to_enu(self, lat_deg: float, lon_deg: float, h_m: float) -> Tuple[float, float, float]:
        """Convert target LLH coordinates into local ENU displacement (East, North, Up)."""
        return llh_to_enu(
            lat_deg, lon_deg, h_m, self._lat, self._lon, self._height, self._ellipsoid
        )

    def from_enu(self, east: float, north: float, up: float) -> Tuple[float, float, float]:
        """Convert local ENU displacement (East, North, Up) to target LLH coordinates."""
        return enu_to_llh(
            east, north, up, self._lat, self._lon, self._height, self._ellipsoid
        )

    def to_ecef(self, east: float, north: float, up: float) -> Tuple[float, float, float]:
        """Convert local ENU displacement (East, North, Up) to ECEF (X, Y, Z)."""
        return enu_to_ecef(
            east, north, up, self._lat, self._lon, self._height, self._ellipsoid
        )

    def from_ecef(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        """Convert ECEF (X, Y, Z) coordinates to local ENU displacement (East, North, Up)."""
        return ecef_to_enu(
            x, y, z, self._lat, self._lon, self._height, self._ellipsoid
        )

    def ecef_to_enu(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        """Alias for from_ecef."""
        return self.from_ecef(x, y, z)

    def enu_to_ecef(self, east: float, north: float, up: float) -> Tuple[float, float, float]:
        """Alias for to_ecef."""
        return self.to_ecef(east, north, up)

    def __repr__(self) -> str:
        return (
            f"LocalFrame(lat={self._lat:.6f}°, lon={self._lon:.6f}°, "
            f"height={self._height:.2f}m)"
        )
