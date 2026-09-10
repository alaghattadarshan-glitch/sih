"""WGS84 Reference Ellipsoid Model and Geodetic Constants.

This module centralizes all WGS84 reference constants and radius of curvature
equations to prevent scattering constants throughout the codebase.
"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class WGS84Ellipsoid:
    """WGS84 Reference Ellipsoid parameters.

    Attributes:
        a (float): Semi-major axis (equatorial radius) in meters.
        f (float): Flattening factor.
        b (float): Semi-minor axis (polar radius) in meters.
        e2 (float): First eccentricity squared (e^2).
        ep2 (float): Second eccentricity squared (e'^2).
    """

    a: float = 6378137.0  # meters
    f_inv: float = 298.257223563  # inverse flattening

    @property
    def f(self) -> float:
        """Flattening factor f."""
        return 1.0 / self.f_inv

    @property
    def b(self) -> float:
        """Semi-minor axis b in meters."""
        return self.a * (1.0 - self.f)

    @property
    def e2(self) -> float:
        """First eccentricity squared (e^2 = (a^2 - b^2) / a^2 = 2f - f^2)."""
        f = self.f
        return 2.0 * f - f * f

    @property
    def ep2(self) -> float:
        """Second eccentricity squared (e'^2 = (a^2 - b^2) / b^2)."""
        a2 = self.a ** 2
        b2 = self.b ** 2
        return (a2 - b2) / b2

    def prime_vertical_radius(self, lat_rad: float) -> float:
        """Calculate Radius of Curvature in the Prime Vertical N(phi).

        Args:
            lat_rad (float): Geodetic latitude in radians.

        Returns:
            float: Prime vertical radius of curvature N in meters.
        """
        sin_lat = math.sin(lat_rad)
        return self.a / math.sqrt(1.0 - self.e2 * (sin_lat ** 2))


# Default global WGS84 instance
WGS84 = WGS84Ellipsoid()
