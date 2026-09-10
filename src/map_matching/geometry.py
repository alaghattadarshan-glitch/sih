"""Geometry and projection utilities for local ENU road polylines.

Handles point-to-segment and point-to-polyline projections, distance computations,
tangent heading derivations, and angular wrap-around calculations.
"""

import math
from typing import Tuple
import numpy as np


def wrap_angle_180(angle_deg: float) -> float:
    """Wrap angle in degrees to range [-180.0, +180.0].

    Args:
        angle_deg (float): Input angle in degrees.

    Returns:
        float: Wrapped angle in [-180.0, 180.0] degrees.
    """
    wrapped = (angle_deg + 180.0) % 360.0 - 180.0
    return float(wrapped)


def wrap_angle_360(angle_deg: float) -> float:
    """Wrap angle in degrees to range [0.0, 360.0).

    Args:
        angle_deg (float): Input angle in degrees.

    Returns:
        float: Wrapped angle in [0.0, 360.0) degrees.
    """
    return float(angle_deg % 360.0)


def calculate_segment_heading(p_start: np.ndarray, p_end: np.ndarray) -> float:
    """Compute geographic heading azimuth (0°=North, 90°=East) from p_start to p_end in ENU.

    ENU convention:
    - East is X (index 0)
    - North is Y (index 1)
    - Heading azimuth = (90° - atan2(dNorth, dEast)) % 360°

    Args:
        p_start (np.ndarray): Start point [East, North, (Up)].
        p_end (np.ndarray): End point [East, North, (Up)].

    Returns:
        float: Geographic heading azimuth in degrees [0.0, 360.0).
    """
    d_east = float(p_end[0] - p_start[0])
    d_north = float(p_end[1] - p_start[1])

    if abs(d_east) < 1e-9 and abs(d_north) < 1e-9:
        return 0.0

    yaw_rad = math.atan2(d_north, d_east)  # 0=East, pi/2=North
    heading_deg = math.degrees(math.pi / 2.0 - yaw_rad) % 360.0
    return float(heading_deg)


def point_to_line_segment_projection(
    p: np.ndarray, a: np.ndarray, b: np.ndarray
) -> Tuple[np.ndarray, float, float]:
    """Project point p onto line segment ab in 3D/2D Cartesian space.

    Args:
        p (np.ndarray): Query point [East, North, Up].
        a (np.ndarray): Segment start vertex [East, North, Up].
        b (np.ndarray): Segment end vertex [East, North, Up].

    Returns:
        Tuple[np.ndarray, float, float]:
            - Projected point on segment [East, North, Up].
            - Projection ratio t in [0.0, 1.0] along segment ab.
            - Euclidean distance from query point p to projected point in meters.
    """
    ab = b - a
    ab_sq = float(np.dot(ab, ab))

    if ab_sq < 1e-12:
        # Degenerate zero-length segment
        dist = float(np.linalg.norm(p - a))
        return a.copy(), 0.0, dist

    ap = p - a
    t = float(np.dot(ap, ab) / ab_sq)
    t_clamped = max(0.0, min(1.0, t))

    proj_point = a + t_clamped * ab
    dist = float(np.linalg.norm(p - proj_point))
    return proj_point, t_clamped, dist


def point_to_polyline_projection(
    p: np.ndarray, polyline_enu: np.ndarray
) -> Tuple[np.ndarray, float, float, float]:
    """Project point p onto a polyline composed of N vertices in local ENU coordinates.

    Finds the closest line segment along the polyline, computes the projected point,
    perpendicular distance, cumulative along-track distance from start of polyline,
    and tangent heading at the projected location.

    Args:
        p (np.ndarray): Query point [East, North, Up] in meters.
        polyline_enu (np.ndarray): (N, 3) or (N, 2) array of polyline vertices.

    Returns:
        Tuple[np.ndarray, float, float, float]:
            - Projected point on polyline [East, North, Up] in meters.
            - Perpendicular distance from p to polyline in meters.
            - Along-track cumulative distance along polyline to projected point in meters.
            - Local tangent heading azimuth in degrees [0, 360) at projected location.
    """
    p = np.asarray(p, dtype=np.float64)
    if p.ndim == 1 and len(p) == 2:
        p = np.array([p[0], p[1], 0.0], dtype=np.float64)

    verts = np.asarray(polyline_enu, dtype=np.float64)
    if verts.shape[1] == 2:
        z_col = np.zeros((verts.shape[0], 1), dtype=np.float64)
        verts = np.hstack([verts, z_col])

    num_verts = verts.shape[0]
    if num_verts < 2:
        raise ValueError("Polyline must have at least 2 vertices.")

    best_proj = verts[0].copy()
    min_dist = float("inf")
    best_along_track = 0.0
    best_heading = 0.0

    cumulative_dist = 0.0

    for i in range(num_verts - 1):
        v_start = verts[i]
        v_end = verts[i + 1]
        seg_len = float(np.linalg.norm(v_end - v_start))

        proj_pt, t_ratio, dist = point_to_line_segment_projection(p, v_start, v_end)

        if dist < min_dist:
            min_dist = dist
            best_proj = proj_pt
            best_along_track = cumulative_dist + t_ratio * seg_len
            best_heading = calculate_segment_heading(v_start, v_end)

        cumulative_dist += seg_len

    return best_proj, min_dist, best_along_track, best_heading
