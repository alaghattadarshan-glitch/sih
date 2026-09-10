"""Road network and Map Matching package for SIH26168.

Provides provider-independent road network representation, polyline geometry,
multi-hypothesis candidate scoring, and synthetic road networks.
"""

from src.map_matching.types import (
    RoadClass,
    MapPoint,
    RoadNode,
    RoadSegment,
    MapCandidate,
    MapMatchResult,
)
from src.map_matching.geometry import (
    wrap_angle_180,
    wrap_angle_360,
    calculate_segment_heading,
    point_to_line_segment_projection,
    point_to_polyline_projection,
)
from src.map_matching.network import RoadNetwork
from src.map_matching.matcher import MapMatcher, MapMatcherConfig
from src.map_matching.synthetic_network import build_synthetic_road_network

__all__ = [
    "RoadClass",
    "MapPoint",
    "RoadNode",
    "RoadSegment",
    "MapCandidate",
    "MapMatchResult",
    "wrap_angle_180",
    "wrap_angle_360",
    "calculate_segment_heading",
    "point_to_line_segment_projection",
    "point_to_polyline_projection",
    "RoadNetwork",
    "MapMatcher",
    "MapMatcherConfig",
    "build_synthetic_road_network",
]
