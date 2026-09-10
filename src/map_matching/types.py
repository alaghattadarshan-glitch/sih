"""Data structures and types for Road Network and Map Matching abstraction.

Defines provider-independent geometric structures for RoadNode, RoadSegment,
MapPoint, MapCandidate, and MapMatchResult.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, Dict, Any
import numpy as np


class RoadClass(str, Enum):
    """Road functional classification."""
    PRIMARY = "primary"
    SECONDARY = "secondary"
    TERTIARY = "tertiary"
    RESIDENTIAL = "residential"
    HIGHWAY = "highway"
    LINK = "link"
    UNCLASSIFIED = "unclassified"


@dataclass
class MapPoint:
    """3D point representation in local ENU metric coordinates and optional geographic coordinates."""
    east_m: float
    north_m: float
    up_m: float = 0.0
    latitude_deg: Optional[float] = None
    longitude_deg: Optional[float] = None
    altitude_m: Optional[float] = None

    @property
    def enu_array(self) -> np.ndarray:
        """Return 3D ENU vector as numpy array [East, North, Up]."""
        return np.array([self.east_m, self.north_m, self.up_m], dtype=np.float64)


@dataclass
class RoadNode:
    """Road intersection or vertex node."""
    node_id: str
    point: MapPoint
    connected_segments: List[str] = field(default_factory=list)


@dataclass
class RoadSegment:
    """Road segment with polyline geometry in local ENU frame."""
    segment_id: str
    start_node_id: str
    end_node_id: str
    polyline_enu: np.ndarray  # Shape (N, 3) or (N, 2) in meters
    road_heading_deg: float  # Base tangent heading in degrees [0, 360) (0=North, 90=East)
    road_length_m: float
    road_class: RoadClass = RoadClass.PRIMARY
    speed_limit_mps: Optional[float] = None
    is_one_way: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate polyline shape and ensure (N, 3) dimensions."""
        arr = np.asarray(self.polyline_enu, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] < 2:
            raise ValueError(f"RoadSegment {self.segment_id} polyline must have at least 2 vertices, got shape {arr.shape}")
        if arr.shape[1] == 2:
            # Pad Up coordinate with zeros
            z_col = np.zeros((arr.shape[0], 1), dtype=np.float64)
            self.polyline_enu = np.hstack([arr, z_col])
        elif arr.shape[1] == 3:
            self.polyline_enu = arr
        else:
            raise ValueError(f"RoadSegment {self.segment_id} polyline dimension must be 2 or 3, got {arr.shape[1]}")


@dataclass
class MapCandidate:
    """Candidate road segment evaluation."""
    segment: RoadSegment
    projected_point_enu: np.ndarray  # [East, North, Up] on polyline in meters
    distance_to_road_m: float  # Perpendicular distance from estimated position to polyline
    road_heading_deg: float  # Segment tangent heading at projected point [0, 360)
    heading_error_deg: float  # Angular error wrap(heading_vehicle - heading_road) [-180, 180]
    along_track_dist_m: float  # Distance from segment start to projected point in meters
    position_score: float  # Likelihood score from perpendicular distance [0, 1]
    heading_score: float  # Likelihood score from heading alignment [0, 1]
    motion_score: float  # Likelihood score from vehicle motion constraint [0, 1]
    transition_score: float  # Continuity score from previous matched segment [0, 1]
    total_score: float  # Combined weighted score
    confidence: float  # Normalized probability across candidate set [0, 1]


@dataclass
class MapMatchResult:
    """Result of map matching search and candidate evaluation."""
    timestamp: float
    is_matched: bool
    selected_candidate: Optional[MapCandidate] = None
    all_candidates: List[MapCandidate] = field(default_factory=list)
    confidence: float = 0.0
    projected_position_enu: Optional[np.ndarray] = None
    distance_to_road_m: Optional[float] = None
    heading_error_deg: Optional[float] = None
    gated_out: bool = False
    gating_reason: Optional[str] = None
