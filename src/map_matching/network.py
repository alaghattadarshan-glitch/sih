"""Road network representation and spatial candidate query engine.

Maintains topology of nodes and segments in local metric ENU coordinates,
providing spatial search and connectivity queries.
"""

from typing import Dict, List, Optional
import numpy as np

from src.map_matching.types import RoadNode, RoadSegment, MapPoint
from src.map_matching.geometry import point_to_polyline_projection


class RoadNetwork:
    """In-memory road network topology container."""

    def __init__(self, name: str = "LocalRoadNetwork"):
        """Initialize empty road network.

        Args:
            name (str): Identifying name of road network.
        """
        self.name = name
        self.nodes: Dict[str, RoadNode] = {}
        self.segments: Dict[str, RoadSegment] = {}

    def add_node(self, node: RoadNode):
        """Add a vertex node to the network."""
        self.nodes[node.node_id] = node

    def add_segment(self, segment: RoadSegment):
        """Add a road segment to the network and update node connectivity."""
        self.segments[segment.segment_id] = segment

        # Update start node connectivity
        if segment.start_node_id in self.nodes:
            if segment.segment_id not in self.nodes[segment.start_node_id].connected_segments:
                self.nodes[segment.start_node_id].connected_segments.append(segment.segment_id)

        # Update end node connectivity
        if segment.end_node_id in self.nodes:
            if segment.segment_id not in self.nodes[segment.end_node_id].connected_segments:
                self.nodes[segment.end_node_id].connected_segments.append(segment.segment_id)

    def get_segment(self, segment_id: str) -> Optional[RoadSegment]:
        """Retrieve road segment by ID."""
        return self.segments.get(segment_id, None)

    def get_node(self, node_id: str) -> Optional[RoadNode]:
        """Retrieve road node by ID."""
        return self.nodes.get(node_id, None)

    def get_connected_segments(self, segment_id: str) -> List[RoadSegment]:
        """Return all adjacent road segments connected to start or end node of given segment."""
        seg = self.get_segment(segment_id)
        if seg is None:
            return []

        connected_ids = set()
        if seg.start_node_id in self.nodes:
            connected_ids.update(self.nodes[seg.start_node_id].connected_segments)
        if seg.end_node_id in self.nodes:
            connected_ids.update(self.nodes[seg.end_node_id].connected_segments)

        return [self.segments[sid] for sid in connected_ids if sid in self.segments]

    def query_candidate_segments(
        self, position_enu: np.ndarray, search_radius_m: float = 50.0
    ) -> List[RoadSegment]:
        """Find all road segments within search radius of query position in local ENU frame.

        Performs bounding-box filtering followed by polyline perpendicular distance check.

        Args:
            position_enu (np.ndarray): Query position [East, North, (Up)] in meters.
            search_radius_m (float): Maximum candidate search radius in meters.

        Returns:
            List[RoadSegment]: Road segments with perpendicular distance <= search_radius_m.
        """
        pos = np.asarray(position_enu, dtype=np.float64)
        if pos.ndim == 1 and len(pos) == 2:
            pos = np.array([pos[0], pos[1], 0.0], dtype=np.float64)

        candidates: List[RoadSegment] = []
        p_east, p_north = pos[0], pos[1]

        for seg in self.segments.values():
            poly = seg.polyline_enu
            # 1. Fast AABB Bounding-box test
            min_e = np.min(poly[:, 0]) - search_radius_m
            max_e = np.max(poly[:, 0]) + search_radius_m
            min_n = np.min(poly[:, 1]) - search_radius_m
            max_n = np.max(poly[:, 1]) + search_radius_m

            if not (min_e <= p_east <= max_e and min_n <= p_north <= max_n):
                continue

            # 2. Exact polyline distance check
            _, dist, _, _ = point_to_polyline_projection(pos, poly)
            if dist <= search_radius_m:
                candidates.append(seg)

        return candidates
