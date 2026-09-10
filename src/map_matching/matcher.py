"""Map Matching Engine with Multi-Hypothesis Candidate Scoring and Temporal Continuity.

Evaluates candidate road segments based on metric distance, heading alignment,
non-holonomic vehicle motion constraints, and topological temporal continuity.
"""

from dataclasses import dataclass, field
import math
from typing import List, Optional, Tuple
import numpy as np

from src.navigation.state import NavigationState
from src.map_matching.types import RoadSegment, MapCandidate, MapMatchResult
from src.map_matching.geometry import (
    point_to_polyline_projection,
    wrap_angle_180,
    calculate_segment_heading,
)
from src.map_matching.network import RoadNetwork


@dataclass
class MapMatcherConfig:
    """Configuration parameters for Map Matching candidate evaluation and scoring."""
    search_radius_m: float = 50.0
    max_heading_error_deg: float = 60.0
    min_confidence_threshold: float = 0.25
    sigma_dist_m: float = 10.0
    sigma_heading_deg: float = 25.0
    sigma_transition_m: float = 15.0
    w_distance: float = 0.40
    w_heading: float = 0.35
    w_transition: float = 0.15
    w_motion: float = 0.10


class MapMatcher:
    """Multi-hypothesis Map Matcher evaluating candidate roads against navigation state."""

    def __init__(self, config: Optional[MapMatcherConfig] = None):
        """Initialize map matcher.

        Args:
            config (Optional[MapMatcherConfig]): Scoring weights and tolerances.
        """
        self.config = config if config is not None else MapMatcherConfig()
        self.last_matched_segment_id: Optional[str] = None
        self.last_matched_pos_enu: Optional[np.ndarray] = None
        self.match_history: List[MapMatchResult] = []

    def reset(self):
        """Reset temporal continuity state."""
        self.last_matched_segment_id = None
        self.last_matched_pos_enu = None
        self.match_history.clear()

    def match(
        self,
        state: NavigationState,
        network: RoadNetwork,
    ) -> MapMatchResult:
        """Evaluate candidate roads and select the most probable map match.

        Args:
            state (NavigationState): Current instantaneous navigation state.
            network (RoadNetwork): Road network topology in local ENU frame.

        Returns:
            MapMatchResult: Match result containing best candidate, confidence, and diagnostics.
        """
        pos_enu = state.position_enu
        heading_veh = state.heading_deg()
        speed = state.speed

        # Query candidate segments within search radius
        candidate_segments = network.query_candidate_segments(
            pos_enu, search_radius_m=self.config.search_radius_m
        )

        if not candidate_segments:
            res = MapMatchResult(
                timestamp=state.timestamp,
                is_matched=False,
                gated_out=True,
                gating_reason="No candidate segments within search radius",
            )
            self.match_history.append(res)
            return res

        # Evaluate each candidate segment
        candidates: List[MapCandidate] = []

        # Get connected segments from last match for continuity
        connected_ids = set()
        if self.last_matched_segment_id is not None:
            connected_segments = network.get_connected_segments(self.last_matched_segment_id)
            connected_ids = {s.segment_id for s in connected_segments}

        for seg in candidate_segments:
            proj_pt, dist_perp, along_track, road_heading = point_to_polyline_projection(
                pos_enu, seg.polyline_enu
            )

            # Heading difference [-180, 180]
            heading_diff = abs(wrap_angle_180(heading_veh - road_heading))

            # Discard immediately if heading discrepancy is too extreme
            if heading_diff > self.config.max_heading_error_deg:
                continue

            # 1. Distance score (Gaussian likelihood)
            s_dist = math.exp(-0.5 * (dist_perp / self.config.sigma_dist_m) ** 2)

            # 2. Heading score (Gaussian likelihood)
            s_head = math.exp(-0.5 * (heading_diff / self.config.sigma_heading_deg) ** 2)

            # 3. Temporal transition / continuity score
            if self.last_matched_segment_id is None:
                s_trans = 1.0
            elif seg.segment_id == self.last_matched_segment_id:
                s_trans = 1.0
            elif seg.segment_id in connected_ids:
                s_trans = 0.85
            else:
                # Penalty for jumping across unconnected segments
                if self.last_matched_pos_enu is not None:
                    jump_dist = float(np.linalg.norm(proj_pt - self.last_matched_pos_enu))
                    s_trans = 0.30 * math.exp(-0.5 * (jump_dist / self.config.sigma_transition_m) ** 2)
                else:
                    s_trans = 0.30

            # 4. Non-holonomic vehicle motion constraint score
            # When moving, body velocity lateral component should be small relative to road alignment
            if speed > 1.0:
                v_body = state.rotation_matrix.T @ state.velocity_enu
                v_lat = abs(v_body[1])  # Body Y is lateral
                s_motion = math.exp(-0.5 * (v_lat / 1.5) ** 2)
            else:
                s_motion = 1.0

            # Total composite score
            total_score = (
                self.config.w_distance * s_dist
                + self.config.w_heading * s_head
                + self.config.w_transition * s_trans
                + self.config.w_motion * s_motion
            )

            cand = MapCandidate(
                segment=seg,
                projected_point_enu=proj_pt,
                distance_to_road_m=dist_perp,
                road_heading_deg=road_heading,
                heading_error_deg=heading_diff,
                along_track_dist_m=along_track,
                position_score=s_dist,
                heading_score=s_head,
                motion_score=s_motion,
                transition_score=s_trans,
                total_score=total_score,
                confidence=0.0,  # Computed below after normalizing
            )
            candidates.append(cand)

        if not candidates:
            res = MapMatchResult(
                timestamp=state.timestamp,
                is_matched=False,
                gated_out=True,
                gating_reason="All candidate segments exceeded heading error threshold",
            )
            self.match_history.append(res)
            return res

        # Normalize confidence across valid candidates
        sum_scores = sum(c.total_score for c in candidates)
        for c in candidates:
            c.confidence = c.total_score / sum_scores if sum_scores > 1e-12 else 0.0

        # Sort descending by total score
        candidates.sort(key=lambda c: c.total_score, reverse=True)
        best_cand = candidates[0]

        # Check confidence threshold
        if best_cand.confidence < self.config.min_confidence_threshold:
            res = MapMatchResult(
                timestamp=state.timestamp,
                is_matched=False,
                selected_candidate=best_cand,
                all_candidates=candidates,
                confidence=best_cand.confidence,
                projected_position_enu=best_cand.projected_point_enu,
                distance_to_road_m=best_cand.distance_to_road_m,
                heading_error_deg=best_cand.heading_error_deg,
                gated_out=True,
                gating_reason=f"Candidate confidence {best_cand.confidence:.3f} below threshold {self.config.min_confidence_threshold:.3f}",
            )
            self.match_history.append(res)
            return res

        # Valid map match found
        self.last_matched_segment_id = best_cand.segment.segment_id
        self.last_matched_pos_enu = best_cand.projected_point_enu.copy()

        res = MapMatchResult(
            timestamp=state.timestamp,
            is_matched=True,
            selected_candidate=best_cand,
            all_candidates=candidates,
            confidence=best_cand.confidence,
            projected_position_enu=best_cand.projected_point_enu,
            distance_to_road_m=best_cand.distance_to_road_m,
            heading_error_deg=best_cand.heading_error_deg,
            gated_out=False,
        )
        self.match_history.append(res)
        return res
