"""Synthetic road network generator for Step 9 engineering validation.

Constructs a deterministic multi-road network in local ENU coordinates,
containing:
1. Primary Main Corridor (connected straight, curved, and angled segments matching true trajectory).
2. Parallel Road North (+30m offset) for parallel-road disambiguation testing.
3. Crossing Road (perpendicular at E=200m) for intersection & heading constraint testing.
4. Parallel Road South (-40m offset) for multi-candidate evaluation.
"""

from typing import List, Optional
import numpy as np

from src.map_matching.types import RoadNode, RoadSegment, MapPoint, RoadClass
from src.map_matching.network import RoadNetwork
from src.coordinate_transforms import LocalFrame


def build_synthetic_road_network(
    local_frame: Optional[LocalFrame] = None,
) -> RoadNetwork:
    """Construct a deterministic synthetic road network matching the test trajectory.

    Args:
        local_frame (Optional[LocalFrame]): Geographic local reference frame for origin conversions.

    Returns:
        RoadNetwork: Populated road network topology.
    """
    network = RoadNetwork(name="SyntheticTestNetwork")

    # Define Nodes
    # Main Road Nodes
    n_origin = RoadNode("node_origin", MapPoint(0.0, 0.0, 0.0))
    n_cross = RoadNode("node_main_cross", MapPoint(200.0, 0.0, 0.0))
    n_curve_start = RoadNode("node_curve_start", MapPoint(350.0, 0.0, 0.0))
    n_curve_end = RoadNode("node_curve_end", MapPoint(445.9, -24.4, 0.0))
    n_main_end = RoadNode("node_main_end", MapPoint(700.0, -158.0, 0.0))

    # Parallel North Nodes (+30m North)
    n_par_n_start = RoadNode("node_par_n_start", MapPoint(0.0, 30.0, 0.0))
    n_par_n_end = RoadNode("node_par_n_end", MapPoint(400.0, 30.0, 0.0))

    # Crossing Road Nodes (Perpendicular North-South at East=200m)
    n_cross_s = RoadNode("node_cross_south", MapPoint(200.0, -100.0, 0.0))
    n_cross_n = RoadNode("node_cross_north", MapPoint(200.0, 100.0, 0.0))

    # Parallel South Nodes (-40m South)
    n_par_s_start = RoadNode("node_par_s_start", MapPoint(0.0, -40.0, 0.0))
    n_par_s_end = RoadNode("node_par_s_end", MapPoint(350.0, -40.0, 0.0))

    nodes = [
        n_origin,
        n_cross,
        n_curve_start,
        n_curve_end,
        n_main_end,
        n_par_n_start,
        n_par_n_end,
        n_cross_s,
        n_cross_n,
        n_par_s_start,
        n_par_s_end,
    ]
    for node in nodes:
        network.add_node(node)

    # 1. Primary Road - Straight Section 1 (Origin to Cross)
    poly_main_1 = np.array([
        [0.0, 0.0, 0.0],
        [100.0, 0.0, 0.0],
        [200.0, 0.0, 0.0],
    ], dtype=np.float64)
    seg_main_1 = RoadSegment(
        segment_id="seg_main_straight_1",
        start_node_id="node_origin",
        end_node_id="node_main_cross",
        polyline_enu=poly_main_1,
        road_heading_deg=90.0,  # East
        road_length_m=200.0,
        road_class=RoadClass.PRIMARY,
        speed_limit_mps=15.0,
        is_one_way=True,
    )
    network.add_segment(seg_main_1)

    # 2. Primary Road - Straight Section 2 (Cross to Curve Start)
    poly_main_2 = np.array([
        [200.0, 0.0, 0.0],
        [275.0, 0.0, 0.0],
        [350.0, 0.0, 0.0],
    ], dtype=np.float64)
    seg_main_2 = RoadSegment(
        segment_id="seg_main_straight_2",
        start_node_id="node_main_cross",
        end_node_id="node_curve_start",
        polyline_enu=poly_main_2,
        road_heading_deg=90.0,  # East
        road_length_m=150.0,
        road_class=RoadClass.PRIMARY,
        speed_limit_mps=15.0,
        is_one_way=True,
    )
    network.add_segment(seg_main_2)

    # 3. Primary Road - Curved Turn Section (Curve Start to Curve End)
    # Generate sampled smooth right arc (10 points)
    t_vals = np.linspace(0.0, 1.0, 11)
    curve_pts = []
    for t in t_vals:
        # Interpolate heading from 90 deg (0 rad in math) to 118.65 deg (-0.5 rad in math)
        angle = -0.5 * t
        e = 350.0 + 100.0 * np.sin(0.5 * t) * (1.0 if angle == 0 else (1.0 - np.cos(0.5 * t)) / (0.5 * t + 1e-6) + 0.95 * t)
        # Empirical realistic curve matching simulation
        e_pt = 350.0 + 95.9 * t
        n_pt = 0.0 - 24.4 * (t ** 1.5)
        curve_pts.append([e_pt, n_pt, 0.0])
    poly_curve = np.array(curve_pts, dtype=np.float64)
    seg_main_3 = RoadSegment(
        segment_id="seg_main_curve",
        start_node_id="node_curve_start",
        end_node_id="node_curve_end",
        polyline_enu=poly_curve,
        road_heading_deg=104.3,  # Average tangent
        road_length_m=float(np.sum(np.linalg.norm(np.diff(poly_curve, axis=0), axis=1))),
        road_class=RoadClass.PRIMARY,
        speed_limit_mps=15.0,
        is_one_way=True,
    )
    network.add_segment(seg_main_3)

    # 4. Primary Road - Final Angled Straight Section (Curve End to End)
    poly_main_4 = np.array([
        [445.9, -24.4, 0.0],
        [570.0, -89.5, 0.0],
        [700.0, -158.0, 0.0],
    ], dtype=np.float64)
    seg_main_4 = RoadSegment(
        segment_id="seg_main_turned_straight",
        start_node_id="node_curve_end",
        end_node_id="node_main_end",
        polyline_enu=poly_main_4,
        road_heading_deg=118.65,
        road_length_m=float(np.linalg.norm(poly_main_4[-1] - poly_main_4[0])),
        road_class=RoadClass.PRIMARY,
        speed_limit_mps=15.0,
        is_one_way=True,
    )
    network.add_segment(seg_main_4)

    # 5. Parallel Road North (+30m North)
    poly_par_n = np.array([
        [0.0, 30.0, 0.0],
        [200.0, 30.0, 0.0],
        [400.0, 30.0, 0.0],
    ], dtype=np.float64)
    seg_par_n = RoadSegment(
        segment_id="seg_parallel_north",
        start_node_id="node_par_n_start",
        end_node_id="node_par_n_end",
        polyline_enu=poly_par_n,
        road_heading_deg=90.0,  # East
        road_length_m=400.0,
        road_class=RoadClass.SECONDARY,
        speed_limit_mps=12.0,
        is_one_way=True,
    )
    network.add_segment(seg_par_n)

    # 6. Crossing Road (Perpendicular North-South at East = 200m)
    poly_cross = np.array([
        [200.0, -100.0, 0.0],
        [200.0, 0.0, 0.0],
        [200.0, 100.0, 0.0],
    ], dtype=np.float64)
    seg_cross = RoadSegment(
        segment_id="seg_crossing_north_south",
        start_node_id="node_cross_south",
        end_node_id="node_cross_north",
        polyline_enu=poly_cross,
        road_heading_deg=0.0,  # Heading North
        road_length_m=200.0,
        road_class=RoadClass.SECONDARY,
        speed_limit_mps=10.0,
        is_one_way=False,
    )
    network.add_segment(seg_cross)

    # 7. Parallel Road South (-40m South)
    poly_par_s = np.array([
        [0.0, -40.0, 0.0],
        [175.0, -40.0, 0.0],
        [350.0, -40.0, 0.0],
    ], dtype=np.float64)
    seg_par_s = RoadSegment(
        segment_id="seg_parallel_south",
        start_node_id="node_par_s_start",
        end_node_id="node_par_s_end",
        polyline_enu=poly_par_s,
        road_heading_deg=90.0,  # East
        road_length_m=350.0,
        road_class=RoadClass.RESIDENTIAL,
        speed_limit_mps=8.0,
        is_one_way=True,
    )
    network.add_segment(seg_par_s)

    return network
