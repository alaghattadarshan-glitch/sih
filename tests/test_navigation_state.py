"""Unit tests for NavigationState class."""

import pytest
import numpy as np
from src.navigation.state import NavigationState


def test_navigation_state_initialization():
    """Test NavigationState default initialization and array vector reshaping."""
    state = NavigationState(timestamp=10.0)

    assert state.timestamp == 10.0
    assert state.position_enu.shape == (3,)
    assert state.velocity_enu.shape == (3,)
    assert state.orientation_quaternion.shape == (4,)
    assert pytest.approx(np.linalg.norm(state.orientation_quaternion)) == 1.0


def test_heading_cardinal_directions():
    """Test heading derivation for North, East, South, West orientation quaternions."""
    # Identity quaternion [1, 0, 0, 0] -> Body X = East, Body Y = North
    # East heading = 90 deg
    state_east = NavigationState(
        timestamp=0.0, orientation_quaternion=np.array([1.0, 0.0, 0.0, 0.0])
    )
    assert pytest.approx(state_east.heading_deg(), abs=1e-5) == 90.0

    # Yaw 90 deg (North) -> q = [cos(45°), 0, 0, sin(45°)] = [0.7071, 0, 0, 0.7071]
    # Body X = North -> Heading = 0 deg
    q_north = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])
    state_north = NavigationState(timestamp=0.0, orientation_quaternion=q_north)
    assert pytest.approx(state_north.heading_deg(), abs=1e-4) == 0.0
