"""Core navigation package containing attitude kinematics, INS, dead reckoning, and state management."""

from src.navigation.state import NavigationState
from src.navigation.quaternion import (
    quaternion_normalize,
    quaternion_multiply,
    quaternion_conjugate,
    quaternion_to_rotation_matrix,
    euler_to_quaternion,
    quaternion_to_euler,
    quaternion_to_heading_deg,
    heading_deg_to_yaw_rad,
    gyro_to_quaternion_delta,
    propagate_orientation,
    stationary_attitude_initialization,
)

from src.navigation.ai_fusion import AIESKFPipeline
from src.navigation.pipeline import EndToEndNavigationPipeline, NavigationMode

__all__ = [
    "NavigationState",
    "AIESKFPipeline",
    "EndToEndNavigationPipeline",
    "NavigationMode",
    "quaternion_normalize",
    "quaternion_multiply",
    "quaternion_conjugate",
    "quaternion_to_rotation_matrix",
    "euler_to_quaternion",
    "quaternion_to_euler",
    "quaternion_to_heading_deg",
    "heading_deg_to_yaw_rad",
    "gyro_to_quaternion_delta",
    "propagate_orientation",
    "stationary_attitude_initialization",
]

