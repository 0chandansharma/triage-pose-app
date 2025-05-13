"""
PhysioTrack: Minimal library for pose detection and angle calculation
"""
from .detector import PoseDetector
from .angles import calculate_angles, calculate_joint_angle
from .models import KeypointData, PoseData, DetectionConfig, ANGLE_DEFINITIONS
from .utils import normalize_keypoints, filter_low_confidence_keypoints

__version__ = "0.1.0"

# Make the key components available at the package level
__all__ = [
    'PoseDetector',
    'calculate_angles',
    'calculate_joint_angle',
    'KeypointData', 
    'PoseData',
    'DetectionConfig',
    'ANGLE_DEFINITIONS',
    'normalize_keypoints',
    'filter_low_confidence_keypoints'
]