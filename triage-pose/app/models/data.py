# Internal data models
"""
Internal data models for Triage-Pose
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Tuple
import numpy as np

class ProcessingOptions(BaseModel):
    """Options for video/frame processing"""
    model_type: str = "body_with_feet"
    detection_frequency: int = 4
    tracking_mode: str = "physiotrack" 
    device: str = "auto"
    backend: str = "auto"
    keypoint_threshold: float = 0.3
    average_likelihood_threshold: float = 0.5
    keypoint_number_threshold: float = 0.3
    joint_angles: List[str] = []
    segment_angles: List[str] = []
    height: float = 1.7
    visible_side: str = "auto"
    save_processed_video: bool = True
    
    class Config:
        arbitrary_types_allowed = True

class ROMAnalysisOptions(BaseModel):
    """Options for ROM analysis"""
    test_name: str = "default"
    time_window: float = 0.4
    interpolate: bool = True
    filter_type: str = "butterworth"
    filter_order: int = 4
    filter_cutoff: float = 6
    height: float = 1.7
    
    class Config:
        arbitrary_types_allowed = True

class FrameContext:
    """Context object for frame-to-frame processing"""
    def __init__(self):
        self.frame_count = 0
        self.keypoints_history = []
        self.angles_history = []
        self.running_min = {}
        self.running_max = {}
        self.time = 0.0
        self.prev_keypoints = None