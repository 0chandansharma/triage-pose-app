# API request models
"""
Request models for the Triage-Pose API
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class ROMAssessmentParams(BaseModel):
    """Parameters for ROM assessment"""
    height: float = Field(1.7, description="Subject height in meters")
    visible_side: str = Field("auto", description="Visible side (auto, front, back, left, right, none)")
    time_range: Optional[List[float]] = Field(None, description="Time range for analysis [start, end]")
    joint_angles: Optional[List[str]] = Field(None, description="List of joint angles to analyze")
    segment_angles: Optional[List[str]] = Field(None, description="List of segment angles to analyze")
    model_type: str = Field("body_with_feet", description="Pose model type")
    detection_frequency: int = Field(4, description="Detection frequency (frames)")
    tracking_mode: str = Field("physiotrack", description="Tracking mode")
    device: str = Field("auto", description="Device for inference")
    backend: str = Field("auto", description="Backend for inference")

class ExerciseGuidanceParams(BaseModel):
    """Parameters for exercise guidance"""
    exercise_type: str = Field(..., description="Type of exercise")
    target_reps: int = Field(10, description="Target number of repetitions")
    height: float = Field(1.7, description="Subject height in meters")
    joint_angles: Optional[List[str]] = Field(None, description="List of joint angles to track")
    segment_angles: Optional[List[str]] = Field(None, description="List of segment angles to track")
    
class RealtimeParams(BaseModel):
    """Parameters for real-time processing"""
    model_type: str = Field("body_with_feet", description="Pose model type")
    detection_frequency: int = Field(4, description="Detection frequency (frames)")
    tracking_mode: str = Field("physiotrack", description="Tracking mode")
    device: str = Field("auto", description="Device for inference")
    backend: str = Field("auto", description="Backend for inference")
    joint_angles: Optional[List[str]] = Field(None, description="List of joint angles to analyze")
    segment_angles: Optional[List[str]] = Field(None, description="List of segment angles to analyze")
    height: float = Field(1.7, description="Subject height in meters")