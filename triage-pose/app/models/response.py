# API response models
"""
Response models for the Triage-Pose API
"""
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any

class AssessmentResponse(BaseModel):
    """Response model for assessment endpoints"""
    assessment_id: str
    status: str
    message: Optional[str] = None
    results: Optional[Dict[str, Any]] = None
    video_url: Optional[str] = None

class ExerciseGuidanceResponse(BaseModel):
    """Response model for exercise guidance endpoints"""
    session_id: str
    status: str
    message: Optional[str] = None
    guidance: Optional[Dict[str, Any]] = None
    video_url: Optional[str] = None

class KeypointData(BaseModel):
    """Keypoint data with coordinates and confidence score"""
    x: float
    y: float
    score: float

class FrameResponse(BaseModel):
    """Response model for a processed frame"""
    keypoints: Dict[str, KeypointData]
    angles: Dict[str, float]
    timestamp: float

class RealtimeResponse(BaseModel):
    """Response model for real-time processing"""
    frame_id: int
    keypoints: Dict[str, KeypointData]
    angles: Dict[str, float]
    rom_data: Optional[Dict[str, Any]] = None