# Pose processing workflow
# app/processors/pose_processor.py
from typing import Dict, List, Tuple, Any
import numpy as np
import cv2
import physiotrack
from ..models.data import ProcessingOptions, FrameContext

class FrameProcessor:
    """Processes individual frames for pose and angles"""
    
    def __init__(self, options: ProcessingOptions):
        """Initialize with processing options"""
        self.options = options
        self.detector = physiotrack.PoseDetector(
            model_type=options.model_type,
            detection_frequency=options.detection_frequency,
            tracking_mode=options.tracking_mode,
            device=options.device
        )
        self.keypoint_names = self.detector.get_keypoint_names()
    
    def process_frame(self, frame: np.ndarray, context: FrameContext = None) -> Tuple[np.ndarray, Dict, FrameContext]:
        """Process a single frame
        
        Args:
            frame: Video frame
            context: Optional context from previous frames
            
        Returns:
            Tuple: (processed_frame, result_data, updated_context)
        """
        # Detect pose
        keypoints, scores = self.detector.detect_pose(frame)
        
        # Filter by confidence
        keypoints, scores = physiotrack.filter_low_confidence_keypoints(
            keypoints, scores, self.options.keypoint_threshold
        )
        
        # Calculate angles using PhysioTrack
        angles = physiotrack.calculate_angles(
            keypoints, scores, self.options.angle_names, self.options.keypoint_threshold
        )
        
        # Visualization (moved from PhysioTrack to here)
        processed_frame = self._visualize_results(frame, keypoints, scores, angles)
        
        # Update context
        updated_context = context or FrameContext()
        updated_context.frame_count += 1
        updated_context.keypoints_history.append(keypoints)
        updated_context.angles_history.append(angles)
        
        return processed_frame, {
            'keypoints': keypoints,
            'scores': scores,
            'angles': angles
        }, updated_context
    
    def _visualize_results(self, frame, keypoints, scores, angles):
        """Create visualization of results on frame"""
        # Visualization logic moved from PhysioTrack