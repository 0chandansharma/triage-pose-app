"""
Video processing service using PhysioTrack
"""
import os
import json
import asyncio
import logging
import traceback
from pathlib import Path
import numpy as np
import cv2
import physiotrack
from typing import Dict, List, Tuple, Any, Optional

from ..models.data import ProcessingOptions, FrameContext
from ..models.response import KeypointData

logger = logging.getLogger(__name__)

class VideoProcessor:
    """Processes videos using PhysioTrack detector"""
    
    def __init__(self, options: ProcessingOptions):
        """Initialize with processing options"""
        self.options = options
        
        # Initialize the pose detector
        self.detector = physiotrack.PoseDetector(
            model_type=options.model_type,
            detection_frequency=options.detection_frequency,
            tracking_mode=options.tracking_mode,
            device=options.device,
            backend=options.backend
        )
        
        # Get keypoint names and IDs
        self.keypoint_names = self.detector.get_keypoint_names()
        self.keypoint_ids = self.detector.get_keypoint_ids()
        
        # Set up angle names
        self.joint_angles = options.joint_angles or [
            'right knee', 'left knee', 'right hip', 'left hip', 
            'right shoulder', 'left shoulder', 'right elbow', 'left elbow'
        ]
        self.segment_angles = options.segment_angles or [
            'right thigh', 'left thigh', 'trunk'
        ]
        self.angle_names = self.joint_angles + self.segment_angles
    
    async def process_video(self, video_path: str, output_dir: str) -> Dict[str, Any]:
        """
        Process a video file and save results
        
        Args:
            video_path: Path to the video file
            output_dir: Directory to save results
            
        Returns:
            Dict: Processing results and status
        """
        # Create status file
        status_file = Path(output_dir) / "status.json"
        with open(status_file, "w") as f:
            json.dump({
                "status": "processing",
                "message": "Starting video processing"
            }, f)
        
        try:
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Set up output video
            assessment_id = Path(output_dir).name
            output_video_path = Path(output_dir) / f"{assessment_id}.mp4"
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_vid = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
            
            # Process frames
            all_keypoints = []
            all_scores = []
            all_angles = []
            frame_times = []
            context = FrameContext()
            
            # Process each frame
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Calculate timestamp
                timestamp = frame_idx / fps
                frame_times.append(timestamp)
                
                # Process frame
                processed_frame, frame_data, context = self.process_frame(frame, context)
                
                # Save processed frame
                if self.options.save_processed_video:
                    out_vid.write(processed_frame)
                
                # Store data
                all_keypoints.append(frame_data['keypoints'])
                all_scores.append(frame_data['scores'])
                all_angles.append(frame_data['angles'])
                
                # Update progress
                if frame_idx % 10 == 0:
                    with open(status_file, "w") as f:
                        json.dump({
                            "status": "processing",
                            "message": f"Processing frame {frame_idx}/{frame_count}",
                            "progress": frame_idx / frame_count
                        }, f)
                
                frame_idx += 1
            
            # Clean up
            cap.release()
            out_vid.release()
            
            # Save angle data
            angles_file = Path(output_dir) / f"{assessment_id}_angles_person00.mot"
            self._save_angles_to_mot(all_angles, frame_times, angles_file)
            
            # Update status
            with open(status_file, "w") as f:
                json.dump({
                    "status": "complete",
                    "message": "Video processing complete"
                }, f)
            
            return {
                "status": "complete",
                "message": "Video processing complete",
                "video_path": str(output_video_path),
                "angles_file": str(angles_file)
            }
            
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Update status with error
            with open(status_file, "w") as f:
                json.dump({
                    "status": "error",
                    "message": f"Error processing video: {str(e)}"
                }, f)
            
            return {
                "status": "error",
                "message": f"Error processing video: {str(e)}"
            }
    
    def process_frame(self, frame: np.ndarray, context: Optional[FrameContext] = None) -> Tuple[np.ndarray, Dict, FrameContext]:
        """
        Process a single frame
        
        Args:
            frame: Input video frame
            context: Optional context from previous frames
            
        Returns:
            Tuple[np.ndarray, Dict, FrameContext]: (processed_frame, result_data, updated_context)
        """
        # Initialize context if not provided
        if context is None:
            context = FrameContext()
        
        # Detect pose
        keypoints, scores = self.detector.detect_pose(frame)
        
        # Track persons across frames
        if context.prev_keypoints is not None and len(keypoints) > 0:
            keypoints, scores = physiotrack.sort_people_physiotrack(
                context.prev_keypoints, keypoints, scores)
        
        # Store for next frame
        context.prev_keypoints = keypoints
        
        # For simplicity, we'll process only the first person
        if len(keypoints) > 0:
            person_keypoints = keypoints[0]
            person_scores = scores[0]
            
            # Calculate angles
            person_angles = physiotrack.calculate_angles(
                person_keypoints, 
                person_scores, 
                self.angle_names,
                self.keypoint_names,
                self.keypoint_ids,
                self.options.keypoint_threshold
            )
            
            # Create simplified keypoints dict for output
            keypoints_dict = {}
            for i, name in enumerate(self.keypoint_names):
                if i < len(person_keypoints) and not np.isnan(person_keypoints[i, 0]):
                    keypoints_dict[name] = KeypointData(
                        x=float(person_keypoints[i, 0]),
                        y=float(person_keypoints[i, 1]),
                        score=float(person_scores[i])
                    )
            
            # Store for context
            context.angles_history.append(person_angles)
            context.keypoints_history.append(keypoints_dict)
            
            # Update running min/max for ROM calculation
            for angle_name, angle_value in person_angles.items():
                if angle_name not in context.running_min:
                    context.running_min[angle_name] = angle_value
                    context.running_max[angle_name] = angle_value
                else:
                    context.running_min[angle_name] = min(context.running_min[angle_name], angle_value)
                    context.running_max[angle_name] = max(context.running_max[angle_name], angle_value)
            
            # Draw visualization
            processed_frame = self._visualize_frame(frame, person_keypoints, person_scores, person_angles)
            
            result_data = {
                'keypoints': person_keypoints,
                'scores': person_scores,
                'angles': person_angles
            }
        else:
            # No person detected
            processed_frame = frame.copy()
            result_data = {
                'keypoints': np.array([]),
                'scores': np.array([]),
                'angles': {}
            }
        
        # Update context
        context.frame_count += 1
        
        return processed_frame, result_data, context
    
    def _visualize_frame(self, frame: np.ndarray, keypoints: np.ndarray, scores: np.ndarray, angles: Dict[str, float]) -> np.ndarray:
        """
        Visualize detection and angles on frame
        
        Args:
            frame: Original frame
            keypoints: Array of keypoint coordinates
            scores: Array of keypoint confidence scores
            angles: Dict of calculated angles
            
        Returns:
            np.ndarray: Visualized frame
        """
        from ..visualization.frame_utils import draw_skeleton, draw_angles_on_frame
        
        # Create a copy of the frame
        vis_frame = frame.copy()
        
        # Draw skeleton
        vis_frame = draw_skeleton(vis_frame, keypoints, scores, threshold=self.options.keypoint_threshold)
        
        # Draw angles
        vis_frame = draw_angles_on_frame(vis_frame, keypoints, angles, self.keypoint_names, self.keypoint_ids)
        
        return vis_frame
    
    def _save_angles_to_mot(self, all_angles: List[Dict[str, float]], frame_times: List[float], output_file: Path):
        """
        Save angles to MOT file
        
        Args:
            all_angles: List of angle dictionaries
            frame_times: List of frame timestamps
            output_file: Path to save the MOT file
        """
        import pandas as pd
        
        # Collect all angle names
        all_angle_names = set()
        for angles in all_angles:
            all_angle_names.update(angles.keys())
        
        # Create DataFrame
        data = {"time": frame_times}
        
        for angle_name in all_angle_names:
            data[angle_name] = [angles.get(angle_name, float('nan')) for angles in all_angles]
        
        df = pd.DataFrame(data)
        
        # Write MOT header
        header = [
            'Coordinates',
            'version=1',
            f'nRows={len(df)}',
            f'nColumns={len(all_angle_names) + 1}',
            'inDegrees=yes',
            '',
            'Units are S.I. units (second, meters, Newtons, ...)',
            "If the header above contains a line with 'inDegrees', this indicates whether rotational values are in degrees (yes) or radians (no).",
            '',
            'endheader',
        ]
        
        with open(output_file, 'w') as f:
            f.write('\n'.join(header) + '\n')
            df.to_csv(f, sep='\t', index=False)