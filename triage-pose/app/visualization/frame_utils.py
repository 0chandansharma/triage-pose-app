# Frame annotation and drawing
"""
Utilities for frame visualization
"""
import cv2
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

def draw_skeleton(frame: np.ndarray, 
                 keypoints: np.ndarray, 
                 scores: np.ndarray, 
                 threshold: float = 0.3, 
                 thickness: int = 2) -> np.ndarray:
    """
    Draw skeleton on frame
    
    Args:
        frame: Original frame
        keypoints: Keypoint coordinates
        scores: Keypoint confidence scores
        threshold: Confidence threshold
        thickness: Line thickness
        
    Returns:
        np.ndarray: Frame with skeleton drawn
    """
    # Define connections for basic skeleton
    connections = [
        # Legs
        (12, 14), (14, 16), (11, 13), (13, 15),
        # Torso
        (11, 12), (5, 11), (6, 12), (5, 6),
        # Arms
        (5, 7), (7, 9), (6, 8), (8, 10),
        # Face
        (0, 17), (17, 18)
        ]
    
    # Create a copy of the frame
    vis_frame = frame.copy()
    
    # Draw connections
    for connection in connections:
        pt1_idx, pt2_idx = connection
        # Check if both points are valid
        if (pt1_idx < len(keypoints) and pt2_idx < len(keypoints) and 
            scores[pt1_idx] >= threshold and scores[pt2_idx] >= threshold):
            pt1 = tuple(map(int, keypoints[pt1_idx]))
            pt2 = tuple(map(int, keypoints[pt2_idx]))
            cv2.line(vis_frame, pt1, pt2, (0, 255, 0), thickness)
    
    # Draw keypoints
    for i, (pt, score) in enumerate(zip(keypoints, scores)):
        if score >= threshold:
            x, y = map(int, pt)
            # Color based on confidence (green to red)
            color_val = min(int(score * 255), 255)
            color = (0, color_val, 255 - color_val)
            cv2.circle(vis_frame, (x, y), 5, color, -1)
    
    return vis_frame

def draw_angles_on_frame(frame: np.ndarray, 
                        keypoints: np.ndarray, 
                        angles: Dict[str, float], 
                        keypoint_names: List[str],
                        keypoint_ids: List[int],
                        show_labels: bool = True,
                        thickness: int = 2) -> np.ndarray:
    """
    Draw angles on frame
    
    Args:
        frame: Original frame
        keypoints: Keypoint coordinates
        angles: Dictionary of angle names and values
        keypoint_names: List of keypoint names
        keypoint_ids: List of keypoint IDs
        show_labels: Whether to show angle labels
        thickness: Line thickness
        
    Returns:
        np.ndarray: Frame with angles drawn
    """
    # Create a copy of the frame
    vis_frame = frame.copy()
    
    # Define colors for different angle types
    joint_color = (0, 255, 0)    # Green
    segment_color = (255, 0, 0)  # Blue
    
    # Define the mappings for keypoints used in angle calculations
    angle_points = {
        'right knee': ['RAnkle', 'RKnee', 'RHip'],
        'left knee': ['LAnkle', 'LKnee', 'LHip'],
        'right hip': ['RKnee', 'RHip', 'Hip', 'Neck'],
        'left hip': ['LKnee', 'LHip', 'Hip', 'Neck'],
        'trunk': ['Neck', 'Hip']
    }
    
    for angle_name, angle_value in angles.items():
        # Skip if no mapping available
        if angle_name not in angle_points:
            continue
        
        # Get the keypoint names for this angle
        point_names = angle_points[angle_name]
        
        # Get the keypoint indices
        point_indices = []
        all_points_valid = True
        
        for point_name in point_names:
            if point_name in keypoint_names:
                idx = keypoint_names.index(point_name)
                if idx < len(keypoint_ids):
                    point_indices.append(keypoint_ids[idx])
                else:
                    all_points_valid = False
                    break
            else:
                all_points_valid = False
                break
        
        if not all_points_valid or any(idx >= len(keypoints) for idx in point_indices):
            continue
        
        # Draw the angle
        if len(point_indices) == 2:  # Segment angle
            # Draw line for segment
            pt1 = tuple(map(int, keypoints[point_indices[0]]))
            pt2 = tuple(map(int, keypoints[point_indices[1]]))
            cv2.line(vis_frame, pt1, pt2, segment_color, thickness)
            
            # Draw angle at midpoint
            mid_x = (pt1[0] + pt2[0]) // 2
            mid_y = (pt1[1] + pt2[1]) // 2
            
            if show_labels:
                # Draw angle value
                text = f"{angle_value:.1f}°"
                cv2.putText(vis_frame, text, (mid_x, mid_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(vis_frame, text, (mid_x, mid_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, segment_color, 1, cv2.LINE_AA)
                
        elif len(point_indices) == 3:  # Joint angle
            # Get points
            pt1 = tuple(map(int, keypoints[point_indices[0]]))
            pt2 = tuple(map(int, keypoints[point_indices[1]]))  # Vertex of angle
            pt3 = tuple(map(int, keypoints[point_indices[2]]))
            
            # Draw lines
            cv2.line(vis_frame, pt1, pt2, joint_color, thickness)
            cv2.line(vis_frame, pt2, pt3, joint_color, thickness)
            
            if show_labels:
                # Draw angle value
                text = f"{angle_value:.1f}°"
                # Position text away from joint
                vec1 = np.array(pt1) - np.array(pt2)
                vec2 = np.array(pt3) - np.array(pt2)
                vec_sum = vec1 + vec2
                length = np.linalg.norm(vec_sum)
                if length > 0:
                    unit_vec = vec_sum / length
                    text_pos = tuple(map(int, np.array(pt2) + unit_vec * 30))
                    cv2.putText(vis_frame, text, text_pos, 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(vis_frame, text, text_pos, 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, joint_color, 1, cv2.LINE_AA)
    
    return vis_frame

def draw_status_panel(frame: np.ndarray, 
                     rom_data: Dict[str, Any], 
                     position: Tuple[int, int] = (10, 30),
                     font_scale: float = 0.6,
                     thickness: int = 1) -> np.ndarray:
    """
    Draw status panel on frame with ROM data
    
    Args:
        frame: Original frame
        rom_data: ROM data dictionary
        position: Top-left position of the panel
        font_scale: Font scale
        thickness: Line thickness
        
    Returns:
        np.ndarray: Frame with status panel
    """
    # Create a copy of the frame
    vis_frame = frame.copy()
    
    # Create semi-transparent background
    overlay = vis_frame.copy()
    panel_width = 300
    panel_height = 200
    cv2.rectangle(overlay, position, 
                 (position[0] + panel_width, position[1] + panel_height),
                 (0, 0, 0), -1)
    alpha = 0.6
    vis_frame = cv2.addWeighted(overlay, alpha, vis_frame, 1 - alpha, 0)
    
    # Draw ROM data
    y_offset = position[1] + 30
    cv2.putText(vis_frame, "ROM Analysis", (position[0] + 10, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    y_offset += 25
    
    for angle_name, data in rom_data.items():
        if angle_name == 'trunk':  # Highlight trunk angle
            color = (0, 255, 255)  # Yellow
        else:
            color = (255, 255, 255)  # White
            
        text = f"{angle_name}: {data['range']:.1f}° ({data['min']:.1f}° to {data['max']:.1f}°)"
        cv2.putText(vis_frame, text, (position[0] + 10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
        y_offset += 25
    
    return vis_frame