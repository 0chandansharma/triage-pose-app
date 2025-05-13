"""
Core angle calculation functionality
"""
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
from .models import ANGLE_DEFINITIONS

def points_to_angles(points_list):
    """
    Calculate angles between points
    
    If len(points_list)==2, computes clockwise angle of ab vector w.r.t. horizontal
    If len(points_list)==3, computes clockwise angle from a to c around b
    If len(points_list)==4, computes clockwise angle between vectors ab and cd
    
    Args:
        points_list: List of coordinates [x, y]
        
    Returns:
        float: Angle in degrees
    """
    if len(points_list) < 2:
        return np.nan
    
    points_array = np.array(points_list)
    dimensions = points_array.shape[-1]

    if len(points_list) == 2:
        vector_u = points_array[0] - points_array[1]
        if len(points_array.shape)==2:
            vector_v = np.array([1, 0, 0]) if dimensions == 3 else np.array([1, 0]) # Horizontal vector
        else:
            vector_v = np.array([[1, 0, 0],] * points_array.shape[1]) if dimensions == 3 else np.array([[1, 0],] * points_array.shape[1])

    elif len(points_list) == 3:
        vector_u = points_array[0] - points_array[1]
        vector_v = points_array[2] - points_array[1]

    elif len(points_list) == 4:
        vector_u = points_array[1] - points_array[0]
        vector_v = points_array[3] - points_array[2]
        
    else:
        return np.nan

    if dimensions == 2: 
        vector_u = vector_u[:2]
        vector_v = vector_v[:2]
        ang = np.arctan2(vector_u[1], vector_u[0]) - np.arctan2(vector_v[1], vector_v[0])
    else:
        cross_product = np.cross(vector_u, vector_v)
        dot_product = np.einsum('ij,ij->i', vector_u, vector_v)
        ang = np.arctan2(np.linalg.norm(cross_product, axis=1), dot_product)

    ang_deg = np.degrees(ang)
    return ang_deg

def fixed_angles(points_list, ang_name):
    """
    Apply offset and scaling to calculated angles based on angle definitions
    
    Args:
        points_list: List of point coordinates
        ang_name: Name of the angle (must be in ANGLE_DEFINITIONS)
        
    Returns:
        float: Adjusted angle in degrees
    """
    ang_params = ANGLE_DEFINITIONS.get(ang_name.lower())
    if not ang_params:
        raise ValueError(f"Angle {ang_name} not found in angle definitions")
        
    ang = points_to_angles(points_list)
    ang += ang_params[2]  # Add offset
    ang *= ang_params[3]  # Apply scaling factor
    
    if ang_name.lower() in ['pelvis', 'shoulders']:
        ang = np.where(ang>90, ang-180, ang)
        ang = np.where(ang<-90, ang+180, ang)
    else:
        ang = np.where(ang>180, ang-360, ang)
        ang = np.where(ang<-180, ang+360, ang)

    return ang

def calculate_joint_angle(points: List[np.ndarray], angle_type: str = "flexion") -> float:
    """
    Calculate angle between points
    
    Args:
        points: List of point coordinates
        angle_type: Type of angle calculation
        
    Returns:
        float: Calculated angle in degrees
    """
    return points_to_angles(points)

def calculate_angles(keypoints: np.ndarray, 
                    scores: np.ndarray,
                    angle_names: List[str],
                    keypoints_names: List[str],
                    keypoints_ids: List[int],
                    threshold: float = 0.3) -> Dict[str, float]:
    """
    Calculate multiple angles from keypoints
    
    Args:
        keypoints: Array of keypoint coordinates (shape [N, 2] for N keypoints)
        scores: Array of keypoint confidence scores (shape [N])
        angle_names: List of angle names to calculate
        keypoints_names: List of keypoint names corresponding to keypoints array
        keypoints_ids: List of keypoint IDs corresponding to keypoints array
        threshold: Confidence threshold for keypoints
        
    Returns:
        Dict[str, float]: Dictionary of angle names and values
    """
    # Initialize result
    result = {}
    
    # Filter keypoints by confidence
    valid_mask = scores >= threshold
    person_X = np.where(valid_mask[:, np.newaxis], keypoints, np.nan)[:, 0]
    person_Y = np.where(valid_mask[:, np.newaxis], keypoints, np.nan)[:, 1]
    
    # Add Neck and Hip if not provided
    new_keypoints_names, new_keypoints_ids = keypoints_names.copy(), keypoints_ids.copy()
    person_X_with_additions = person_X.copy()
    person_Y_with_additions = person_Y.copy()
    
    # Helper function to add missing keypoints
    def add_neck_hip_coords(kpt_name, p_X, p_Y):
        if kpt_name == 'Neck':
            # Average of left and right shoulder
            l_shoulder_idx = keypoints_ids[keypoints_names.index('LShoulder')]
            r_shoulder_idx = keypoints_ids[keypoints_names.index('RShoulder')]
            mid_X = (p_X[l_shoulder_idx] + p_X[r_shoulder_idx]) / 2
            mid_Y = (p_Y[l_shoulder_idx] + p_Y[r_shoulder_idx]) / 2
        elif kpt_name == 'Hip':
            # Average of left and right hip
            l_hip_idx = keypoints_ids[keypoints_names.index('LHip')]
            r_hip_idx = keypoints_ids[keypoints_names.index('RHip')]
            mid_X = (p_X[l_hip_idx] + p_X[r_hip_idx]) / 2
            mid_Y = (p_Y[l_hip_idx] + p_Y[r_hip_idx]) / 2
        else:
            raise ValueError(f"Unknown keypoint: {kpt_name}")
            
        return mid_X, mid_Y
    
    for kpt in ['Neck', 'Hip']:
        if kpt not in new_keypoints_names:
            try:
                mid_X, mid_Y = add_neck_hip_coords(kpt, person_X_with_additions, person_Y_with_additions)
                new_keypoints_names.append(kpt)
                new_id = len(person_X_with_additions)
                new_keypoints_ids.append(new_id)
                
                # Extend arrays
                person_X_with_additions = np.append(person_X_with_additions, mid_X)
                person_Y_with_additions = np.append(person_Y_with_additions, mid_Y)
            except (ValueError, IndexError) as e:
                # Skip if required keypoints not found
                continue
    
    # Calculate each requested angle
    for ang_name in angle_names:
        ang_params = ANGLE_DEFINITIONS.get(ang_name.lower())
        if not ang_params:
            continue
            
        kpts = ang_params[0]
        if any(item not in new_keypoints_names for item in kpts):
            # Skip angles that require keypoints we don't have
            continue
            
        try:
            # Get coordinates for each keypoint
            pts_for_angles = []
            for pt in kpts:
                idx = new_keypoints_ids[new_keypoints_names.index(pt)]
                pts_for_angles.append([person_X_with_additions[idx], person_Y_with_additions[idx]])
                
            # Calculate the angle
            ang = fixed_angles(pts_for_angles, ang_name)
            result[ang_name] = float(ang)
        except Exception as e:
            # Skip on error
            continue
    
    return result