"""
Utility functions for PhysioTrack
"""
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

def normalize_keypoints(keypoints: np.ndarray, 
                       image_width: int, 
                       image_height: int) -> np.ndarray:
    """
    Normalize keypoint coordinates to [0,1] range
    
    Args:
        keypoints: Array of keypoint coordinates [N, 2]
        image_width: Width of the image
        image_height: Height of the image
        
    Returns:
        np.ndarray: Normalized keypoints
    """
    normalized = keypoints.copy()
    normalized[:, 0] /= image_width
    normalized[:, 1] /= image_height
    return normalized

def filter_low_confidence_keypoints(keypoints: np.ndarray, 
                                   scores: np.ndarray, 
                                   threshold: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter out keypoints with low confidence scores
    
    Args:
        keypoints: Array of keypoint coordinates [N, 2]
        scores: Array of keypoint confidence scores [N]
        threshold: Confidence threshold
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Filtered keypoints and scores
    """
    mask = scores >= threshold
    filtered_keypoints = np.where(mask[:, np.newaxis], keypoints, np.nan)
    filtered_scores = np.where(mask, scores, np.nan)
    return filtered_keypoints, filtered_scores

def sort_people_physiotrack(prev_keypoints: np.ndarray, 
                          current_keypoints: np.ndarray, 
                          scores: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Associate persons across frames using simple distance metric
    
    Args:
        prev_keypoints: Keypoints from previous frame [P, K, 2] (P persons, K keypoints)
        current_keypoints: Keypoints from current frame [P, K, 2]
        scores: Keypoint scores from current frame [P, K]
        
    Returns:
        Tuple: (sorted_prev_keypoints, sorted_current_keypoints, sorted_scores)
    """
    import itertools as it
    
    # Check if inputs are empty
    if len(prev_keypoints) == 0 or len(current_keypoints) == 0:
        return prev_keypoints, current_keypoints, scores
    
    # Generate all possible person correspondences
    max_len = max(len(prev_keypoints), len(current_keypoints))
    
    # Pad arrays to same length if necessary
    def pad_array(arr, target_len):
        if len(arr) < target_len:
            pad_shape = (target_len - len(arr),) + arr.shape[1:]
            padding = np.full(pad_shape, np.nan)
            return np.concatenate((arr, padding))
        return arr
    
    prev_keypoints_padded = pad_array(prev_keypoints, max_len)
    current_keypoints_padded = pad_array(current_keypoints, max_len)
    if scores is not None:
        scores_padded = pad_array(scores, max_len)
    
    # Compute distance between persons from one frame to another
    person_pairs = list(it.product(range(len(prev_keypoints)), range(len(current_keypoints))))
    
    # Calculate Euclidean distance between corresponding keypoints
    distances = []
    for prev_idx, curr_idx in person_pairs:
        # Get keypoints for this pair
        prev_kpts = prev_keypoints[prev_idx]
        curr_kpts = current_keypoints[curr_idx]
        
        # Calculate distance, ignoring NaNs
        valid_mask = ~np.isnan(prev_kpts).any(axis=1) & ~np.isnan(curr_kpts).any(axis=1)
        if not np.any(valid_mask):
            distances.append(float('inf'))
            continue
            
        valid_prev_kpts = prev_kpts[valid_mask]
        valid_curr_kpts = curr_kpts[valid_mask]
        
        dist = np.linalg.norm(valid_prev_kpts - valid_curr_kpts, axis=1).mean()
        distances.append(dist)
    
    # Create correspondence tuples (previous_idx, current_idx, distance)
    correspondences = []
    for i, (prev_idx, curr_idx) in enumerate(person_pairs):
        correspondences.append((prev_idx, curr_idx, distances[i]))
    
    # Sort by distance
    correspondences.sort(key=lambda x: x[2])
    
    # Assign correspondences (each person can only be matched once)
    matched_prev = set()
    matched_curr = set()
    final_matches = []
    
    for prev_idx, curr_idx, dist in correspondences:
        if prev_idx not in matched_prev and curr_idx not in matched_curr:
            final_matches.append((prev_idx, curr_idx))
            matched_prev.add(prev_idx)
            matched_curr.add(curr_idx)
    
    # Sort current keypoints according to matches
    sorted_current = np.full_like(current_keypoints_padded, np.nan)
    if scores is not None:
        sorted_scores = np.full_like(scores_padded, np.nan)
    
    for prev_idx, curr_idx in final_matches:
        # If index is within bounds
        if prev_idx < len(sorted_current):
            sorted_current[prev_idx] = current_keypoints[curr_idx]
            if scores is not None:
                sorted_scores[prev_idx] = scores[curr_idx]
    
    # Fill unmatched slots
    curr_used = {curr for _, curr in final_matches}
    curr_unused = set(range(len(current_keypoints))) - curr_used
    prev_used = {prev for prev, _ in final_matches}
    prev_unused = set(range(len(sorted_current))) - prev_used
    
    for prev_idx, curr_idx in zip(sorted(prev_unused), sorted(curr_unused)):
        if curr_idx < len(current_keypoints) and prev_idx < len(sorted_current):
            sorted_current[prev_idx] = current_keypoints[curr_idx]
            if scores is not None:
                sorted_scores[prev_idx] = scores[curr_idx]
    
    # Keep track of previous values when missing
    sorted_prev_keypoints = np.where(np.isnan(sorted_current) & ~np.isnan(prev_keypoints_padded), 
                                    prev_keypoints_padded, sorted_current)
    
    if scores is not None:
        return sorted_prev_keypoints, sorted_current, sorted_scores
    else:
        return sorted_prev_keypoints, sorted_current