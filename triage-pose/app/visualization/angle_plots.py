# Angle visualization
# app/visualization/angle_plots.py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import io
from typing import Dict, List, Any, Tuple

def draw_angles_on_frame(frame: np.ndarray, 
                         keypoints: np.ndarray,
                         angles: Dict[str, float],
                         display_options: Dict[str, Any] = None) -> np.ndarray:
    """Draw angles on a frame
    
    Moved from physiotrack/process.py draw_angles()
    """
    # Angle drawing logic
    
def create_angle_time_series_plot(angle_data: pd.DataFrame, 
                                 angle_names: List[str] = None) -> bytes:
    """Create a time series plot of angles
    
    Returns:
        bytes: PNG image data
    """
    # Plot creation logic from physiotrack/process.py