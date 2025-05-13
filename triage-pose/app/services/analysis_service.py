"""
ROM analysis service
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Any

from ..models.data import ROMAnalysisOptions

class ROMAnalyzer:
    """Analyzes Range of Motion from angle data"""
    
    def __init__(self, options: ROMAnalysisOptions = None):
        """Initialize with analysis options"""
        self.options = options or ROMAnalysisOptions()
    
    def analyze_rom(self, angles_file: str) -> Dict[str, Any]:
        """
        Analyze ROM from an angles file
        
        Args:
            angles_file: Path to the angles MOT file
            
        Returns:
            Dict: ROM analysis results
        """
        # Skip the header lines
        with open(angles_file, 'r') as f:
            for i, line in enumerate(f):
                if line.startswith('time'):
                    header_rows = i
                    break
        
        # Read the data
        angles_data = pd.read_csv(angles_file, sep='\t', skiprows=header_rows)
        
        # Calculate ROM for each joint
        rom_results = {}
        for col in angles_data.columns:
            if col == 'time':
                continue
            
            # Calculate min, max, and ROM
            min_val = float(angles_data[col].min())
            max_val = float(angles_data[col].max())
            rom = float(max_val - min_val)
            mean = float(angles_data[col].mean())
            std = float(angles_data[col].std())
            
            # Extract time series data for this angle
            time_series = [{
                "time": float(row["time"]),
                "value": float(row[col])
            } for _, row in angles_data.iterrows()]
            
            rom_results[col] = {
                "min": min_val,
                "max": max_val,
                "rom": rom,
                "mean": mean,
                "std": std,
                "time_series": time_series
            }
        
        return {
            "rom_analysis": rom_results,
            "summary": {
                "total_frames": len(angles_data),
                "duration": float(angles_data["time"].max()),
                "angles_measured": list(rom_results.keys())
            }
        }
    
    def generate_rom_data(self, angle_data: pd.DataFrame, test_name: str) -> Dict[str, Any]:
        """
        Generate ROM data in the standardized format
        
        Args:
            angle_data: DataFrame with angle data
            test_name: Name of the test
            
        Returns:
            Dict: ROM data in standard format
        """
        # Initialize the ROM data structure
        rom_data = {}
        
        # Calculate window size in seconds and convert to indices
        window_size = self.options.time_window
        if 'trunk' in angle_data.columns:
            time_step = angle_data['time'].iloc[1] - angle_data['time'].iloc[0]
            window_size_idx = int(window_size / time_step)
            
            # Apply transformation (180 - angle) as in test_rom_analysis.py
            transformed_angles = 180 - angle_data['trunk']
            
            # Compute rolling mean over the window
            smoothed_angles = transformed_angles.rolling(window=window_size_idx, min_periods=1).mean()
            
            # Calculate the running min/max values and ROM at each time point
            running_min = np.zeros(len(smoothed_angles))
            running_max = np.zeros(len(smoothed_angles))
            running_rom = np.zeros(len(smoothed_angles))
            
            for i in range(len(smoothed_angles)):
                # Consider all angles from start up to current point
                current_segment = smoothed_angles.iloc[0:i+1]
                
                if not current_segment.empty:
                    # Calculate min, max and rom from start to current point
                    current_min = current_segment.min()
                    current_max = current_segment.max()
                    current_rom = current_max - current_min
                    
                    running_min[i] = current_min
                    running_max[i] = current_max
                    running_rom[i] = current_rom
        
        # Process each time point/frame
        for idx, row in angle_data.iterrows():
            time_val = str(round(row['time'], 3))  # Use time as key, rounded to 3 decimal places
            
            # Initialize angles dict for this time point
            angles_dict = {}
            
            # Collect all angles for this time point
            for col in angle_data.columns:
                if col == 'time':
                    continue
                
                # Add angle to the angles dictionary
                angles_dict[col] = float(round(row[col], 1))
            
            # Default ROM values
            rom_min = 0.0
            rom_max = 0.0
            rom_range = 0.0
            
            # Set ROM values if trunk angle is available and we've calculated them
            if 'trunk' in angle_data.columns and idx < len(running_rom):
                rom_min = float(running_min[idx])
                rom_max = float(running_max[idx])
                rom_range = float(running_rom[idx])
                
            # Create entry for this time point with all angles together
            rom_data[time_val] = {
                "test": test_name,
                "is_ready": True,
                "angles": angles_dict,
                "ROM": [rom_min, rom_max],  # Update with calculated values for trunk
                "rom_range": rom_range,  # Update with calculated ROM for trunk
                "position_valid": True,
                "guidance": "Good posture",
                "posture_message": "Good posture",
                "ready_progress": 100,
                "status": "success"
            }
        
        return rom_data