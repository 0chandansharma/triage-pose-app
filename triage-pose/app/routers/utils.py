# Utility endpoints
"""
Utility API endpoints
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, Response
from typing import List

from ..visualization.plot_utils import create_angle_plot, create_rom_comparison_chart
import physiotrack

router = APIRouter()

@router.get("/angle-definitions")
async def get_angle_definitions():
    """Get all available angle definitions"""
    return {
        "angle_definitions": physiotrack.ANGLE_DEFINITIONS
    }

@router.get("/joint-angles")
async def get_joint_angles():
    """Get available joint angles"""
    joint_angles = [name for name, params in physiotrack.ANGLE_DEFINITIONS.items() 
                   if len(params[0]) > 2]  # Joint angles use 3+ points
    return {
        "joint_angles": joint_angles
    }

@router.get("/segment-angles")
async def get_segment_angles():
    """Get available segment angles"""
    segment_angles = [name for name, params in physiotrack.ANGLE_DEFINITIONS.items() 
                     if len(params[0]) == 2]  # Segment angles use 2 points
    return {
        "segment_angles": segment_angles
    }

@router.get("/version")
async def get_version():
    """Get application and PhysioTrack versions"""
    return {
        "app_version": "0.1.0",
        "physiotrack_version": physiotrack.__version__
    }

@router.get("/plot/sample", response_class=Response)
async def get_sample_plot():
    """Generate a sample plot for testing"""
    import numpy as np
    
    # Create sample data
    times = np.linspace(0, 10, 100)
    angles = {
        "right knee": 60 + 30 * np.sin(times),
        "left knee": 60 + 30 * np.sin(times + np.pi),
        "trunk": 90 + 20 * np.sin(times / 2)
    }
    
    # Create plot
    plot_data = create_angle_plot(angles, times.tolist(), "Sample Angle Plot")
    
    return Response(content=plot_data, media_type="image/png")