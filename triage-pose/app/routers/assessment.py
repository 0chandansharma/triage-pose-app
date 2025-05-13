"""
Assessment API endpoints
"""
from fastapi import APIRouter, File, UploadFile, BackgroundTasks, Form, Depends, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Optional
import uuid
import os
import json
import pandas as pd
import tempfile
from pathlib import Path

from ..services.video_service import VideoProcessor
from ..services.analysis_service import ROMAnalyzer
from ..models.request import ROMAssessmentParams
from ..models.response import AssessmentResponse
from ..models.data import ProcessingOptions, ROMAnalysisOptions
from ..config import settings

router = APIRouter()

@router.post("/rom", response_model=AssessmentResponse)
async def assess_rom(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    params: Optional[str] = Form("{}")
):
    """
    Analyze range of motion from a video
    
    - **video**: Video file to analyze
    - **params**: JSON string of assessment parameters
    """
    # Create a temporary file to store the uploaded video
    temp_dir = Path(settings.get_temp_path())
    
    # Generate a unique ID for this assessment
    assessment_id = str(uuid.uuid4())
    
    # Create a directory for this assessment
    assessment_dir = temp_dir / assessment_id
    assessment_dir.mkdir(exist_ok=True)
    
    # Save the uploaded video
    video_path = assessment_dir / f"input{Path(video.filename).suffix}"
    with open(video_path, "wb") as f:
        f.write(await video.read())
    
    # Parse parameters
    try:
        params_dict = json.loads(params)
        assessment_params = ROMAssessmentParams(**params_dict)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in params")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid parameters: {str(e)}")
    
    # Set up processing options
    options = ProcessingOptions(
        model_type=assessment_params.model_type,
        detection_frequency=assessment_params.detection_frequency,
        tracking_mode=assessment_params.tracking_mode,
        device=assessment_params.device,
        backend=assessment_params.backend,
        joint_angles=assessment_params.joint_angles or [
            'right knee', 'left knee', 'right hip', 'left hip', 
            'right shoulder', 'left shoulder', 'right elbow', 'left elbow'
        ],
        segment_angles=assessment_params.segment_angles or [
            'right thigh', 'left thigh', 'trunk'
        ],
        height=assessment_params.height,
        visible_side=assessment_params.visible_side
    )
    
    # Set up analysis options
    analysis_options = ROMAnalysisOptions(
        test_name="lb-flexion",
        height=assessment_params.height
    )
    
    # Initialize processors
    video_processor = VideoProcessor(options)
    rom_analyzer = ROMAnalyzer(analysis_options)
    
    # Process the video using PhysioTrack (in background)
    background_tasks.add_task(
        process_assessment,
        video_processor,
        rom_analyzer,
        str(video_path),
        str(assessment_dir),
        assessment_id
    )
    
    # Return response with assessment ID
    return {
        "assessment_id": assessment_id,
        "status": "processing",
        "message": "Video uploaded and being processed. Check status endpoint for results."
    }

@router.get("/rom/{assessment_id}", response_model=AssessmentResponse)
async def get_rom_assessment(assessment_id: str):
    """
    Get the results of a range of motion assessment
    
    - **assessment_id**: ID of the assessment to retrieve
    """
    assessment_dir = Path(settings.get_temp_path()) / assessment_id
    if not assessment_dir.exists():
        raise HTTPException(status_code=404, detail="Assessment not found")
    
    # Check if processing is complete
    status_file = assessment_dir / "status.json"
    if not status_file.exists():
        return {
            "assessment_id": assessment_id,
            "status": "processing",
            "message": "Assessment is still being processed"
        }
    
    # Read results
    with open(status_file, "r") as f:
        status = json.load(f)
    
    # If processing complete, analyze ROM from angles file
    if status["status"] == "complete":
        angles_file = assessment_dir / f"{assessment_id}_angles_person00.mot"
        if angles_file.exists():
            # Initialize ROM analyzer
            rom_analyzer = ROMAnalyzer()
            rom_analysis = rom_analyzer.analyze_rom(str(angles_file))
            
            # Get the video URL
            video_url = f"/static/temp/{assessment_id}/{assessment_id}.mp4"
            
            return {
                "assessment_id": assessment_id,
                "status": "complete",
                "results": rom_analysis,
                "video_url": video_url
            }
    
    # Return current status
    return {
        "assessment_id": assessment_id,
        "status": status["status"],
        "message": status.get("message", "")
    }

async def process_assessment(
    video_processor: VideoProcessor,
    rom_analyzer: ROMAnalyzer,
    video_path: str,
    output_dir: str,
    assessment_id: str
):
    """
    Process a video for ROM assessment
    
    Args:
        video_processor: VideoProcessor instance
        rom_analyzer: ROMAnalyzer instance
        video_path: Path to video file
        output_dir: Directory to save results
        assessment_id: Unique assessment ID
    """
    # Process the video
    result = await video_processor.process_video(video_path, output_dir)
    
    # If successful, analyze ROM and generate ROM data
    if result["status"] == "complete" and "angles_file" in result:
        # Load angle data
        angles_file = result["angles_file"]
        
        # Analyze ROM
        rom_analysis = rom_analyzer.analyze_rom(angles_file)
        
        # Read the angle data for ROM data generation
        with open(angles_file, 'r') as f:
            for i, line in enumerate(f):
                if line.startswith('time'):
                    header_rows = i
                    break
        
        angle_data = pd.read_csv(angles_file, sep='\t', skiprows=header_rows)
        
        # Generate ROM data
        rom_data = rom_analyzer.generate_rom_data(angle_data, "lb-flexion")
        
        # Save ROM data
        rom_data_file = Path(output_dir) / f"{assessment_id}_rom_data.json"
        with open(rom_data_file, "w") as f:
            json.dump(rom_data, f, indent=4)