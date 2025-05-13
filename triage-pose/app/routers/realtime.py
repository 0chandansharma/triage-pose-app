"""
Real-time assessment API endpoints
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
import json
from typing import Dict, Any

from ..services.streaming_service import StreamingService
from ..models.data import ProcessingOptions
from ..models.request import RealtimeParams

router = APIRouter()

# Create a global streaming service
streaming_service = StreamingService()

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time processing"""
    await websocket.accept()
    
    try:
        # Receive configuration
        config_json = await websocket.receive_text()
        config = json.loads(config_json)
        
        # Parse configuration
        try:
            params = RealtimeParams(**config)
        except Exception as e:
            await websocket.send_json({"error": f"Invalid configuration: {str(e)}"})
            await websocket.close()
            return
        
        # Create processing options
        options = ProcessingOptions(
            model_type=params.model_type,
            detection_frequency=params.detection_frequency,
            tracking_mode=params.tracking_mode,
            device=params.device,
            backend=params.backend,
            joint_angles=params.joint_angles or [
                'right knee', 'left knee', 'right hip', 'left hip', 
                'right shoulder', 'left shoulder', 'right elbow', 'left elbow'
            ],
            segment_angles=params.segment_angles or [
                'right thigh', 'left thigh', 'trunk'
            ],
            height=params.height
        )
        
        # Process the stream
        await streaming_service.process_stream(websocket, options)
        
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        try:
            await websocket.send_json({"error": f"Error: {str(e)}"})
            await websocket.close()
        except:
            pass