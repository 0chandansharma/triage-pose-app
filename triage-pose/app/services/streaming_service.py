"""
Real-time streaming service
"""
from fastapi import WebSocket
import json
import asyncio
import base64
import cv2
import numpy as np
from typing import Dict, Any, Optional

from ..models.data import ProcessingOptions, FrameContext
from .video_service import VideoProcessor

class StreamingService:
    """Real-time streaming service for pose detection and analysis"""
    
    def __init__(self):
        """Initialize the streaming service"""
        self.active_sessions = {}
    
    async def process_stream(self, websocket: WebSocket, options: ProcessingOptions):
        """
        Process a real-time video stream
        
        Args:
            websocket: WebSocket connection
            options: Processing options
        """
        # Initialize video processor
        processor = VideoProcessor(options)
        context = FrameContext()
        
        try:
            while True:
                # Receive frame data
                data = await websocket.receive_text()
                
                # Check if it's a control message
                if data.startswith('{"command":'):
                    command = json.loads(data)
                    if command.get("command") == "stop":
                        break
                    continue
                
                # Decode base64 image
                try:
                    # Handle data URL format or pure base64
                    if data.startswith('data:image'):
                        base64_img = data.split(',')[1]
                    else:
                        base64_img = data
                        
                    img_bytes = base64.b64decode(base64_img)
                    img_np = np.frombuffer(img_bytes, np.uint8)
                    frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
                except Exception as e:
                    await websocket.send_json({"error": f"Failed to decode image: {str(e)}"})
                    continue
                
                # Process frame
                processed_frame, frame_data, context = processor.process_frame(frame, context)
                
                # Convert angles and keypoints for JSON
                angles_json = frame_data["angles"]
                
                keypoints_json = {}
                for i, name in enumerate(processor.keypoint_names):
                    if i < len(frame_data["keypoints"]) and not np.isnan(frame_data["keypoints"][i, 0]):
                        keypoints_json[name] = {
                            "x": float(frame_data["keypoints"][i, 0]),
                            "y": float(frame_data["keypoints"][i, 1]),
                            "score": float(frame_data["scores"][i])
                        }
                
                # Build ROM data
                rom_data = {}
                for angle_name in angles_json:
                    if angle_name in context.running_min and angle_name in context.running_max:
                        rom_data[angle_name] = {
                            "min": context.running_min[angle_name],
                            "max": context.running_max[angle_name],
                            "range": context.running_max[angle_name] - context.running_min[angle_name]
                        }
                
                # Encode processed frame
                _, buffer = cv2.imencode('.jpg', processed_frame)
                processed_b64 = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
                
                # Send response
                await websocket.send_json({
                    "frame_id": context.frame_count,
                    "processed_frame": processed_b64,
                    "keypoints": keypoints_json,
                    "angles": angles_json,
                    "rom_data": rom_data
                })
        
        except Exception as e:
            await websocket.send_json({"error": f"Error processing stream: {str(e)}"})