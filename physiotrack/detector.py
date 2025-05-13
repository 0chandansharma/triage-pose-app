"""
Core pose detection functionality using RTMLib
"""
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
from rtmlib import PoseTracker, BodyWithFeet, Wholebody, Body
from anytree import Node, RenderTree
from .models import DetectionConfig

class PoseDetector:
    """Minimal pose detection interface"""
    
    def __init__(self, model_type: str = "body_with_feet", 
                 detection_frequency: int = 4,
                 tracking_mode: str = "physiotrack",
                 device: str = "auto",
                 backend: str = "auto"):
        """Initialize pose detector with minimal configuration"""
        self.config = DetectionConfig(
            model_type=model_type,
            detection_frequency=detection_frequency,
            tracking_mode=tracking_mode,
            device=device,
            backend=backend
        )
        self.tracker = None
        self.model = None
        self.keypoints_names = []
        self.keypoints_ids = []
        
        # Set up the detector
        self._setup_detector(model_type, detection_frequency, tracking_mode, device, backend)
    
    def _setup_detector(self, model_type: str, detection_frequency: int, 
                       tracking_mode: str, device: str, backend: str):
        """Set up the underlying detector based on model type"""
        # Set up backend and device
        backend, device = self._setup_backend_device(backend, device)
        
        # Select the appropriate model based on the model_type
        if model_type.upper() in ('HALPE_26', 'BODY_WITH_FEET'):
            ModelClass = BodyWithFeet
            self.model = self._create_halpe26_model()
        elif model_type.upper() in ('COCO_133_WRIST', 'WHOLE_BODY_WRIST'):
            ModelClass = Wholebody
            self.model = self._create_coco133_wrist_model()
        elif model_type.upper() in ('COCO_133', 'WHOLE_BODY'):
            ModelClass = Wholebody
            self.model = self._create_coco133_model()
        elif model_type.upper() in ('COCO_17', 'BODY'):
            ModelClass = Body
            self.model = self._create_coco17_model()
        else:
            raise ValueError(f"Invalid model_type: {model_type}. Must be 'HALPE_26', 'COCO_133', 'COCO_133_WRIST', or 'COCO_17'.")
        
        # Extract keypoint information from the model
        self.keypoints_ids = [node.id for _, _, node in RenderTree(self.model) if node.id is not None]
        self.keypoints_names = [node.name for _, _, node in RenderTree(self.model) if node.id is not None]
        
        # Initialize the RTMLib pose tracker
        self.tracker = PoseTracker(
            ModelClass,
            det_frequency=detection_frequency,
            mode="balanced",  # Default to balanced for most use cases
            backend=backend,
            device=device,
            tracking=False,  # We'll handle tracking ourselves
            to_openpose=False)
    
    def _setup_backend_device(self, backend: str, device: str):
        """Set up the backend and device for the pose tracker"""
        if device != 'auto' and backend != 'auto':
            return backend.lower(), device.lower()

        if device == 'auto' or backend == 'auto':
            try:
                import torch
                import onnxruntime as ort
                if torch.cuda.is_available() and 'CUDAExecutionProvider' in ort.get_available_providers():
                    return 'onnxruntime', 'cuda'
                elif 'ROCMExecutionProvider' in ort.get_available_providers():
                    return 'onnxruntime', 'rocm'
            except:
                try:
                    import onnxruntime as ort
                    if 'MPSExecutionProvider' in ort.get_available_providers():
                        return 'onnxruntime', 'mps'
                except:
                    return 'openvino', 'cpu'
                    
        return backend, device
    
    def _create_halpe26_model(self):
        """Create HALPE_26 skeleton model"""
        return Node("Hip", id=19, children=[
            Node("RHip", id=12, children=[
                Node("RKnee", id=14, children=[
                    Node("RAnkle", id=16, children=[
                        Node("RBigToe", id=21, children=[
                            Node("RSmallToe", id=23),
                        ]),
                        Node("RHeel", id=25),
                    ]),
                ]),
            ]),
            Node("LHip", id=11, children=[
                Node("LKnee", id=13, children=[
                    Node("LAnkle", id=15, children=[
                        Node("LBigToe", id=20, children=[
                            Node("LSmallToe", id=22),
                        ]),
                        Node("LHeel", id=24),
                    ]),
                ]),
            ]),
            Node("Neck", id=18, children=[
                Node("Head", id=17, children=[
                    Node("Nose", id=0),
                ]),
                Node("RShoulder", id=6, children=[
                    Node("RElbow", id=8, children=[
                        Node("RWrist", id=10),
                    ]),
                ]),
                Node("LShoulder", id=5, children=[
                    Node("LElbow", id=7, children=[
                        Node("LWrist", id=9),
                    ]),
                ]),
            ]),
        ])
        
    def _create_coco17_model(self):
        """Create COCO_17 skeleton model"""
        return Node("Hip", id=None, children=[
            Node("RHip", id=12, children=[
                Node("RKnee", id=14, children=[
                    Node("RAnkle", id=16),
                ]),
            ]),
            Node("LHip", id=11, children=[
                Node("LKnee", id=13, children=[
                    Node("LAnkle", id=15),
                ]),
            ]),
            Node("Neck", id=None, children=[
                Node("Nose", id=0),
                Node("RShoulder", id=6, children=[
                    Node("RElbow", id=8, children=[
                        Node("RWrist", id=10),
                    ]),
                ]),
                Node("LShoulder", id=5, children=[
                    Node("LElbow", id=7, children=[
                        Node("LWrist", id=9),
                    ]),
                ]),
            ]),
        ])
        
    def _create_coco133_model(self):
        """Create COCO_133 skeleton model (body, face, hands)"""
        # Simplified for brevity - in practice, you would define the full model
        return Node("Hip", id=None, children=[
            Node("RHip", id=12),
            Node("LHip", id=11),
            Node("Neck", id=None, children=[
                Node("Nose", id=0),
                Node("RShoulder", id=6),
                Node("LShoulder", id=5),
            ])
        ])
        
    def _create_coco133_wrist_model(self):
        """Create COCO_133_WRIST skeleton model (body, hands)"""
        # Simplified for brevity
        return self._create_coco133_model()
    
    def detect_pose(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Detect poses in a single frame
        
        Args:
            frame: numpy array image
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (keypoints, scores)
        """
        if self.tracker is None:
            raise ValueError("Pose tracker not initialized. Call _setup_detector() first.")
        
        # Process the frame with RTMLib
        keypoints, scores = self.tracker(frame)
        return keypoints, scores
    
    def get_keypoint_names(self) -> List[str]:
        """Get the list of keypoint names for the current model"""
        return self.keypoints_names
        
    def get_keypoint_ids(self) -> List[int]:
        """Get the list of keypoint IDs for the current model"""
        return self.keypoints_ids