from dataclasses import dataclass, field
from typing import Tuple, List
import os

@dataclass
class CameraConfig:
    width: int = 800
    height: int = 600
    device_id: int = 0
    fps_limit: int = 30

@dataclass
class ModelPaths:
    age_proto: str = "models/age_deploy.prototxt"
    age_model: str = "models/age_net.caffemodel"
    
    @property
    def files_exist(self) -> bool:
        return os.path.exists(self.age_proto) and os.path.exists(self.age_model)

@dataclass
class AgeConfig:
    model_paths: ModelPaths = field(default_factory=ModelPaths)
    mean_values: Tuple[float, float, float] = (78.4263377603, 87.7689143744, 114.895847746)
    age_buckets: List[str] = field(default_factory=lambda: [
        '(0-2)', '(4-6)', '(8-12)', '(15-20)', 
        '(25-32)', '(38-43)', '(48-53)', '(60-100)'
    ])
    confidence_threshold: float = 0.5

@dataclass
class MediaPipeConfig:
    static_image_mode: bool = False
    model_complexity: int = 0
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    refine_face_landmarks: bool = False
    enable_face: bool = True
    enable_pose: bool = True
    enable_hands: bool = True

@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    age: AgeConfig = field(default_factory=AgeConfig)
    mediapipe: MediaPipeConfig = field(default_factory=MediaPipeConfig)
    processing_width: int = 320 # Downscale width for inference
    window_name: str = "Face & Hand Landmarks + Age Estimation"
