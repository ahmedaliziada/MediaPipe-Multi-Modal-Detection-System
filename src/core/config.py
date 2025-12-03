"""
Configuration module for MediaPipe Detection System.
Contains all configurable parameters and paths.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import os

# Base paths
BASE_DIR = Path(__file__).parent.parent.parent
MODEL_DIR = BASE_DIR / "models"  # Changed to use local models folder
OUTPUT_DIR = BASE_DIR / "output"

# Ensure directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "logs").mkdir(exist_ok=True)
(OUTPUT_DIR / "frames").mkdir(exist_ok=True)
(OUTPUT_DIR / "videos").mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)  # Ensure models directory exists


@dataclass
class CameraConfig:
    """Camera configuration parameters."""
    index: int = int(os.getenv('CAMERA_INDEX', 1))
    width: int = int(os.getenv('CAMERA_WIDTH', 1280))  # Increased from 640
    height: int = int(os.getenv('CAMERA_HEIGHT', 720))  # Increased from 480
    warmup_frames: int = 10
    backend: int = None  # cv2.CAP_AVFOUNDATION
    fps: int = 30


@dataclass
class ProcessingConfig:
    """Processing configuration parameters."""
    emotion_frame_skip: int = 10
    fps_update_interval: float = 1.0
    enable_performance_metrics: bool = True
    max_detection_faces: int = 1
    max_detection_hands: int = 2


@dataclass
class ModelPaths:
    """Paths to MediaPipe model files."""
    face_landmarker: Path = MODEL_DIR / "face_landmarker.task"
    hand_landmarker: Path = MODEL_DIR / "hand_landmarker.task"
    pose_landmarker: Path = MODEL_DIR / "pose_landmarker_lite.task"  # Using lite version
    object_detector: Path = MODEL_DIR / "efficientdet_lite0.tflite"
    
    def validate(self) -> bool:
        """Validate that all model files exist."""
        missing = []
        for name, path in self.__dict__.items():
            if isinstance(path, Path) and not path.exists():
                missing.append(f"  - {name}: {path.name}")
        
        if missing:
            error_msg = (
                f"\n❌ Missing model files in {MODEL_DIR}:\n" + 
                "\n".join(missing) +
                f"\n\n💡 Download models using:\n"
                f"   python scripts/download_models.py\n"
            )
            raise FileNotFoundError(error_msg)
        return True


@dataclass
class DetectionThresholds:
    """Threshold values for various detections."""
    object_confidence: float = 0.5
    face_confidence: float = 0.5
    hand_confidence: float = 0.5
    pose_confidence: float = 0.5


@dataclass
class DisplayColors:
    """BGR color values for visualization."""
    FACE: Tuple[int, int, int] = (255, 200, 0)
    HAND: Tuple[int, int, int] = (0, 255, 0)
    POSE: Tuple[int, int, int] = (0, 100, 255)
    EMOTION: Tuple[int, int, int] = (255, 255, 255)
    OBJECT: Tuple[int, int, int] = (0, 255, 255)
    PERSON: Tuple[int, int, int] = (255, 0, 255)
    TEXT: Tuple[int, int, int] = (200, 200, 200)
    SUCCESS: Tuple[int, int, int] = (0, 255, 0)
    WARNING: Tuple[int, int, int] = (0, 165, 255)
    ERROR: Tuple[int, int, int] = (0, 0, 255)


# Instantiate configurations
camera_config = CameraConfig()
processing_config = ProcessingConfig()
model_paths = ModelPaths()
detection_thresholds = DetectionThresholds()
colors = DisplayColors()
