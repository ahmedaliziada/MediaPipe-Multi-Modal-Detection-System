"""
MediaPipe Multi-Modal Detection System
Professional computer vision system for real-time analysis.
"""

__version__ = "2.0.0"
__author__ = "Ahmed Ziada"
__description__ = "Professional MediaPipe Detection System"

from src.system import MediaPipeDetectionSystem
from src.core.config import (
    camera_config,
    processing_config,
    model_paths,
    detection_thresholds,
    colors
)
from src.core.logger import logger

__all__ = [
    'MediaPipeDetectionSystem',
    'camera_config',
    'processing_config',
    'model_paths',
    'detection_thresholds',
    'colors',
    'logger',
]
