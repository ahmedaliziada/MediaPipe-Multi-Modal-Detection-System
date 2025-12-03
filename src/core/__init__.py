"""
Core modules for the detection system.
"""

from src.core.config import (
    camera_config,
    processing_config,
    model_paths,
    detection_thresholds,
    colors,
    CameraConfig,
    ProcessingConfig,
    ModelPaths,
    DetectionThresholds,
    DisplayColors,
)
from src.core.logger import logger, setup_logger
from src.core.performance import PerformanceMonitor

__all__ = [
    'camera_config',
    'processing_config',
    'model_paths',
    'detection_thresholds',
    'colors',
    'CameraConfig',
    'ProcessingConfig',
    'ModelPaths',
    'DetectionThresholds',
    'DisplayColors',
    'logger',
    'setup_logger',
    'PerformanceMonitor',
]
