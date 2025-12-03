"""
Detection modules for various computer vision tasks.
"""

from src.detectors.emotion import EmotionAnalyzer
from src.detectors.gesture import GestureDetector
from src.detectors.posture import PostureAnalyzer
from src.detectors.context import ContextAnalyzer

__all__ = [
    'EmotionAnalyzer',
    'GestureDetector',
    'PostureAnalyzer',
    'ContextAnalyzer',
]
