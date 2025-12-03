"""
Base classes for detectors.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional


class BaseDetector(ABC):
    """Abstract base class for all detectors."""
    
    def __init__(self, name: str):
        """
        Initialize detector.
        
        Args:
            name: Detector name for logging
        """
        self.name = name
        self.detection_count = 0
    
    @abstractmethod
    def detect(self, *args, **kwargs) -> Any:
        """
        Perform detection.
        
        Returns:
            Detection results
        """
        pass
    
    def reset(self):
        """Reset detector state."""
        self.detection_count = 0
