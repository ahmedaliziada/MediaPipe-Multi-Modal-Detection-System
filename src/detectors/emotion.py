"""
Emotion detection using DeepFace.
"""

from typing import Optional, Dict, List
import numpy as np
from deepface import DeepFace

from src.core.logger import logger
from src.detectors.base import BaseDetector


class EmotionAnalyzer(BaseDetector):
    """Handles emotion detection using DeepFace."""
    
    EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    
    def __init__(self):
        super().__init__("EmotionAnalyzer")
        self.last_emotion: Optional[Dict[str, float]] = None
        self.emotion_history: List[str] = []
        self.max_history = 10
    
    def detect(
        self,
        face_roi: np.ndarray,
        face_landmarks: Optional[List] = None
    ) -> Optional[Dict[str, float]]:
        """
        Analyze emotion from face ROI.
        
        Args:
            face_roi: Face region of interest
            face_landmarks: Optional face landmarks for better cropping
            
        Returns:
            Dictionary of emotion scores or None if analysis fails
        """
        try:
            if face_roi.size == 0:
                logger.warning("Empty face ROI provided")
                return self.last_emotion
            
            # Ensure minimum size
            if face_roi.shape[0] < 48 or face_roi.shape[1] < 48:
                logger.warning(f"Face ROI too small: {face_roi.shape}")
                return self.last_emotion
            
            analysis = DeepFace.analyze(
                face_roi,
                actions=['emotion'],
                enforce_detection=False,
                silent=True
            )
            
            self.detection_count += 1
            emotions = analysis[0]['emotion']
            self.last_emotion = emotions
            
            # Update history
            dominant = max(emotions, key=emotions.get)
            self.emotion_history.append(dominant)
            if len(self.emotion_history) > self.max_history:
                self.emotion_history.pop(0)
            
            logger.debug(
                f"Emotion #{self.detection_count}: {dominant} "
                f"({emotions[dominant]:.1f}%)"
            )
            return emotions
            
        except Exception as e:
            logger.error(f"Emotion analysis failed: {e}")
            return self.last_emotion
    
    def get_dominant_emotion(self) -> Optional[str]:
        """Get the most common emotion from recent history."""
        if not self.emotion_history:
            return None
        
        # Return most common emotion
        return max(set(self.emotion_history), key=self.emotion_history.count)
    
    def get_emotion_trend(self) -> str:
        """Get emotion trend description."""
        if len(self.emotion_history) < 3:
            return "Insufficient data"
        
        recent = self.emotion_history[-3:]
        if len(set(recent)) == 1:
            return "Stable"
        else:
            return "Fluctuating"
