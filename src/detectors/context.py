"""
Contextual analysis combining multiple detection modalities.
"""

from typing import Optional, List
from src.core.logger import logger
from src.detectors.base import BaseDetector


class ContextAnalyzer(BaseDetector):
    """Analyzes context from multiple detection outputs."""
    
    CONTEXT_PATTERNS = {
        'POSITIVE_FEEDBACK': '[+] Positive Feedback',
        'TECH_FRUSTRATION': '[!] Possible Tech Frustration',
        'TIRED_STRESSED': '[~] Appears Tired/Stressed',
        'FOCUSED': '[*] Appears Focused',
        'RELAXED': '[=] Peaceful/Relaxed',
        'EXCITED': '[^] Excited/Enthusiastic',
        'PRESENTING': '[>] Presenting/Speaking',
        'WORKING': '[#] Working/Concentrating',
    }
    
    def __init__(self):
        super().__init__("ContextAnalyzer")
        self.context_history: List[str] = []
        self.max_history = 10
    
    def detect(
        self,
        emotion: Optional[str],
        gestures: List[str],
        posture: Optional[str],
        objects: List[str]
    ) -> Optional[str]:
        """
        Combine multiple signals to infer context.
        
        Args:
            emotion: Detected emotion
            gestures: List of detected gestures
            posture: Detected posture
            objects: List of detected objects
            
        Returns:
            Context interpretation or None
        """
        try:
            self.detection_count += 1
            contexts = []
            
            # Normalize inputs
            emotion = emotion.lower() if emotion else ""
            gestures_str = " ".join(gestures).lower()
            posture = posture.lower() if posture else ""
            objects_str = " ".join(objects).lower()
            
            # Positive feedback detection
            if "happy" in emotion and ("thumbs up" in gestures_str or "ok" in gestures_str):
                contexts.append(self.CONTEXT_PATTERNS['POSITIVE_FEEDBACK'])
            
            # Tech frustration detection
            tech_keywords = ['laptop', 'cell phone', 'keyboard', 'mouse', 'computer']
            has_tech = any(keyword in objects_str for keyword in tech_keywords)
            
            if has_tech and emotion in ["angry", "sad", "fear"]:
                contexts.append(self.CONTEXT_PATTERNS['TECH_FRUSTRATION'])
            
            # Tired/stressed detection
            if emotion in ["sad", "fear"] and "slouch" in posture:
                contexts.append(self.CONTEXT_PATTERNS['TIRED_STRESSED'])
            
            # Focused/working detection
            if emotion in ["neutral", "happy"] and "upright" in posture and has_tech:
                contexts.append(self.CONTEXT_PATTERNS['WORKING'])
            elif emotion in ["neutral", "happy"] and "upright" in posture:
                contexts.append(self.CONTEXT_PATTERNS['FOCUSED'])
            
            # Relaxed/peaceful detection
            if "happy" in emotion and ("peace" in gestures_str or "open hand" in gestures_str):
                contexts.append(self.CONTEXT_PATTERNS['RELAXED'])
            
            # Excited detection
            if "happy" in emotion or "surprise" in emotion:
                if "rock" in gestures_str or "upright" in posture:
                    contexts.append(self.CONTEXT_PATTERNS['EXCITED'])
            
            # Presenting detection
            if "pointing" in gestures_str or "open hand" in gestures_str:
                if "upright" in posture:
                    contexts.append(self.CONTEXT_PATTERNS['PRESENTING'])
            
            result = " | ".join(contexts) if contexts else None
            
            if result:
                self.context_history.append(result)
                if len(self.context_history) > self.max_history:
                    self.context_history.pop(0)
            
            return result
            
        except Exception as e:
            logger.error(f"Context analysis failed: {e}")
            return None
    
    def get_primary_context(self) -> Optional[str]:
        """Get the most common context from recent history."""
        if not self.context_history:
            return None
        
        return max(set(self.context_history), key=self.context_history.count)
