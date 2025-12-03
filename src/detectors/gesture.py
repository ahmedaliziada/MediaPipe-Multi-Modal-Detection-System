"""
Hand gesture recognition.
"""

from typing import List
from src.core.logger import logger
from src.detectors.base import BaseDetector


class GestureDetector(BaseDetector):
    """Detects hand gestures from landmarks."""
    
    GESTURES = {
        'THUMBS_UP': 'Thumbs Up',
        'THUMBS_DOWN': 'Thumbs Down',
        'PEACE': 'Peace Sign',
        'OK': 'OK Sign',
        'OPEN_HAND': 'Open Hand',
        'FIST': 'Fist',
        'POINTING': 'Pointing',
        'ROCK': 'Rock Sign',
    }
    
    def __init__(self):
        super().__init__("GestureDetector")
    
    def detect(self, hand_landmarks: List) -> str:
        """
        Detect gesture from hand landmarks.
        
        Args:
            hand_landmarks: List of hand landmark points (21 points)
            
        Returns:
            Gesture name
        """
        if not hand_landmarks or len(hand_landmarks) < 21:
            return "Unknown"
        
        try:
            self.detection_count += 1
            
            # Extract key landmarks
            thumb_tip = hand_landmarks[4]
            thumb_ip = hand_landmarks[3]
            thumb_base = hand_landmarks[2]
            
            index_tip = hand_landmarks[8]
            index_dip = hand_landmarks[7]
            index_pip = hand_landmarks[6]
            index_mcp = hand_landmarks[5]
            
            middle_tip = hand_landmarks[12]
            middle_pip = hand_landmarks[10]
            
            ring_tip = hand_landmarks[16]
            ring_pip = hand_landmarks[14]
            
            pinky_tip = hand_landmarks[20]
            pinky_pip = hand_landmarks[18]
            
            # Calculate finger states (extended or not)
            thumb_extended = thumb_tip.y < thumb_ip.y
            index_extended = index_tip.y < index_pip.y
            middle_extended = middle_tip.y < middle_pip.y
            ring_extended = ring_tip.y < ring_pip.y
            pinky_extended = pinky_tip.y < pinky_pip.y
            
            # Thumbs up detection
            if (thumb_tip.y < thumb_base.y - 0.1 and
                not index_extended and not middle_extended and
                not ring_extended and not pinky_extended):
                return self.GESTURES['THUMBS_UP']
            
            # Thumbs down detection
            if (thumb_tip.y > thumb_base.y + 0.1 and
                not index_extended and not middle_extended and
                not ring_extended and not pinky_extended):
                return self.GESTURES['THUMBS_DOWN']
            
            # Peace sign (V)
            if (index_extended and middle_extended and
                not ring_extended and not pinky_extended):
                return self.GESTURES['PEACE']
            
            # Rock sign (🤘)
            if (index_extended and pinky_extended and
                not middle_extended and not ring_extended):
                return self.GESTURES['ROCK']
            
            # OK sign
            thumb_index_distance = abs(thumb_tip.x - index_tip.x) + abs(thumb_tip.y - index_tip.y)
            if (thumb_index_distance < 0.05 and
                middle_extended and ring_extended and pinky_extended):
                return self.GESTURES['OK']
            
            # Pointing
            if (index_extended and not middle_extended and
                not ring_extended and not pinky_extended and
                not thumb_extended):
                return self.GESTURES['POINTING']
            
            # Fist
            if (not index_extended and not middle_extended and
                not ring_extended and not pinky_extended):
                return self.GESTURES['FIST']
            
            # Open hand (default)
            if (index_extended and middle_extended and
                ring_extended and pinky_extended):
                return self.GESTURES['OPEN_HAND']
            
            return "Unknown Gesture"
            
        except Exception as e:
            logger.error(f"Gesture detection failed: {e}")
            return "Error"
