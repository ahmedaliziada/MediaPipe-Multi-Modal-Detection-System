"""
Body posture analysis.
"""

from typing import Optional, List
from src.core.logger import logger
from src.detectors.base import BaseDetector


class PostureAnalyzer(BaseDetector):
    """Analyzes body posture from pose landmarks."""
    
    POSTURE_TYPES = {
        'UPRIGHT': 'Upright & Confident',
        'SLOUCHING': 'Slouching',
        'TILTED': 'Tilted/Asymmetric',
        'FORWARD_HEAD': 'Forward Head Posture',
        'NEUTRAL': 'Neutral Posture',
    }
    
    def __init__(self):
        super().__init__("PostureAnalyzer")
        self.posture_history: List[str] = []
        self.max_history = 10
    
    def detect(self, pose_landmarks: List) -> Optional[str]:
        """
        Analyze body posture.
        
        Args:
            pose_landmarks: List of pose landmark points (33 points)
            
        Returns:
            Posture description or None
        """
        if not pose_landmarks or len(pose_landmarks) < 33:
            return None
        
        try:
            self.detection_count += 1
            
            # Key landmarks (MediaPipe Pose indices)
            nose = pose_landmarks[0]
            left_shoulder = pose_landmarks[11]
            right_shoulder = pose_landmarks[12]
            left_hip = pose_landmarks[23]
            right_hip = pose_landmarks[24]
            left_ear = pose_landmarks[7]
            right_ear = pose_landmarks[8]
            
            # Calculate metrics
            shoulder_diff = abs(left_shoulder.y - right_shoulder.y)
            hip_diff = abs(left_hip.y - right_hip.y)
            
            # Spine alignment (vertical distance between shoulder center and hip center)
            shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
            hip_center_y = (left_hip.y + right_hip.y) / 2
            torso_vertical = abs(shoulder_center_y - hip_center_y)
            
            # Head forward position (using z-coordinate if available)
            head_forward = False
            if hasattr(nose, 'z') and hasattr(left_shoulder, 'z'):
                head_forward = nose.z > left_shoulder.z + 0.05
            
            # Ear-shoulder alignment (forward head posture)
            ear_center_y = (left_ear.y + right_ear.y) / 2
            forward_head_y = ear_center_y > shoulder_center_y + 0.05
            
            # Determine posture
            posture = None
            
            if shoulder_diff < 0.03 and torso_vertical > 0.2 and not head_forward:
                posture = self.POSTURE_TYPES['UPRIGHT']
            elif shoulder_diff > 0.08 or hip_diff > 0.08:
                posture = self.POSTURE_TYPES['TILTED']
            elif head_forward or forward_head_y:
                posture = self.POSTURE_TYPES['FORWARD_HEAD']
            elif torso_vertical < 0.15:
                posture = self.POSTURE_TYPES['SLOUCHING']
            else:
                posture = self.POSTURE_TYPES['NEUTRAL']
            
            # Update history
            self.posture_history.append(posture)
            if len(self.posture_history) > self.max_history:
                self.posture_history.pop(0)
            
            logger.debug(f"Posture #{self.detection_count}: {posture}")
            return posture
                
        except Exception as e:
            logger.error(f"Posture analysis failed: {e}")
            return None
    
    def get_posture_stability(self) -> str:
        """Check if posture is stable or changing."""
        if len(self.posture_history) < 3:
            return "Insufficient data"
        
        recent = self.posture_history[-5:]
        if len(set(recent)) == 1:
            return "Stable posture"
        else:
            return "Changing posture"
