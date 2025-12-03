"""
Visualization and rendering utilities for detection results.
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from typing import List, Optional, Tuple, Dict

from src.core.config import colors
from src.core.logger import logger


class VisualizationRenderer:
    """Handles all visualization and rendering tasks."""
    
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Emotion icons (text-based, no emoji)
        self.emotion_display = {
            'happy': '[HAPPY]',
            'sad': '[SAD]',
            'angry': '[ANGRY]',
            'fear': '[FEAR]',
            'surprise': '[SURPRISE]',
            'disgust': '[DISGUST]',
            'neutral': '[NEUTRAL]'
        }
    
    def draw_face_landmarks(
        self,
        image: np.ndarray,
        detection_result,
        show_tesselation: bool = True
    ) -> np.ndarray:
        """
        Draw face mesh on image.
        
        Args:
            image: Input image
            detection_result: MediaPipe face detection result
            show_tesselation: Whether to show face mesh tesselation
            
        Returns:
            Annotated image
        """
        if not detection_result.face_landmarks:
            return image
        
        annotated_image = np.copy(image)
        
        for face_landmarks in detection_result.face_landmarks:
            face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            face_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) 
                for lm in face_landmarks
            ])
            
            if show_tesselation:
                self.mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks_proto,
                    connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles
                    .get_default_face_mesh_tesselation_style())
            
            # Draw face contours
            self.mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles
                .get_default_face_mesh_contours_style())
        
        return annotated_image
    
    def draw_hand_landmarks(
        self,
        image: np.ndarray,
        detection_result,
        gestures: List[str]
    ) -> np.ndarray:
        """
        Draw hand landmarks and gesture labels.
        
        Args:
            image: Input image
            detection_result: MediaPipe hand detection result
            gestures: List of detected gestures
            
        Returns:
            Annotated image
        """
        if not detection_result.hand_landmarks:
            return image
        
        annotated_image = np.copy(image)
        height, width = image.shape[:2]
        
        for idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
            # Draw landmarks
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) 
                for lm in hand_landmarks
            ])
            
            self.mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks_proto,
                mp.solutions.hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style())
            
            # Draw gesture label
            if idx < len(gestures) and idx < len(detection_result.handedness):
                handedness = detection_result.handedness[idx][0].category_name
                gesture = gestures[idx]
                
                # Position label near wrist
                wrist = hand_landmarks[0]
                x = int(wrist.x * width)
                y = int(wrist.y * height) - 30
                
                label = f"{handedness}: {gesture}"
                
                # Draw background rectangle
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(
                    annotated_image,
                    (x - 5, y - text_size[1] - 5),
                    (x + text_size[0] + 5, y + 5),
                    (0, 0, 0),
                    -1
                )
                
                # Draw text
                cv2.putText(
                    annotated_image,
                    label,
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    colors.HAND,
                    2
                )
        
        return annotated_image
    
    def draw_pose_landmarks(
        self,
        image: np.ndarray,
        detection_result,
        posture: Optional[str] = None
    ) -> np.ndarray:
        """
        Draw pose landmarks and posture label.
        
        Args:
            image: Input image
            detection_result: MediaPipe pose detection result
            posture: Detected posture description
            
        Returns:
            Annotated image
        """
        if not detection_result.pose_landmarks:
            return image
        
        annotated_image = np.copy(image)
        height, width = image.shape[:2]
        
        for pose_landmarks in detection_result.pose_landmarks:
            # Draw landmarks
            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            pose_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) 
                for lm in pose_landmarks
            ])
            
            self.mp_drawing.draw_landmarks(
                annotated_image,
                pose_landmarks_proto,
                mp.solutions.pose.POSE_CONNECTIONS,
                self.mp_drawing_styles.get_default_pose_landmarks_style())
            
            # Draw posture label
            if posture and len(pose_landmarks) > 0:
                # Position near nose
                nose = pose_landmarks[0]
                x = int(nose.x * width)
                y = int(nose.y * height) - 40
                
                # Draw background
                text_size = cv2.getTextSize(posture, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(
                    annotated_image,
                    (x - 5, y - text_size[1] - 5),
                    (x + text_size[0] + 5, y + 5),
                    (0, 0, 0),
                    -1
                )
                
                cv2.putText(
                    annotated_image,
                    posture,
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    colors.POSE,
                    2
                )
        
        return annotated_image
    
    def draw_object_detections(
        self,
        image: np.ndarray,
        detection_result
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Draw bounding boxes for detected objects.
        
        Args:
            image: Input image
            detection_result: MediaPipe object detection result
            
        Returns:
            Tuple of (annotated image, list of detected object names)
        """
        if not detection_result.detections:
            return image, []
        
        annotated_image = np.copy(image)
        detected_objects = []
        
        for detection in detection_result.detections:
            bbox = detection.bounding_box
            category = detection.categories[0]
            object_name = category.category_name
            confidence = round(category.score, 2)
            
            # Different color for persons vs objects
            is_person = object_name.lower() == 'person'
            box_color = colors.PERSON if is_person else colors.OBJECT
            
            # Draw bounding box with rounded corners effect
            start_point = (bbox.origin_x, bbox.origin_y)
            end_point = (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height)
            cv2.rectangle(annotated_image, start_point, end_point, box_color, 3)
            
            # Draw semi-transparent background for label
            label = f'{object_name} {confidence}'
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            label_bg_start = (bbox.origin_x, bbox.origin_y - text_size[1] - 10)
            label_bg_end = (bbox.origin_x + text_size[0] + 10, bbox.origin_y)
            
            # Create semi-transparent overlay
            overlay = annotated_image.copy()
            cv2.rectangle(overlay, label_bg_start, label_bg_end, box_color, -1)
            cv2.addWeighted(overlay, 0.6, annotated_image, 0.4, 0, annotated_image)
            
            # Draw label text
            cv2.putText(
                annotated_image,
                label,
                (bbox.origin_x + 5, bbox.origin_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
            
            detected_objects.append(object_name)
        
        return annotated_image, detected_objects
    
    def create_info_panel(
        self,
        image: np.ndarray,
        emotion_scores: Optional[Dict[str, float]],
        gestures: List[str],
        posture: Optional[str],
        objects: List[str],
        context: Optional[str],
        stats: Dict[str, float]
    ) -> np.ndarray:
        """
        Create beautiful comprehensive information panel overlay WITHOUT covering face.
        
        Args:
            image: Input image
            emotion_scores: Emotion analysis results
            gestures: Detected gestures
            posture: Detected posture
            objects: Detected objects
            context: Contextual interpretation
            stats: Performance statistics
            
        Returns:
            Image with beautiful information overlay
        """
        display = image.copy()
        height, width = display.shape[:2]
        
        # Create compact side panels instead of top overlay
        # Left side panel for emotion and context
        panel_width = 350
        panel_x = 10
        panel_y = 10
        
        # Draw semi-transparent panels on the LEFT side
        if emotion_scores or gestures or posture or objects or context:
            y_current = panel_y
            
            # Emotion panel
            if emotion_scores:
                dominant_emotion = max(emotion_scores, key=emotion_scores.get)
                confidence = emotion_scores[dominant_emotion]
                emotion_label = self.emotion_display.get(dominant_emotion, '[UNKNOWN]')
                
                # Small compact box
                box_height = 90
                overlay = display.copy()
                cv2.rectangle(overlay, (panel_x, y_current), (panel_x + panel_width, y_current + box_height), (20, 20, 30), -1)
                cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)
                cv2.rectangle(display, (panel_x, y_current), (panel_x + panel_width, y_current + box_height), (100, 150, 255), 2)
                
                # Emotion text
                self._draw_text_with_shadow(
                    display,
                    f'Emotion: {emotion_label}',
                    (panel_x + 15, y_current + 30),
                    0.7,
                    colors.EMOTION,
                    2
                )
                
                # Confidence bar
                bar_width = 200
                bar_x = panel_x + 15
                bar_y = y_current + 45
                confidence_width = int((confidence / 100) * bar_width)
                
                cv2.rectangle(display, (bar_x, bar_y), (bar_x + bar_width, bar_y + 8), (50, 50, 50), -1)
                cv2.rectangle(display, (bar_x, bar_y), (bar_x + confidence_width, bar_y + 8), colors.SUCCESS, -1)
                cv2.putText(display, f'{confidence:.1f}%', (bar_x + bar_width + 10, bar_y + 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors.TEXT, 1, cv2.LINE_AA)
                
                # Top 3 emotions
                sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)[:3]
                emotion_texts = []
                for emo, score in sorted_emotions:
                    emo_display = self.emotion_display.get(emo, emo)
                    emotion_texts.append(f'{emo_display} {score:.0f}%')
                
                secondary_text = ' | '.join(emotion_texts)
                self._draw_text_with_shadow(
                    display,
                    secondary_text,
                    (panel_x + 15, y_current + 75),
                    0.45,
                    (180, 180, 200),
                    1
                )
                
                y_current += box_height + 10
            
            # Gestures, Posture, Objects in compact boxes
            info_items = []
            if gestures:
                clean_gestures = [g.encode('ascii', 'ignore').decode('ascii').strip() for g in gestures if g]
                if clean_gestures:
                    info_items.append(('Gestures', ', '.join(clean_gestures), colors.HAND))
            
            if posture:
                clean_posture = posture.encode('ascii', 'ignore').decode('ascii').strip()
                info_items.append(('Posture', clean_posture, colors.POSE))
            
            if objects:
                unique_objects = list(set(objects))[:3]
                info_items.append(('Objects', ', '.join(unique_objects), colors.OBJECT))
            
            for label, value, color in info_items:
                box_height = 40
                overlay = display.copy()
                cv2.rectangle(overlay, (panel_x, y_current), (panel_x + panel_width, y_current + box_height), (20, 20, 30), -1)
                cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)
                cv2.rectangle(display, (panel_x, y_current), (panel_x + panel_width, y_current + box_height), color, 2)
                
                text = f'{label}: {value}'
                self._draw_text_with_shadow(display, text, (panel_x + 15, y_current + 27), 0.5, color, 1)
                
                y_current += box_height + 10
            
            # Context box if present
            if context:
                clean_context = context.encode('ascii', 'ignore').decode('ascii').strip()
                box_height = 45
                overlay = display.copy()
                cv2.rectangle(overlay, (panel_x, y_current), (panel_x + panel_width, y_current + box_height), (50, 100, 150), -1)
                cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)
                cv2.rectangle(display, (panel_x, y_current), (panel_x + panel_width, y_current + box_height), colors.WARNING, 2)
                
                self._draw_text_with_shadow(display, f'CONTEXT: {clean_context}', (panel_x + 15, y_current + 30), 0.55, (255, 255, 255), 2)
        
        # Performance panel (top right) - keep it small
        self._draw_performance_panel(display, stats, width, height)
        
        return display
    
    def _draw_text_with_shadow(
        self,
        image: np.ndarray,
        text: str,
        position: Tuple[int, int],
        scale: float,
        color: Tuple[int, int, int],
        thickness: int
    ):
        """Draw text with shadow effect for better readability."""
        x, y = position
        
        # Shadow
        cv2.putText(
            image,
            text,
            (x + 2, y + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (0, 0, 0),
            thickness + 1,
            cv2.LINE_AA
        )
        
        # Main text
        cv2.putText(
            image,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA
        )
    
    def _draw_performance_panel(
        self,
        image: np.ndarray,
        stats: Dict[str, float],
        width: int,
        height: int
    ):
        """Draw elegant performance metrics panel."""
        panel_width = 200
        panel_height = 100
        panel_x = width - panel_width - 20
        panel_y = 20
        
        # Semi-transparent background
        overlay = image.copy()
        cv2.rectangle(
            overlay,
            (panel_x - 10, panel_y - 10),
            (panel_x + panel_width, panel_y + panel_height),
            (30, 30, 40),
            -1
        )
        cv2.addWeighted(overlay, 0.8, image, 0.2, 0, image)
        
        # Border
        cv2.rectangle(
            image,
            (panel_x - 10, panel_y - 10),
            (panel_x + panel_width, panel_y + panel_height),
            (100, 150, 255),
            2
        )
        
        # FPS with color coding
        fps = stats.get('fps', 0)
        fps_color = colors.SUCCESS if fps > 25 else colors.WARNING if fps > 15 else colors.ERROR
        
        cv2.putText(
            image,
            'PERFORMANCE',
            (panel_x, panel_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
            cv2.LINE_AA
        )
        
        cv2.putText(
            image,
            f'FPS: {fps:.1f}',
            (panel_x, panel_y + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            fps_color,
            2,
            cv2.LINE_AA
        )
        
        frame_text = f'Frame: {stats.get("frame_count", 0)}'
        cv2.putText(
            image,
            frame_text,
            (panel_x, panel_y + 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            colors.TEXT,
            1,
            cv2.LINE_AA
        )
        
        # Frame time
        avg_time = stats.get('avg_frame_time', 0) * 1000
        time_text = f'Time: {avg_time:.1f}ms'
        cv2.putText(
            image,
            time_text,
            (panel_x, panel_y + 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (150, 150, 150),
            1,
            cv2.LINE_AA
        )
    
    def draw_status_bar(
        self,
        image: np.ndarray,
        show_face: bool,
        show_hands: bool,
        show_pose: bool,
        show_objects: bool
    ) -> np.ndarray:
        """Draw elegant status bar showing active detectors."""
        display = image.copy()
        height, width = display.shape[:2]
        
        # Create gradient background for status bar
        overlay = display.copy()
        bar_height = 45
        
        for i in range(bar_height):
            alpha = 0.8 * (i / bar_height * 0.5 + 0.5)
            cv2.rectangle(
                overlay,
                (0, height - bar_height + i),
                (width, height - bar_height + i + 1),
                (20, 20, 30),
                -1
            )
        
        cv2.addWeighted(overlay, 0.85, display, 0.15, 0, display)
        
        # Draw top border
        cv2.line(display, (0, height - bar_height), (width, height - bar_height), (100, 150, 255), 2)
        
        # Status indicators with modern design
        statuses = [
            ("[1] FACE", show_face, colors.FACE),
            ("[2] HANDS", show_hands, colors.HAND),
            ("[3] POSE", show_pose, colors.POSE),
            ("[4] OBJECTS", show_objects, colors.OBJECT),
        ]
        
        x_pos = 20
        y_pos = height - 20
        
        for status_text, is_active, status_color in statuses:
            # Draw indicator circle
            circle_x = x_pos + 10
            circle_y = y_pos - 5
            
            if is_active:
                cv2.circle(display, (circle_x, circle_y), 6, status_color, -1)
                cv2.circle(display, (circle_x, circle_y), 7, (255, 255, 255), 1)
                text_color = status_color
            else:
                cv2.circle(display, (circle_x, circle_y), 6, (60, 60, 60), -1)
                cv2.circle(display, (circle_x, circle_y), 7, (100, 100, 100), 1)
                text_color = (120, 120, 120)
            
            # Draw text
            cv2.putText(
                display,
                status_text,
                (x_pos + 25, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                text_color,
                2 if is_active else 1,
                cv2.LINE_AA
            )
            
            x_pos += 160
        
        # Help text
        help_text = "[Q] Quit  [S] Save  [R] Report"
        text_width = cv2.getTextSize(help_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
        cv2.putText(
            display,
            help_text,
            (width - text_width - 20, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
            cv2.LINE_AA
        )
        
        return display
