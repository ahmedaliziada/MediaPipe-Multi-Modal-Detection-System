"""
Main detection system orchestrator.
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime

from src.core.config import (
    camera_config, processing_config, model_paths,
    detection_thresholds, OUTPUT_DIR
)
from src.core.logger import logger
from src.core.performance import PerformanceMonitor
from src.detectors import (
    EmotionAnalyzer, GestureDetector,
    PostureAnalyzer, ContextAnalyzer
)
from src.visualization.renderer import VisualizationRenderer


class MediaPipeDetectionSystem:
    """Main detection system orchestrator."""
    
    def __init__(self):
        """Initialize the detection system."""
        self.detectors_loaded = False
        self.cap: Optional[cv2.VideoCapture] = None
        
        # Modules
        self.performance_monitor = PerformanceMonitor()
        self.emotion_analyzer = EmotionAnalyzer()
        self.gesture_detector = GestureDetector()
        self.posture_analyzer = PostureAnalyzer()
        self.context_analyzer = ContextAnalyzer()
        self.renderer = VisualizationRenderer()
        
        # Toggle states
        self.show_face = True
        self.show_hands = True
        self.show_pose = True
        self.show_objects = True
        
        # Last analysis results
        self.last_emotion_scores: Optional[Dict[str, float]] = None
        self.last_gestures: List[str] = []
        self.last_posture: Optional[str] = None
        self.last_objects: List[str] = []
        
        logger.info("Detection system initialized")
    
    def load_models(self) -> bool:
        """Load all MediaPipe models."""
        try:
            logger.info("Loading MediaPipe models...")
            model_paths.validate()
            
            # Face Landmarker
            logger.info("  Loading Face Landmarker...")
            face_options = vision.FaceLandmarkerOptions(
                base_options=python.BaseOptions(
                    model_asset_path=str(model_paths.face_landmarker)
                ),
                num_faces=processing_config.max_detection_faces
            )
            self.face_detector = vision.FaceLandmarker.create_from_options(face_options)
            
            # Hand Landmarker
            logger.info("  Loading Hand Landmarker...")
            hand_options = vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(
                    model_asset_path=str(model_paths.hand_landmarker)
                ),
                num_hands=processing_config.max_detection_hands
            )
            self.hand_detector = vision.HandLandmarker.create_from_options(hand_options)
            
            # Pose Landmarker
            logger.info("  Loading Pose Landmarker...")
            pose_options = vision.PoseLandmarkerOptions(
                base_options=python.BaseOptions(
                    model_asset_path=str(model_paths.pose_landmarker)
                )
            )
            self.pose_detector = vision.PoseLandmarker.create_from_options(pose_options)
            
            # Object Detector
            logger.info("  Loading Object Detector...")
            object_options = vision.ObjectDetectorOptions(
                base_options=python.BaseOptions(
                    model_asset_path=str(model_paths.object_detector)
                ),
                score_threshold=detection_thresholds.object_confidence
            )
            self.object_detector = vision.ObjectDetector.create_from_options(object_options)
            
            self.detectors_loaded = True
            logger.info("✓ All models loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            return False
    
    def initialize_camera(self) -> bool:
        """Initialize camera with proper error handling."""
        try:
            logger.info(f"Opening camera at index {camera_config.index}...")
            
            self.cap = cv2.VideoCapture(camera_config.index, cv2.CAP_AVFOUNDATION)
            time.sleep(1)
            
            if not self.cap.isOpened():
                logger.warning("AVFoundation backend failed, trying default...")
                self.cap = cv2.VideoCapture(camera_config.index)
                time.sleep(1)
            
            if not self.cap.isOpened():
                logger.error("Could not open camera!")
                return False
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_config.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_config.height)
            
            logger.info("Warming up camera...")
            for i in range(camera_config.warmup_frames):
                ret, _ = self.cap.read()
                if not ret:
                    logger.warning(f"Warmup frame {i+1} failed")
                time.sleep(0.1)
            
            logger.info("✓ Camera ready!")
            return True
            
        except Exception as e:
            logger.error(f"Camera initialization failed: {e}")
            return False
    
    def process_frame(self, frame: np.ndarray, frame_count: int) -> Optional[np.ndarray]:
        """Process a single frame through all detectors."""
        try:
            frame_start_time = time.time()
            
            # Convert to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            display_frame = rgb_frame.copy()
            
            # Face detection & emotion
            if self.show_face:
                detection_start = time.time()
                face_result = self.face_detector.detect(mp_image)
                self.performance_monitor.update_detection_time('face', time.time() - detection_start)
                
                display_frame = self.renderer.draw_face_landmarks(display_frame, face_result)
                
                if face_result.face_landmarks and frame_count % processing_config.emotion_frame_skip == 0:
                    emotion_start = time.time()
                    face_landmarks = face_result.face_landmarks[0]
                    height, width = rgb_frame.shape[:2]
                    
                    x_coords = [lm.x * width for lm in face_landmarks]
                    y_coords = [lm.y * height for lm in face_landmarks]
                    
                    padding = 20
                    x1 = max(0, int(min(x_coords)) - padding)
                    y1 = max(0, int(min(y_coords)) - padding)
                    x2 = min(width, int(max(x_coords)) + padding)
                    y2 = min(height, int(max(y_coords)) + padding)
                    
                    face_roi = rgb_frame[y1:y2, x1:x2]
                    self.last_emotion_scores = self.emotion_analyzer.detect(face_roi, face_landmarks)
                    self.performance_monitor.update_detection_time('emotion', time.time() - emotion_start)
            
            # Hand detection & gestures
            if self.show_hands:
                detection_start = time.time()
                hand_result = self.hand_detector.detect(mp_image)
                self.performance_monitor.update_detection_time('hand', time.time() - detection_start)
                
                self.last_gestures = []
                if hand_result.hand_landmarks:
                    for hand_landmarks in hand_result.hand_landmarks:
                        gesture = self.gesture_detector.detect(hand_landmarks)
                        self.last_gestures.append(gesture)
                
                display_frame = self.renderer.draw_hand_landmarks(
                    display_frame, hand_result, self.last_gestures
                )
            
            # Pose detection & posture
            if self.show_pose:
                detection_start = time.time()
                pose_result = self.pose_detector.detect(mp_image)
                self.performance_monitor.update_detection_time('pose', time.time() - detection_start)
                
                self.last_posture = None
                if pose_result.pose_landmarks:
                    for pose_landmarks in pose_result.pose_landmarks:
                        self.last_posture = self.posture_analyzer.detect(pose_landmarks)
                
                display_frame = self.renderer.draw_pose_landmarks(
                    display_frame, pose_result, self.last_posture
                )
            
            # Object detection
            if self.show_objects:
                detection_start = time.time()
                object_result = self.object_detector.detect(mp_image)
                self.performance_monitor.update_detection_time('object', time.time() - detection_start)
                
                display_frame, self.last_objects = self.renderer.draw_object_detections(
                    display_frame, object_result
                )
            
            # Context analysis
            context = None
            if self.last_emotion_scores:
                dominant_emotion = max(self.last_emotion_scores, key=self.last_emotion_scores.get)
                context = self.context_analyzer.detect(
                    dominant_emotion,
                    self.last_gestures,
                    self.last_posture,
                    self.last_objects
                )
            
            # Create visualization
            stats = self.performance_monitor.get_stats()
            display_frame = self.renderer.create_info_panel(
                display_frame,
                self.last_emotion_scores,
                self.last_gestures,
                self.last_posture,
                self.last_objects,
                context,
                stats
            )
            
            display_frame = self.renderer.draw_status_bar(
                display_frame,
                self.show_face,
                self.show_hands,
                self.show_pose,
                self.show_objects
            )
            
            # Convert back to BGR
            display_frame = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)
            
            # Update performance
            frame_time = time.time() - frame_start_time
            self.performance_monitor.update(frame_time)
            
            return display_frame
            
        except Exception as e:
            logger.error(f"Frame processing failed: {e}")
            return None
    
    def save_frame(self, frame: np.ndarray, frame_count: int):
        """Save current frame to disk."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = OUTPUT_DIR / "frames" / f"frame_{timestamp}_#{frame_count}.jpg"
            cv2.imwrite(str(filename), frame)
            logger.info(f"Saved frame to: {filename}")
        except Exception as e:
            logger.error(f"Failed to save frame: {e}")
    
    def handle_key_press(self, key: int, frame: np.ndarray, frame_count: int) -> bool:
        """Handle keyboard inputs."""
        if key == ord('q'):
            return False
        elif key == ord('s'):
            self.save_frame(frame, frame_count)
        elif key == ord('1'):
            self.show_face = not self.show_face
            logger.info(f"Face detection: {'ON' if self.show_face else 'OFF'}")
        elif key == ord('2'):
            self.show_hands = not self.show_hands
            logger.info(f"Hand detection: {'ON' if self.show_hands else 'OFF'}")
        elif key == ord('3'):
            self.show_pose = not self.show_pose
            logger.info(f"Pose detection: {'ON' if self.show_pose else 'OFF'}")
        elif key == ord('4'):
            self.show_objects = not self.show_objects
            logger.info(f"Object detection: {'ON' if self.show_objects else 'OFF'}")
        elif key == ord('r'):
            logger.info("\n" + self.performance_monitor.get_detailed_report())
        
        return True
    
    def run(self):
        """Main application loop."""
        if not self.detectors_loaded:
            logger.error("Models not loaded. Call load_models() first.")
            return
        
        if not self.cap or not self.cap.isOpened():
            logger.error("Camera not initialized. Call initialize_camera() first.")
            return
        
        logger.info("\n" + "=" * 60)
        logger.info("CONTROLS")
        logger.info("=" * 60)
        logger.info("[Q] Quit application")
        logger.info("[S] Save current frame")
        logger.info("[1] Toggle face detection")
        logger.info("[2] Toggle hand detection")
        logger.info("[3] Toggle pose detection")
        logger.info("[4] Toggle object detection")
        logger.info("[R] Show performance report")
        logger.info("=" * 60 + "\n")
        
        # Create larger window
        window_name = 'MediaPipe Detection System'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)  # Set initial size
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = self.cap.read()
                
                if not ret or frame is None:
                    logger.warning("Failed to capture frame")
                    continue
                
                frame_count += 1
                display_frame = self.process_frame(frame, frame_count)
                
                if display_frame is not None:
                    cv2.imshow(window_name, display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if not self.handle_key_press(key, display_frame, frame_count):
                    break
                    
        except KeyboardInterrupt:
            logger.info("\nInterrupted by user")
        except Exception as e:
            logger.error(f"Runtime error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        logger.info("\nCleaning up...")
        
        if self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        
        logger.info("\n" + self.performance_monitor.get_detailed_report())
        logger.info("System shutdown complete")
