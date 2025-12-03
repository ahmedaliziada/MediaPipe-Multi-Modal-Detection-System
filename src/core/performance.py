"""
Performance monitoring and metrics tracking.
"""

import time
from typing import Dict, List
from collections import deque


class PerformanceMonitor:
    """Monitors and displays performance metrics."""
    
    def __init__(self, window_size: int = 30):
        """
        Initialize performance monitor.
        
        Args:
            window_size: Number of frames to average over
        """
        self.window_size = window_size
        self.frame_times: deque = deque(maxlen=window_size)
        self.last_fps_update = time.time()
        self.current_fps = 0.0
        self.frame_count = 0
        
        # Detailed timing
        self.detection_times: Dict[str, deque] = {
            'face': deque(maxlen=window_size),
            'hand': deque(maxlen=window_size),
            'pose': deque(maxlen=window_size),
            'object': deque(maxlen=window_size),
            'emotion': deque(maxlen=window_size),
        }
    
    def update(self, frame_time: float):
        """Update performance metrics."""
        self.frame_times.append(frame_time)
        self.frame_count += 1
        
        # Update FPS every second
        current_time = time.time()
        if current_time - self.last_fps_update >= 1.0:
            if self.frame_times:
                avg_time = sum(self.frame_times) / len(self.frame_times)
                self.current_fps = 1.0 / avg_time if avg_time > 0 else 0
            self.last_fps_update = current_time
    
    def update_detection_time(self, detector_name: str, detection_time: float):
        """Update detection time for specific detector."""
        if detector_name in self.detection_times:
            self.detection_times[detector_name].append(detection_time)
    
    def get_stats(self) -> Dict[str, float]:
        """Get current performance statistics."""
        stats = {
            'fps': self.current_fps,
            'avg_frame_time': sum(self.frame_times) / len(self.frame_times) if self.frame_times else 0,
            'frame_count': self.frame_count,
        }
        
        # Add detector-specific times
        for detector, times in self.detection_times.items():
            if times:
                stats[f'{detector}_time'] = sum(times) / len(times)
        
        return stats
    
    def get_detailed_report(self) -> str:
        """Generate detailed performance report."""
        stats = self.get_stats()
        report = [
            "=" * 50,
            "PERFORMANCE REPORT",
            "=" * 50,
            f"Total Frames: {stats['frame_count']}",
            f"Current FPS: {stats['fps']:.2f}",
            f"Avg Frame Time: {stats['avg_frame_time']*1000:.2f}ms",
            "",
            "Detector Times:",
        ]
        
        for detector in ['face', 'hand', 'pose', 'object', 'emotion']:
            key = f'{detector}_time'
            if key in stats:
                report.append(f"  {detector.capitalize()}: {stats[key]*1000:.2f}ms")
        
        report.append("=" * 50)
        return "\n".join(report)
