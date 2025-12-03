"""
MediaPipe Multi-Modal Detection System - Main Entry Point
Professional computer vision system for real-time analysis.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.system import MediaPipeDetectionSystem
from src.core.logger import logger


def main():
    """Main entry point."""
    try:
        logger.info("=" * 70)
        logger.info(" " * 15 + "MediaPipe Multi-Modal Detection System")
        logger.info(" " * 20 + "Version 2.0.0 - Professional Edition")
        logger.info("=" * 70)
        
        # Create system instance
        system = MediaPipeDetectionSystem()
        
        # Load models
        if not system.load_models():
            logger.error("Failed to load models. Exiting.")
            return 1
        
        # Initialize camera
        if not system.initialize_camera():
            logger.error("Failed to initialize camera. Exiting.")
            return 1
        
        # Run detection
        system.run()
        
        return 0
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
