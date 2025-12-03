"""
Check which models are available and their status.
"""

from pathlib import Path
import sys

def main():
    """Check model availability."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    model_dir = project_root / "models"
    
    required_models = [
        "face_landmarker.task",
        "hand_landmarker.task",
        "pose_landmarker_lite.task",
        "efficientdet_lite0.tflite"
    ]
    
    print("=" * 70)
    print("Model Status Check")
    print("=" * 70)
    print(f"📁 Models directory: {model_dir}\n")
    
    missing = []
    existing = []
    
    for model_name in required_models:
        model_path = model_dir / model_name
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            existing.append(f"✅ {model_name} ({size_mb:.1f} MB)")
        else:
            missing.append(f"❌ {model_name}")
    
    if existing:
        print("Found models:")
        for item in existing:
            print(f"  {item}")
        print()
    
    if missing:
        print("Missing models:")
        for item in missing:
            print(f"  {item}")
        print(f"\n💡 Run: python scripts/download_models.py")
        return 1
    else:
        print("🎉 All models are present!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
