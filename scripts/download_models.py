"""
Utility script to download MediaPipe model files.
"""

import urllib.request
from pathlib import Path
import sys

# Model URLs with alternatives
MODELS = {
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "pose_landmarker_lite.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "efficientdet_lite0.tflite": "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite",
}

# Alternative URLs if primary fails
ALTERNATIVE_MODELS = {
    "pose_landmarker_lite.task": [
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
    ]
}


def download_model(name: str, url: str, output_dir: Path) -> bool:
    """Download a single model file with progress indicator."""
    output_path = output_dir / name
    
    if output_path.exists():
        file_size = output_path.stat().st_size / (1024 * 1024)
        print(f"✓ {name} already exists ({file_size:.1f} MB), skipping...")
        return True
    
    try:
        print(f"⬇️  Downloading {name}...", end=" ", flush=True)
        
        def show_progress(block_num, block_size, total_size):
            if total_size > 0:
                downloaded = block_num * block_size
                percent = min(downloaded * 100 / total_size, 100)
                print(f"\r⬇️  Downloading {name}... {percent:.1f}%", end="", flush=True)
        
        urllib.request.urlretrieve(url, output_path, reporthook=show_progress)
        file_size = output_path.stat().st_size / (1024 * 1024)
        print(f"\r✓ Downloaded {name} ({file_size:.1f} MB)    ")
        return True
    except Exception as e:
        print(f"\r✗ Failed to download {name}: {e}")
        if output_path.exists():
            output_path.unlink()
        
        # Try alternatives if available
        if name in ALTERNATIVE_MODELS:
            for alt_url in ALTERNATIVE_MODELS[name]:
                print(f"  Trying alternative URL...")
                try:
                    urllib.request.urlretrieve(alt_url, output_path, reporthook=show_progress)
                    file_size = output_path.stat().st_size / (1024 * 1024)
                    print(f"\r✓ Downloaded {name} from alternative ({file_size:.1f} MB)    ")
                    return True
                except:
                    continue
        
        return False


def main():
    """Main download function."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    model_dir = project_root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("  MediaPipe Model Downloader")
    print("=" * 70)
    print(f"📁 Download location: {model_dir}")
    print(f"📦 Total models: {len(MODELS)}")
    print(f"💾 Total size: ~25 MB\n")
    
    success_count = 0
    total = len(MODELS)
    
    for name, url in MODELS.items():
        if download_model(name, url, model_dir):
            success_count += 1
        print()
    
    print("=" * 70)
    if success_count == total:
        print(f"✅ Successfully downloaded all {total} models!")
        print(f"\n📂 Models location: {model_dir}")
        print(f"\n🚀 You can now run: python app.py")
        return 0
    else:
        print(f"⚠️  Downloaded {success_count}/{total} models")
        print(f"❌ {total - success_count} model(s) failed")
        print(f"\n💡 Check your internet connection and try again")
        return 1


if __name__ == "__main__":
    sys.exit(main())
