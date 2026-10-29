"""
Model Downloader Utility
Downloads YuNet (Face Detection) and SFace (Face Recognition) ONNX models
from the official OpenCV Zoo repository.
"""

import os
import sys
import urllib.request

MODELS = {
    "face_detection_yunet_2023mar.onnx": {
        "url": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "min_size": 200_000,  # ~232 KB
        "description": "YuNet Face Detection Model (Mar 2023)"
    },
    "face_recognition_sface_2021dec.onnx": {
        "url": "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "min_size": 35_000_000,  # ~38.6 MB
        "description": "SFace Face Recognition / Embedding Model (Dec 2021)"
    }
}


def download_models(target_dir: str = "models", force: bool = False) -> bool:
    """
    Downloads required ONNX models if they do not exist or are incomplete.
    Returns True if both models are ready.
    """
    os.makedirs(target_dir, exist_ok=True)
    all_ok = True

    for filename, info in MODELS.items():
        filepath = os.path.join(target_dir, filename)
        if not force and os.path.exists(filepath):
            file_size = os.path.getsize(filepath)
            if file_size >= info["min_size"]:
                print(f"[OK] {info['description']} is already present ({file_size / (1024*1024):.2f} MB).")
                continue
            else:
                print(f"[!] {filename} exists but appears corrupted or incomplete ({file_size} bytes). Re-downloading...")

        print(f"[*] Downloading {info['description']}...")
        print(f"    Source: {info['url']}")
        req = urllib.request.Request(info["url"], headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        try:
            with urllib.request.urlopen(req) as resp, open(filepath, "wb") as f:
                total_len = int(resp.headers.get("Content-Length", 0))
                downloaded = 0
                while chunk := resp.read(1024 * 64):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_len > 0:
                        pct = int(downloaded / total_len * 100)
                        bar = "=" * (pct // 5) + " " * (20 - (pct // 5))
                        print(f"\r    Progress: [{bar}] {pct}% ({downloaded / (1024*1024):.2f} MB)", end="", flush=True)
                print()
            print(f"[SUCCESS] Downloaded {filename} ({os.path.getsize(filepath) / (1024*1024):.2f} MB)")
        except Exception as e:
            print(f"[ERROR] Failed to download {filename}: {e}")
            all_ok = False

    return all_ok


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    success = download_models(target_dir=current_dir)
    sys.exit(0 if success else 1)
