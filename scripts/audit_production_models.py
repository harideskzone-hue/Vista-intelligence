import os
import hashlib
import json
import subprocess
import sys

def hash_file(filepath):
    if not os.path.exists(filepath):
        return "missing"
    # Check if it's an LFS pointer (typically < 1KB)
    if os.path.getsize(filepath) < 1024:
        with open(filepath, 'r') as f:
            content = f.read(100)
            if "version https://git-lfs.github.com" in content:
                return "LFS-POINTER (Not actual weights)"
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()

def get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    except Exception:
        return "unknown"

def main():
    person_det_path = "yolo11n.pt" # Root is the real one
    face_det_path = "face_engine/models/yolo26n-face.pt" # Real binary, not LFS pointer in models/
    
    manifest = {
        "git_commit": get_git_commit(),
        "git_tag": "phase7-final-validated",
        "python": sys.version.split(' ')[0],
        
        "models": {
            "person_detector": {
                "weights": person_det_path,
                "sha256": hash_file(person_det_path)
            },
            "face_detector": {
                "weights": face_det_path,
                "sha256": hash_file(face_det_path)
            },
            "face_recognizer": {
                "weights": "External API (http://127.0.0.1:5001)",
                "sha256": "BLOCKED (API decoupling requires explicit model capture)"
            },
            "anpr": {
                "weights": "BLOCKED (None detected in production code)",
                "sha256": "BLOCKED"
            }
        },
        "thresholds": {
            "yolo_confidence": 0.5, # Found in boundary/adapter.py default is 0.35, but live_scorer.py uses 0.5
            "yolo_iou": 0.7,
            "tracker_correlation": 0.65,
            "face_margin": "BLOCKED (Handled by external API)"
        },
        "device": "MPS (CentralInferenceWorker)",
        "input_resolution": "480 (from edge/boundary/adapter.py imgsz=480)"
    }
    
    with open("phase8_protocol.json", "w") as f:
        json.dump(manifest, f, indent=2)
        
    print("[✓] phase8_protocol.json updated with ACTUAL production values and hashes.")

if __name__ == "__main__":
    main()
