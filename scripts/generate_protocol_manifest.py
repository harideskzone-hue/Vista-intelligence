import os
import json
import hashlib
import subprocess
import sys

def get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    except Exception:
        return "unknown"

def hash_file(filepath):
    if not os.path.exists(filepath):
        return "missing"
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()

def generate_manifest():
    manifest = {
        "git_commit": get_git_commit(),
        "git_tag": "phase7-final-validated",
        
        "inference_mode": "mps_batch",
        "batch_size": 1,
        "optimize_scheduling": True,
        "lazy_encoding": True,
        
        "models": {
            "person_detector": {
                "weights": "models/yolo26n-person.pt",
                "sha256": hash_file("models/yolo26n-person.pt")
            },
            "face_detector": {
                "weights": "models/yolo26n-face.pt",
                "sha256": hash_file("models/yolo26n-face.pt")
            },
            "face_recognizer": {
                "weights": "FaceAnalyzer (InsightFace internal)",
                "sha256": "N/A - library internal"
            },
            "anpr": {
                "weights": "TBD",
                "sha256": "TBD"
            }
        },
        
        "thresholds": {
            "yolo_confidence": 0.5,
            "yolo_iou": 0.7,
            "tracker_correlation": 0.65,
            "face_margin": "TBD"
        },
        "input_resolution": "Auto (640x640 padding)",
        "device": "MPS",
        "python": sys.version.split(' ')[0]
    }
    
    with open("phase8_protocol.json", "w") as f:
        json.dump(manifest, f, indent=2)
        
    print("[✓] phase8_protocol.json successfully generated and frozen.")

if __name__ == "__main__":
    generate_manifest()
