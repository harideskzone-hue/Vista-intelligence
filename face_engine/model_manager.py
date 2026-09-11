"""
model_manager.py — Centralized AI Model Registry & Validator
=============================================================
Auto-downloads missing models with SHA256 verification, supports
fallback mirrors, and provides a warmup function for cold-start prevention.
"""
from __future__ import annotations

import os
import sys
import hashlib
import logging
import shutil
import tempfile
from typing import Any, Dict, List, TypedDict

log = logging.getLogger("model_manager")


class ModelInfo(TypedDict):
    urls: List[str]
    sha256: str
    size_mb: float
    required: bool

# ── Model Registry ──────────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

MODEL_REGISTRY: Dict[str, ModelInfo] = {
    "yolo26n.pt": {
        "urls": [
            "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
            "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt",
        ],
        "sha256": "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef",
        "size_mb": 5.3,
        "required": False,
    },
    "yolo26n-face.pt": {
        "urls": [
            "https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolo26n-face.pt",
        ],
        "sha256": "c6a5405127a2e351292315a6a8084ea3e790dbec25b9d16a8e80d1e3f866efe1",
        "size_mb": 5.6,
        "required": False,   # fallback: body-only detection still works
    },
    "face_landmarker.task": {
        "urls": [
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        ],
        "sha256": "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
        "size_mb": 3.6,
        "required": False,
    },
    "pose_landmarker_full.task": {
        "urls": [
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
        ],
        "sha256": "4eaa5eb7a98365221087693fcc286334cf0858e2eb6e15b506aa4a7ecdcec4ad",
        "size_mb": 9.0,
        "required": False,
    },
}


def _sha256_file(path: str) -> str:
    """Compute SHA256 of a file in 64KB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_models() -> dict:
    """
    Check all models: exist + SHA256 match.
    Returns dict of {filename: {"exists": bool, "sha_ok": bool, "path": str}}
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    results = {}
    for fname, info in MODEL_REGISTRY.items():
        path = os.path.join(MODELS_DIR, fname)
        exists = os.path.isfile(path)
        sha_ok = False
        if exists:
            actual = _sha256_file(path)
            sha_ok = actual == info["sha256"]
            if not sha_ok:
                log.warning("Checksum mismatch for %s: expected=%s got=%s",
                            fname, info["sha256"][:16], actual[:16])  # type: ignore
        results[fname] = {
            "exists": exists,
            "sha_ok": sha_ok,
            "path": path,
            "required": info["required"],
            "size_mb": info["size_mb"],
        }
    return results


def download_missing(force: bool = False) -> bool:
    """
    Download any missing or checksum-mismatched models.
    Uses atomic rename to prevent corrupted partial downloads.
    Returns True if all required models are ready.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    status = validate_models()
    all_ok = True

    for fname, info in MODEL_REGISTRY.items():
        s = status[fname]
        if s["exists"] and s["sha_ok"] and not force:
            continue

        dest = os.path.join(MODELS_DIR, fname)
        log.info("Downloading %s (%.1f MB)...", fname, info["size_mb"])
        print(f"  ⬇  Downloading {fname} ({info['size_mb']:.1f} MB)...")

        downloaded = False
        for url in info["urls"]:
            try:
                import urllib.request
                tmp_fd, tmp_path = tempfile.mkstemp(dir=MODELS_DIR, suffix=".download")
                os.close(tmp_fd)

                urllib.request.urlretrieve(url, tmp_path)

                # Verify checksum before committing
                actual_sha = _sha256_file(tmp_path)
                if actual_sha != info["sha256"]:
                    log.warning("Checksum mismatch after download from %s", url)
                    os.unlink(tmp_path)
                    continue

                # Atomic rename
                shutil.move(tmp_path, dest)
                log.info("✓ Downloaded and verified %s", fname)
                print(f"  ✓  {fname} downloaded and verified.")
                downloaded = True
                break

            except Exception as e:
                log.warning("Download failed from %s: %s", url, e)
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                continue

        if not downloaded:
            if info["required"]:
                log.error("FATAL: Could not download required model: %s", fname)
                print(f"  ✗  FATAL: Could not download {fname}")
                all_ok = False
            else:
                log.warning("Optional model unavailable: %s (pipeline will use fallback)", fname)
                print(f"  ⚠  Optional model {fname} unavailable (fallback active)")

    return all_ok


def warmup_models() -> Dict[str, Any]:
    """
    Preload YOLO and InsightFace models to avoid cold-start delays.
    Returns timing info for each model.
    """
    import time
    results: Dict[str, Any] = {}

    # Warmup YOLO
    yolo_path = os.path.join(MODELS_DIR, "yolo26n.pt")
    if os.path.isfile(yolo_path):
        try:
            t0 = time.time()
            from ultralytics import YOLO  # type: ignore
            _model = YOLO(yolo_path)
            results["yolo"] = {"loaded": True, "time_s": round(time.time() - t0, 2)}  # type: ignore
            log.info("YOLO warmup: %.2fs", results["yolo"]["time_s"])
        except Exception as e:
            results["yolo"] = {"loaded": False, "error": str(e)}
            log.warning("YOLO warmup failed: %s", e)
    else:
        results["yolo"] = {"loaded": False, "error": "file not found"}

    # Warmup InsightFace
    try:
        t0 = time.time()
        import insightface  # type: ignore
        _app = insightface.app.FaceAnalysis(name="buffalo_sc", providers=["CPUExecutionProvider"])
        _app.prepare(ctx_id=0, det_size=(640, 640))
        results["insightface"] = {"loaded": True, "time_s": round(time.time() - t0, 2)}  # type: ignore
        log.info("InsightFace warmup: %.2fs", results["insightface"]["time_s"])
    except Exception as e:
        results["insightface"] = {"loaded": False, "error": str(e)}
        log.warning("InsightFace warmup failed: %s", e)

    return results


def print_status_table():
    """Print a pretty preflight status table for all models."""
    status = validate_models()
    print("\n┌─ MODEL PREFLIGHT ─────────────────────────────────────┐")
    for fname, s in status.items():
        if s["exists"] and s["sha_ok"]:
            sym = "✓"
            detail = f"{s['size_mb']:.1f} MB, SHA OK"
        elif s["exists"]:
            sym = "⚠"
            detail = "checksum mismatch"
        else:
            sym = "✗"
            detail = "MISSING"
            if not s["required"]:
                detail += " (optional)"
        print(f"│ {sym} {fname:<28} ({detail:<20}) │")
    print("└───────────────────────────────────────────────────────┘\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print_status_table()
    if not download_missing():
        print("Some required models are unavailable.")
        sys.exit(1)
    print_status_table()
    print("All models ready.")
