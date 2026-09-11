"""
BoundaryTrackingAdapter — Real-Time & Offline Object Detection/Tracking Adapter.

Connects computer vision person/vehicle detectors (Ultralytics YOLO + BoT-SORT / ByteTrack)
to the Boundary Intelligence Subsystem by translating raw model outputs into standardized
TrackObservation instances.

Architecture:
Video / Webcam / RTSP
       │
       ▼
BoundaryTrackingAdapter (YOLO Person Detector + ByteTrack / BoT-SORT)
       │
       ▼
TrackObservation(track_id, bbox, class_name, confidence, timestamp)
       │
       ▼
BoundaryManager.process_frame()
       │
       ▼
EventHub & EventClipManager
"""

import os
import time
import logging
from typing import List, Tuple, Optional, Dict, Any, Generator
import cv2
import numpy as np

from edge.boundary.crossing_engine import TrackObservation

log = logging.getLogger("boundary_adapter")


class BoundaryTrackingAdapter:
    """
    Converts raw camera frames or video sequences into stable TrackObservations.
    """
    DEFAULT_MODEL_CANDIDATES = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models/yolo11n.pt")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../../Vehicle Intelligence/SIH26187/yolo11n.pt")),
        "yolo11n.pt",
        "yolov8n.pt"
    ]

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.35,
        target_classes: Optional[List[int]] = None,
        tracker_type: str = "bytetrack.yaml",
        imgsz: int = 480,
        skip_frames: int = 1
    ):
        self.conf_threshold = conf_threshold
        # COCO class 0 is 'person'. Vehicles are 2 (car), 3 (motorcycle), 5 (bus), 7 (truck)
        self.target_classes = target_classes if target_classes is not None else [0]
        self.tracker_type = tracker_type
        self.imgsz = imgsz
        self.skip_frames = max(1, int(skip_frames))

        self.model = None
        self._model_path = None
        self._frame_count = 0
        self._cached_observations: List[TrackObservation] = []

        self._init_detector(model_path)

    def _init_detector(self, user_model_path: Optional[str]) -> None:
        candidates = [user_model_path] if user_model_path else self.DEFAULT_MODEL_CANDIDATES
        for path in candidates:
            if not path:
                continue
            try:
                from ultralytics import YOLO  # type: ignore
                if os.path.exists(path) or not path.endswith(".pt"):
                    log.info(f"Loading tracking detector model from: {path}")
                    self.model = YOLO(path)
                    self._model_path = path
                    return
            except Exception as e:
                log.warning(f"Could not load detector from {path}: {e}")

        log.warning("Ultralytics detector could not be loaded; falling back to heuristic/synthetic tracker")

    @property
    def is_model_loaded(self) -> bool:
        return self.model is not None

    def reset(self) -> None:
        """Resets multi-object tracking state between video sequences."""
        self._frame_count = 0
        self._cached_observations.clear()
        if self.model is not None and hasattr(self.model, "predictor") and self.model.predictor is not None:
            if hasattr(self.model.predictor, "trackers"):
                self.model.predictor.trackers = []

    def detect_and_track(
        self,
        frame: np.ndarray,
        timestamp: float = 0.0
    ) -> List[TrackObservation]:
        """
        Runs detection and multi-object tracking on a single frame.
        Outputs a list of TrackObservation objects with persistent track_ids.
        """
        self._frame_count += 1

        # If skipping frames, return cached observations
        if self.skip_frames > 1 and (self._frame_count % self.skip_frames != 0) and self._cached_observations:
            # Update timestamp on cached observations
            return [
                TrackObservation(
                    track_id=o.track_id,
                    bbox=o.bbox,
                    class_name=o.class_name,
                    confidence=o.confidence,
                    timestamp=timestamp
                )
                for o in self._cached_observations
            ]

        if self.model is None:
            # Fallback: no detector model loaded
            return self._cached_observations

        try:
            # Run Ultralytics tracking with persistent track state
            results = self.model.track(
                frame,
                persist=True,
                tracker=self.tracker_type,
                classes=self.target_classes,
                conf=self.conf_threshold,
                imgsz=self.imgsz,
                verbose=False
            )

            observations: List[TrackObservation] = []
            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                for b in boxes:
                    # Only tracks with an assigned ID are valid for boundary analysis
                    if b.id is None:
                        continue

                    track_id = int(b.id[0])
                    conf = float(b.conf[0])
                    cls_id = int(b.cls[0])
                    class_name = self.model.names.get(cls_id, "person")
                    xyxy = b.xyxy[0].cpu().numpy().tolist()
                    bbox = (float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]))

                    observations.append(
                        TrackObservation(
                            track_id=track_id,
                            bbox=bbox,
                            class_name=class_name,
                            confidence=conf,
                            timestamp=timestamp
                        )
                    )

            self._cached_observations = observations
            return observations

        except Exception as e:
            log.error(f"Tracking inference failed: {e}", exc_info=True)
            return []

    def iter_video_stream(
        self,
        video_path: str,
        fps_override: Optional[float] = None
    ) -> Generator[Tuple[float, np.ndarray, List[TrackObservation]], None, None]:
        """
        Sequentially decodes a video and yields (timestamp, frame, observations).
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 1.0:
            fps = 25.0
        if fps_override and fps_override > 1.0:
            fps = fps_override

        frame_idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                ts = frame_idx / fps
                obs = self.detect_and_track(frame, timestamp=ts)
                yield ts, frame, obs
                frame_idx += 1
        finally:
            cap.release()


_global_adapter: Optional[BoundaryTrackingAdapter] = None


def get_tracking_adapter() -> BoundaryTrackingAdapter:
    """Returns the global shared tracking adapter singleton."""
    global _global_adapter
    if _global_adapter is None:
        _global_adapter = BoundaryTrackingAdapter()
    return _global_adapter
