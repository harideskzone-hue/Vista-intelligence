from dataclasses import dataclass
from typing import List, Tuple, Optional
import torch
import numpy as np
from ultralytics import YOLO

from core.config import config
from core.schemas import VehicleClass
from vehicle_ai.classifier import VehicleClassifier

@dataclass
class TrackedVehicle:
    track_id: int
    vehicle_class: VehicleClass
    confidence: float
    bbox: Tuple[int, int, int, int] # (x1, y1, x2, y2)

    @property
    def x1(self) -> int: return self.bbox[0]
    @property
    def y1(self) -> int: return self.bbox[1]
    @property
    def x2(self) -> int: return self.bbox[2]
    @property
    def y2(self) -> int: return self.bbox[3]
    @property
    def width(self) -> int: return self.x2 - self.x1
    @property
    def height(self) -> int: return self.y2 - self.y1
    @property
    def area(self) -> int: return self.width * self.height

class VehicleTracker:
    """
    Production Vehicle Tracker using YOLO11n and ByteTrack.
    Tracks classes: car (2), motorcycle (3), bus (5), truck (7).
    """
    TRACKED_CLASSES = [2, 3, 5, 7]

    def __init__(
        self,
        model_path: str = config.vehicle_model_path,
        conf_threshold: float = config.vehicle_conf_threshold,
        device: Optional[str] = None
    ):
        self.model_path = str(model_path)
        self.conf_threshold = conf_threshold

        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self.model = YOLO(self.model_path)

    def track(self, frame: np.ndarray) -> List[TrackedVehicle]:
        """
        Processes a single frame, updates ByteTrack tracker, and returns active vehicle tracks.
        """
        if frame is None or frame.size == 0:
            return []

        results = self.model.track(
            source=frame,
            classes=self.TRACKED_CLASSES,
            persist=True,
            tracker="bytetrack.yaml",
            conf=self.conf_threshold,
            device=self.device,
            verbose=False
        )

        tracked_vehicles: List[TrackedVehicle] = []

        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            confs = results[0].boxes.conf.cpu().numpy()
            class_ids = results[0].boxes.cls.int().cpu().tolist()

            for box, track_id, conf, cls_id in zip(boxes, track_ids, confs, class_ids):
                x1, y1, x2, y2 = map(int, box)
                vclass = VehicleClassifier.classify_by_id(cls_id)
                tracked_vehicles.append(
                    TrackedVehicle(
                        track_id=track_id,
                        vehicle_class=vclass,
                        confidence=float(conf),
                        bbox=(x1, y1, x2, y2)
                    )
                )

        return tracked_vehicles
