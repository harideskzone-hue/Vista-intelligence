import cv2
import torch
import numpy as np
from typing import List, Tuple
from pathlib import Path
from ultralytics import YOLO

from core.config import config

class PlateDetector:
    """
    Production License Plate Detector using trained YOLO model.
    """
    def __init__(
        self,
        model_path: str = config.plate_model_path,
        conf_threshold: float = config.plate_det_conf_threshold,
        device: str | None = None
    ):
        self.model_path = str(model_path)
        self.conf_threshold = conf_threshold
        
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self.model = YOLO(self.model_path)

    def detect(self, vehicle_crop: np.ndarray) -> List[Tuple[Tuple[int, int, int, int], float]]:
        """
        Detect license plates within a cropped vehicle image.
        
        Returns:
            List of ((x1, y1, x2, y2), confidence) tuples relative to vehicle_crop.
        """
        if vehicle_crop is None or vehicle_crop.size == 0:
            return []

        results = self.model.predict(
            source=vehicle_crop,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False
        )

        detections = []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                lp_box = boxes.xyxy[i].cpu().numpy()
                lp_conf = float(boxes.conf[i].cpu().numpy())
                x1, y1, x2, y2 = map(int, lp_box)
                detections.append(((x1, y1, x2, y2), lp_conf))

        return detections
