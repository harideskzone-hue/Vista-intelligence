from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import torch
from ultralytics import YOLO


@dataclass
class VehicleDetection:
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]

    @property
    def x1(self) -> int:
        return self.bbox[0]

    @property
    def y1(self) -> int:
        return self.bbox[1]

    @property
    def x2(self) -> int:
        return self.bbox[2]

    @property
    def y2(self) -> int:
        return self.bbox[3]


class VehicleDetector:
    """
    Production vehicle detector for SIH26187.

    Current model:
        YOLO11n COCO

    Supported vehicle classes:
        car
        motorcycle
        bus
        truck
    """

    VEHICLE_CLASSES = {
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
    }

    def __init__(
        self,
        model_path: str | Path = "yolo11n.pt",
        confidence: float = 0.40,
        device: str | None = None,
    ) -> None:
        self.model_path = str(model_path)
        self.confidence = confidence

        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"

        self.device = device
        self.model = YOLO(self.model_path)

    def detect(self, frame: Any) -> list[VehicleDetection]:
        """
        Detect vehicles in a single BGR OpenCV frame.
        """

        if frame is None:
            raise ValueError("Frame cannot be None")

        if not hasattr(frame, "shape"):
            raise TypeError("Frame must be a NumPy/OpenCV image")

        results = self.model.predict(
            source=frame,
            classes=list(self.VEHICLE_CLASSES.keys()),
            conf=self.confidence,
            device=self.device,
            verbose=False,
        )

        detections: list[VehicleDetection] = []

        for result in results:
            if result.boxes is None:
                continue

            boxes = result.boxes

            for i in range(len(boxes)):
                class_id = int(boxes.cls[i].item())
                confidence = float(boxes.conf[i].item())

                if class_id not in self.VEHICLE_CLASSES:
                    continue

                coordinates = boxes.xyxy[i].tolist()

                x1, y1, x2, y2 = (
                    int(coordinates[0]),
                    int(coordinates[1]),
                    int(coordinates[2]),
                    int(coordinates[3]),
                )

                detections.append(
                    VehicleDetection(
                        class_id=class_id,
                        class_name=self.VEHICLE_CLASSES[class_id],
                        confidence=confidence,
                        bbox=(x1, y1, x2, y2),
                    )
                )

        return detections

    def annotate(
        self,
        frame: Any,
        detections: list[VehicleDetection],
    ) -> Any:
        """
        Draw vehicle detections on an OpenCV frame.
        """

        output = frame.copy()

        for detection in detections:
            x1, y1, x2, y2 = detection.bbox

            label = (
                f"{detection.class_name} "
                f"{detection.confidence:.2f}"
            )

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                (255, 255, 255),
                2,
            )

            cv2.putText(
                output,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        return output
