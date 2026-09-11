import os
import sys
import cv2
import numpy as np
from typing import List, Optional, Dict, Any

# We need to import the SIH26187 package from the submodule.
# Add to sys.path so we can import from it.
V_INT_PATH = "/Users/hariharans/Documents/SIH26187/Vehicle Intelligence/SIH26187"
if V_INT_PATH not in sys.path:
    sys.path.insert(0, V_INT_PATH)

from core.config import PipelineConfig
from integration.vehicle_anpr_pipeline import VehicleANPRPipeline
from core.schemas import VehicleRecord
from edge.vehicle.schemas import VehicleObservation

class VehicleIntelligenceAdapter:
    def __init__(self, camera_id: str = "CAM_01"):
        self.camera_id = camera_id
        
        # Configure absolute paths for the pipeline
        self.config = PipelineConfig()
        
        base_dir = V_INT_PATH
        self.config.vehicle_model_path = os.path.join(base_dir, "yolo11n.pt")
        self.config.plate_model_path = os.path.join(base_dir, "anpr/models/best.pt")
        
        # Fetch the original paths from the default config and prepend base_dir
        default_config = PipelineConfig()
        self.config.ocr_model_path = os.path.join(base_dir, default_config.ocr_model_path)
        self.config.ocr_config_path = os.path.join(base_dir, default_config.ocr_config_path)
        
        self.pipeline = VehicleANPRPipeline(
            camera_id=self.camera_id,
            custom_config=self.config
        )

    def extract_dominant_color(self, vehicle_crop: np.ndarray) -> str:
        """
        Fast HSV-based color estimation on the vehicle body crop.
        (White, Black, Silver/Gray, Red, Blue, Green, Yellow, Orange, Brown)
        """
        if vehicle_crop.size == 0:
            return "UNKNOWN"
            
        hsv = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2HSV)
        
        h, s, v = cv2.split(hsv)
        
        mask_black = (v < 50)
        mask_white = (v > 200) & (s < 30)
        mask_gray = (v >= 50) & (v <= 200) & (s < 40)
        
        mask_red1 = (h < 10) & (s >= 40) & (v >= 50)
        mask_red2 = (h > 170) & (s >= 40) & (v >= 50)
        mask_red = mask_red1 | mask_red2
        
        mask_orange = (h >= 10) & (h < 25) & (s >= 40) & (v >= 50)
        mask_yellow = (h >= 25) & (h < 35) & (s >= 40) & (v >= 50)
        mask_green = (h >= 35) & (h < 85) & (s >= 40) & (v >= 50)
        mask_blue = (h >= 85) & (h < 130) & (s >= 40) & (v >= 50)
        
        counts = {
            "Black": np.count_nonzero(mask_black),
            "White": np.count_nonzero(mask_white),
            "Silver/Gray": np.count_nonzero(mask_gray),
            "Red": np.count_nonzero(mask_red),
            "Orange": np.count_nonzero(mask_orange),
            "Yellow": np.count_nonzero(mask_yellow),
            "Green": np.count_nonzero(mask_green),
            "Blue": np.count_nonzero(mask_blue)
        }
        
        dominant_color = max(counts.items(), key=lambda x: x[1])
        if dominant_color[1] > 0:
            return dominant_color[0]
        return "UNKNOWN"

    def process_frame(self, frame: np.ndarray, timestamp: float) -> List[VehicleObservation]:
        """
        Process a single frame and return VISTA normalized vehicle observations.
        """
        records = self.pipeline.process_frame(frame, timestamp=timestamp)
        
        observations = []
        for rec in records:
            observations.append(self.normalize_record(rec, frame, timestamp))
            
        return observations

    def normalize_record(self, record: VehicleRecord, frame: np.ndarray, timestamp: float) -> VehicleObservation:
        """
        Convert native VehicleRecord to VISTA VehicleObservation.
        """
        x1, y1, x2, y2 = map(int, record.bbox)
        h, w = frame.shape[:2]
        x1_c = max(0, min(w, x1))
        y1_c = max(0, min(h, y1))
        x2_c = max(0, min(w, x2))
        y2_c = max(0, min(h, y2))
        
        veh_crop = frame[y1_c:y2_c, x1_c:x2_c]
        color = self.extract_dominant_color(veh_crop)
        
        opt_qual = None
        if record.anpr.optical_quality:
            # handle both pydantic v1 and v2 dict methods
            if hasattr(record.anpr.optical_quality, "model_dump"):
                opt_qual = record.anpr.optical_quality.model_dump()
            elif hasattr(record.anpr.optical_quality, "dict"):
                opt_qual = record.anpr.optical_quality.dict()
            
        obs = VehicleObservation(
            camera_id=self.camera_id,
            track_id=record.track_id,
            timestamp=timestamp,
            bbox=record.bbox,
            vehicle_class=record.vehicle_class.value if hasattr(record.vehicle_class, "value") else str(record.vehicle_class),
            make="UNKNOWN", 
            model="UNKNOWN", 
            color=color,
            plate_text=record.anpr.plate,
            plate_confidence=record.anpr.confidence,
            plate_status=record.anpr.status.value if hasattr(record.anpr.status, "value") else str(record.anpr.status),
            detection_confidence=record.vehicle_confidence,
            format_valid=record.anpr.format_valid,
            temporal_support=record.anpr.temporal_support,
            optical_quality=opt_qual
        )
        return obs
