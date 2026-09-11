from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum
import time

class PlateStatus(str, Enum):
    NO_PLATE_DETECTED = "NO_PLATE_DETECTED"
    INSUFFICIENT_RESOLUTION = "INSUFFICIENT_RESOLUTION"
    LOW_QUALITY = "LOW_QUALITY"
    OCR_UNCERTAIN = "OCR_UNCERTAIN"
    PROBABLE = "PROBABLE"
    CONFIRMED = "CONFIRMED"
    UNKNOWN = "UNKNOWN"

class PlateQualityState(str, Enum):
    READABLE = "READABLE"
    LOW_QUALITY = "LOW_QUALITY"
    INSUFFICIENT_RESOLUTION = "INSUFFICIENT_RESOLUTION"

class OpticalQuality(BaseModel):
    width: int = 0
    height: int = 0
    area: int = 0
    aspect_ratio: float = 0.0
    sharpness: float = 0.0
    contrast: float = 0.0

class ANPRResult(BaseModel):
    plate: Optional[str] = None
    status: PlateStatus = PlateStatus.UNKNOWN
    confidence: Optional[float] = None
    format_valid: bool = False
    optical_quality: Optional[OpticalQuality] = None
    observations: int = 0
    temporal_support: int = 0
    first_confirmed_frame: Optional[int] = None
    stable_confirmed_frame: Optional[int] = None

class VehicleClass(str, Enum):
    CAR = "car"
    MOTORCYCLE = "motorcycle"
    BUS = "bus"
    TRUCK = "truck"
    UNKNOWN = "unknown"

class VehicleRecord(BaseModel):
    track_id: int
    trajectory_id: Optional[str] = None
    vehicle_class: VehicleClass
    vehicle_confidence: float
    bbox: List[int] # [x1, y1, x2, y2]
    anpr: ANPRResult = Field(default_factory=ANPRResult)
    first_seen: float
    last_seen: float
    last_frame: int

class CameraInfo(BaseModel):
    id: str = "CAM_01"
    source_type: str = "video" # "video", "rtsp", "webcam"

class ProcessingMetrics(BaseModel):
    frames_processed: int = 0
    effective_fps: float = 0.0

class PipelineMetrics(BaseModel):
    vehicle_detections: int = 0
    unique_tracks: int = 0
    plate_detector_calls: int = 0
    plate_detections: int = 0
    lpr_calls: int = 0
    confirmed: int = 0
    probable: int = 0
    unknown: int = 0

class ReportJSON(BaseModel):
    schema_version: str = "1.0"
    camera: CameraInfo
    processing: ProcessingMetrics
    vehicles: List[VehicleRecord]
    metrics: PipelineMetrics
