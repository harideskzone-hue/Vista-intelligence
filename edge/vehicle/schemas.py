from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class VehicleObservation(BaseModel):
    camera_id: str
    track_id: int
    timestamp: float
    bbox: List[int]                     # [x1, y1, x2, y2]
    vehicle_class: str                  # car, motorcycle, bus, truck, unknown
    make: Optional[str] = "UNKNOWN"
    model: Optional[str] = "UNKNOWN"
    color: Optional[str] = "UNKNOWN"
    plate_text: Optional[str] = None
    plate_confidence: Optional[float] = None
    plate_status: str                   # NO_PLATE_DETECTED, LOW_QUALITY, OCR_UNCERTAIN, PROBABLE, CONFIRMED
    detection_confidence: float = 0.0
    format_valid: bool = False
    temporal_support: int = 0
    optical_quality: Optional[Dict[str, Any]] = None

class VehicleWatchlistEntry(BaseModel):
    plate_text: str
    reason: Optional[str] = None
    severity: str = "WARNING"
    notes: Optional[str] = None

class VehicleTestResponse(BaseModel):
    success: bool
    vehicles_detected: int
    vehicles: List[VehicleObservation]
    total_frames: int
    duration_seconds: float
    clips_generated: int
    error: Optional[str] = None
