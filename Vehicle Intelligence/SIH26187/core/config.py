from pydantic import BaseModel
from typing import Optional

class PipelineConfig(BaseModel):
    # --------------------------------------------------------------------------
    # Frozen A.10 Baseline Scheduling
    # NOTE: min_vehicle_area=10000 is a vehicle-crop scheduling optimization,
    # NOT an optical plate readability threshold.
    # --------------------------------------------------------------------------
    plate_stride: int = 6
    min_vehicle_area: int = 10000
    verification_stride: int = 60
    
    # --------------------------------------------------------------------------
    # Model Artifact Paths
    # --------------------------------------------------------------------------
    vehicle_model_path: str = "yolo11n.pt"
    plate_model_path: str = "anpr/models/best.pt"
    ocr_model_path: str = "data/fast_plate_ocr/models/fine_tuned/2026-09-09_11-29-31/best.onnx"
    ocr_config_path: str = "data/fast_plate_ocr/models/cct_xs_v2_global_plate_config.yaml"
    
    # --------------------------------------------------------------------------
    # Optical Boundary Findings (A.13)
    # The A.13 empirical analysis demonstrated that single-variable cutoffs
    # (such as area or Laplacian sharpness) do NOT produce monotonic readability.
    # Therefore, no artificial single-variable thresholds are established.
    # --------------------------------------------------------------------------
    plate_min_area: Optional[int] = None           # NOT ESTABLISHED
    plate_min_sharpness: Optional[float] = None     # NOT ESTABLISHED
    plate_min_contrast: Optional[float] = None      # NOT ESTABLISHED
    
    # Physical optical sanity bounds (rejecting degenerate/pathological crops only)
    sanity_min_crop_area: int = 100
    sanity_min_crop_dim: int = 8
    sanity_min_aspect_ratio: float = 1.0
    sanity_max_aspect_ratio: float = 8.0
    
    # Detection & Recognition Confidence
    vehicle_conf_threshold: float = 0.40
    plate_det_conf_threshold: float = 0.25
    lpr_min_confidence: float = 0.20
    
    # Temporal Voting Parameters
    min_observations_for_confirmation: int = 3
    min_evidence_for_confirmation: float = 0.10

config = PipelineConfig()
