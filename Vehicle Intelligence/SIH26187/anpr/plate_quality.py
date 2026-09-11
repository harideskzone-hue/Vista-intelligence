import cv2
import numpy as np
from core.schemas import PlateQualityState, OpticalQuality
from core.config import config

class PlateQualityAnalyzer:
    """
    Multi-dimensional optical quality analyzer for license plate crops.
    Adheres to A.13 empirical findings:
    - Does NOT use arbitrary hard single-variable cutoffs for area or sharpness.
    - Evaluates physical sanity bounds (rejecting microscopic/degenerate crops).
    - Extracts multi-dimensional optical evidence (geometry, Laplacian sharpness, RMS contrast).
    """
    def __init__(
        self,
        min_dim: int = config.sanity_min_crop_dim,
        min_area: int = config.sanity_min_crop_area,
        min_ar: float = config.sanity_min_aspect_ratio,
        max_ar: float = config.sanity_max_aspect_ratio
    ):
        self.min_dim = min_dim
        self.min_area = min_area
        self.min_ar = min_ar
        self.max_ar = max_ar

    @staticmethod
    def laplacian_variance(image: np.ndarray) -> float:
        if image is None or image.size == 0:
            return 0.0
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def contrast_rms(image: np.ndarray) -> float:
        if image is None or image.size == 0:
            return 0.0
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        return float(gray.std())

    def analyze(self, plate_crop: np.ndarray) -> tuple[PlateQualityState, OpticalQuality, float]:
        """
        Analyzes an extracted plate crop.
        
        Returns:
            (quality_state, optical_quality_metrics, normalized_quality_score)
        """
        if plate_crop is None or plate_crop.size == 0:
            return (
                PlateQualityState.INSUFFICIENT_RESOLUTION,
                OpticalQuality(),
                0.0
            )

        h, w = plate_crop.shape[:2]
        area = w * h
        aspect_ratio = float(w) / float(h) if h > 0 else 0.0
        sharpness = self.laplacian_variance(plate_crop)
        contrast = self.contrast_rms(plate_crop)

        metrics = OpticalQuality(
            width=w,
            height=h,
            area=area,
            aspect_ratio=round(aspect_ratio, 2),
            sharpness=round(sharpness, 2),
            contrast=round(contrast, 2)
        )

        # 1. Physical Resolution Check:
        # If crop is too small to contain character strokes (e.g. area < 100 or width < 8)
        if area < self.min_area or w < self.min_dim or h < 5:
            return PlateQualityState.INSUFFICIENT_RESOLUTION, metrics, 0.0

        # 2. Geometric Sanity & Contrast Check:
        # Standard Indian plates are horizontal rectangular plates (AR between 1.0 and 8.0).
        # Near-zero contrast indicates solid black/washed-out frame.
        if aspect_ratio < self.min_ar or aspect_ratio > self.max_ar or contrast < 4.0:
            return PlateQualityState.LOW_QUALITY, metrics, 0.1

        # 3. Sufficient visual evidence for LPR
        # Quality score scaled smoothly for evidence weighting (0.2 to 1.0)
        norm_res = min(1.0, area / 5000.0)
        norm_contrast = min(1.0, contrast / 50.0)
        quality_score = float(np.clip(0.3 * norm_res + 0.7 * norm_contrast, 0.2, 1.0))

        return PlateQualityState.READABLE, metrics, quality_score
