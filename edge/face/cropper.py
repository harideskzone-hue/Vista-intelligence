"""
Cropper: Produces consistent 'passport-style' crops from aligned faces.
Retains head and shoulders geometry with fixed aspect ratio.
"""
from __future__ import annotations

import numpy as np
import cv2

from edge.face.types import FaceDetection

class FaceCropper:
    """
    Passport-style crop configuration.
    Not claiming government ID compliance; simply defines structural framing.
    """
    
    ASPECT_RATIO = 3.0 / 4.0  # Width / Height
    TARGET_WIDTH = 300
    TARGET_HEIGHT = 400
    
    # Ratios relative to face bounding box size
    TOP_MARGIN_RATIO = 0.6    # Extra space above head
    BOTTOM_MARGIN_RATIO = 0.9 # Extra space below chin (shoulders)
    SIDE_MARGIN_RATIO = 0.5   # Extra space on sides
    
    @classmethod
    def transform_box(cls, x1: float, y1: float, x2: float, y2: float, M: np.ndarray) -> tuple[float, float, float, float]:
        """Transform a bounding box using affine matrix M."""
        points = np.array([
            [x1, y1, 1],
            [x2, y1, 1],
            [x2, y2, 1],
            [x1, y2, 1]
        ])
        transformed = np.dot(M, points.T).T
        
        nx1 = np.min(transformed[:, 0])
        ny1 = np.min(transformed[:, 1])
        nx2 = np.max(transformed[:, 0])
        ny2 = np.max(transformed[:, 1])
        
        return nx1, ny1, nx2, ny2

    @classmethod
    def passport_style_crop(cls, aligned_image: np.ndarray, detection: FaceDetection, M: np.ndarray | None = None) -> np.ndarray:
        """
        Produces a consistent head-and-shoulders crop.
        """
        x1, y1, x2, y2 = detection.x1, detection.y1, detection.x2, detection.y2
        
        if M is not None:
            x1, y1, x2, y2 = cls.transform_box(x1, y1, x2, y2, M)
            
        face_w = x2 - x1
        face_h = y2 - y1
        
        if face_w <= 0 or face_h <= 0:
            raise ValueError("Invalid detection bounds for cropping.")
            
        # Calculate ideal crop box
        cx = x1 + face_w / 2.0
        cy = y1 + face_h / 2.0
        
        crop_w = face_w * (1.0 + 2 * cls.SIDE_MARGIN_RATIO)
        crop_h = face_h * (1.0 + cls.TOP_MARGIN_RATIO + cls.BOTTOM_MARGIN_RATIO)
        
        # Enforce aspect ratio
        current_ratio = crop_w / crop_h
        if current_ratio > cls.ASPECT_RATIO:
            crop_h = crop_w / cls.ASPECT_RATIO
        else:
            crop_w = crop_h * cls.ASPECT_RATIO
            
        # Shift center vertically based on margins
        # The face center cy is shifted up relative to the crop center
        shift_y = (cls.BOTTOM_MARGIN_RATIO - cls.TOP_MARGIN_RATIO) * face_h / 2.0
        crop_cy = cy + shift_y
        
        cx1 = int(cx - crop_w / 2.0)
        cy1 = int(crop_cy - crop_h / 2.0)
        cx2 = int(cx + crop_w / 2.0)
        cy2 = int(crop_cy + crop_h / 2.0)
        
        # Pad if outside image bounds to preserve deterministic framing
        h, w = aligned_image.shape[:2]
        pad_top = max(0, -cy1)
        pad_bottom = max(0, cy2 - h)
        pad_left = max(0, -cx1)
        pad_right = max(0, cx2 - w)
        
        if any((pad_top, pad_bottom, pad_left, pad_right)):
            padded = cv2.copyMakeBorder(
                aligned_image, 
                pad_top, pad_bottom, pad_left, pad_right, 
                cv2.BORDER_REPLICATE
            )
            cy1 += pad_top
            cy2 += pad_top
            cx1 += pad_left
            cx2 += pad_left
        else:
            padded = aligned_image
            
        crop = padded[cy1:cy2, cx1:cx2]
        
        if crop.size == 0:
            raise ValueError("Crop resulted in an empty image.")
            
        # Final deterministic resize
        return cv2.resize(crop, (cls.TARGET_WIDTH, cls.TARGET_HEIGHT), interpolation=cv2.INTER_AREA)
