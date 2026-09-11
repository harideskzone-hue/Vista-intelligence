"""
Aligner: Provides deterministic facial alignment.
Focuses on preserving facial detail without fabricating landmarks.
"""
from __future__ import annotations

import numpy as np
import cv2
from edge.face.types import FaceDetection

class AlignmentError(Exception):
    pass

class FaceAligner:
    """
    Aligns a face image to level the eyes horizontally.
    Does not aggressively crop, preserving head and shoulders for later stages.
    """
    
    @staticmethod
    def align(image: np.ndarray, detection: FaceDetection) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            aligned_image (np.ndarray): The rotationally aligned image.
            rotation_matrix (np.ndarray): The 2x3 affine matrix used for alignment.
        """
        if detection.landmarks is None or len(detection.landmarks) < 2:
            raise AlignmentError("Invalid or missing landmarks for alignment.")
            
        left_eye = detection.landmarks[0]
        right_eye = detection.landmarks[1]
        
        # Calculate angle to level the eyes
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Center of rotation is midway between the eyes
        center = (
            (left_eye[0] + right_eye[0]) / 2.0,
            (left_eye[1] + right_eye[1]) / 2.0
        )
        
        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, scale=1.0)
        
        # Apply rotation
        h, w = image.shape[:2]
        aligned_image = cv2.warpAffine(
            image, M, (w, h), 
            flags=cv2.INTER_LINEAR, 
            borderMode=cv2.BORDER_REPLICATE
        )
        
        return aligned_image, M
