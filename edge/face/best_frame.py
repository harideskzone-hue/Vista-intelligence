"""
Best Frame: Evaluates frame quality to retain the most useful face images.
"""
from __future__ import annotations

import logging
from edge.face.types import QualityDecision, FaceDetection

log = logging.getLogger("best_frame")

class BestFrameSelector:
    """
    Scores candidate frames using measurable signals (sharpness, size, confidence).
    Retains the best useful frame per person_id.
    """
    
    def __init__(self):
        # Maps person_id -> best_score
        self.best_scores: dict[str, float] = {}
        
    @staticmethod
    def calculate_score(detection: FaceDetection, quality: QualityDecision) -> float:
        """
        Calculates a unified usefulness score. Higher is better.
        Combines face size, sharpness, and detection confidence.
        """
        # 1. Face Size
        face_w = detection.x2 - detection.x1
        face_h = detection.y2 - detection.y1
        size_area = face_w * face_h
        
        # Normalize size relative to an expected good size (e.g., 200x200 = 40000)
        norm_size = min(1.0, size_area / 40000.0)
        
        # 2. Sharpness
        # Assuming sharpness is typically 0-100, where higher is sharper
        norm_sharpness = min(1.0, quality.sharpness / 100.0)
        
        # 3. Detection Confidence
        conf = detection.confidence
        
        # Optional: Penalize if not frontal or if occluded (placeholders for future extension)
        # if quality.pose_score is not None:
        #     score += quality.pose_score * 0.1
        
        # Weighted score
        score = (norm_size * 0.4) + (norm_sharpness * 0.4) + (conf * 0.2)
        return score

    def evaluate_and_update(self, person_id: str, detection: FaceDetection, quality: QualityDecision) -> bool:
        """
        Evaluates if the current frame is the best seen for the person.
        Returns True if it's a new best frame (or first frame).
        If True, the internal best score is updated.
        """
        # Unknown identities shouldn't be tracked for best frame
        if person_id is None or person_id == "UNKNOWN":
            return False
            
        score = self.calculate_score(detection, quality)
        
        if person_id not in self.best_scores:
            self.best_scores[person_id] = score
            return True
            
        current_best = self.best_scores[person_id]
        if score > current_best:
            self.best_scores[person_id] = score
            return True
            
        return False
        
    def clear(self):
        """Reset best frame tracking."""
        self.best_scores.clear()
