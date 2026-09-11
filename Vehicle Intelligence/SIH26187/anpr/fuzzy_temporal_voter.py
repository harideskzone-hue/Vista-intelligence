from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass, field
import numpy as np

from core.schemas import PlateStatus, ANPRResult, OpticalQuality, PlateQualityState
from core.config import config
from anpr.plate_format import IndianFormatValidator

def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def normalized_levenshtein(s1: str, s2: str) -> float:
    if not s1 and not s2: return 1.0
    dist = levenshtein_distance(s1, s2)
    return 1.0 - (dist / max(len(s1), len(s2)))

@dataclass
class PlateObservation:
    text: str
    ocr_confidence: float
    detector_confidence: float
    quality_score: float
    frame_index: int
    optical_quality: OpticalQuality
    is_valid_format: bool
    
    @property
    def evidence(self) -> float:
        return float(self.ocr_confidence * self.detector_confidence * self.quality_score)

class TrackANPRState:
    """
    Maintains temporal ANPR state and observations for a single vehicle track.
    """
    def __init__(self, track_id: int):
        self.track_id = track_id
        self.observations: List[PlateObservation] = []
        self.plate_detection_attempts: int = 0
        self.plate_crops_found: int = 0
        self.insufficient_res_count: int = 0
        self.low_quality_count: int = 0
        self.best_optical_quality: Optional[OpticalQuality] = None
        
        self.first_confirmed_frame: Optional[int] = None
        self.stable_confirmed_frame: Optional[int] = None
        self.current_status: PlateStatus = PlateStatus.UNKNOWN
        self.current_plate: Optional[str] = None
        self.current_confidence: float = 0.0

    def add_attempt(self):
        self.plate_detection_attempts += 1

    def add_rejected_crop(self, state: PlateQualityState, metrics: OpticalQuality):
        self.plate_crops_found += 1
        if state == PlateQualityState.INSUFFICIENT_RESOLUTION:
            self.insufficient_res_count += 1
        elif state == PlateQualityState.LOW_QUALITY:
            self.low_quality_count += 1
        if self.best_optical_quality is None or metrics.area > self.best_optical_quality.area:
            self.best_optical_quality = metrics

    def add_observation(self, obs: PlateObservation):
        self.plate_crops_found += 1
        self.observations.append(obs)
        if self.best_optical_quality is None or obs.optical_quality.area > self.best_optical_quality.area:
            self.best_optical_quality = obs.optical_quality

class FuzzyTemporalVoter:
    """
    Fail-closed temporal voting engine.
    - Aggregates multi-frame plate evidence for each vehicle track.
    - Employs Levenshtein clustering and positional character voting.
    - Enforces strict criteria for CONFIRMED, failing closed to PROBABLE or UNKNOWN.
    """
    def __init__(
        self,
        min_observations: int = config.min_observations_for_confirmation,
        min_evidence: float = config.min_evidence_for_confirmation
    ):
        self.min_observations = min_observations
        self.min_evidence = min_evidence
        self.format_validator = IndianFormatValidator()

    def _cluster_observations(self, observations: List[PlateObservation]) -> List[List[PlateObservation]]:
        clusters = []
        for obs in observations:
            added = False
            for cluster in clusters:
                centroid = cluster[0]
                sim = normalized_levenshtein(obs.text, centroid.text)
                len_diff = abs(len(obs.text) - len(centroid.text))
                if sim >= 0.60 and len_diff <= 2:
                    cluster.append(obs)
                    cluster.sort(key=lambda o: o.evidence, reverse=True)
                    added = True
                    break
            if not added:
                clusters.append([obs])
        return sorted(clusters, key=lambda c: sum(o.evidence for o in c), reverse=True)

    def _align_and_vote(self, cluster: List[PlateObservation]) -> Tuple[str, float]:
        if not cluster:
            return "", 0.0
            
        # Determine dominant length weighted by evidence
        lengths = defaultdict(float)
        for obs in cluster:
            lengths[len(obs.text)] += obs.evidence
            
        target_len = max(lengths.keys(), key=lambda k: lengths[k])
        aligned = [o for o in cluster if len(o.text) == target_len]
        if not aligned:
            return cluster[0].text, sum(o.evidence for o in cluster)
            
        total_evidence = sum(o.evidence for o in aligned)
        final_chars = []
        for i in range(target_len):
            char_votes = defaultdict(float)
            for obs in aligned:
                char_votes[obs.text[i]] += obs.evidence
            best_char = max(char_votes.keys(), key=lambda k: char_votes[k])
            final_chars.append(best_char)
            
        return "".join(final_chars), total_evidence

    def evaluate(self, track_state: TrackANPRState, current_frame: int) -> ANPRResult:
        """
        Evaluates the track's observations and returns an ANPRResult according to fail-closed state logic.
        """
        # 1. No plate ever detected by detector
        if track_state.plate_crops_found == 0:
            return ANPRResult(
                plate=None,
                status=PlateStatus.NO_PLATE_DETECTED,
                confidence=0.0,
                format_valid=False,
                optical_quality=track_state.best_optical_quality,
                observations=0,
                temporal_support=0
            )

        # 2. Crops were detected, but all were rejected by optical quality gate
        if not track_state.observations:
            if track_state.insufficient_res_count >= track_state.low_quality_count:
                status = PlateStatus.INSUFFICIENT_RESOLUTION
            else:
                status = PlateStatus.LOW_QUALITY
                
            return ANPRResult(
                plate=None,
                status=status,
                confidence=0.0,
                format_valid=False,
                optical_quality=track_state.best_optical_quality,
                observations=track_state.plate_crops_found,
                temporal_support=0
            )

        # 3. LPR observations exist -> run fuzzy temporal clustering
        clusters = self._cluster_observations(track_state.observations)
        dominant_cluster = clusters[0]
        
        # Check for ambiguity: if competing cluster has high relative evidence
        if len(clusters) > 1:
            dom_ev = sum(o.evidence for o in dominant_cluster)
            sec_ev = sum(o.evidence for o in clusters[1])
            if dom_ev > 0 and (sec_ev / dom_ev) >= 0.75:
                # Ambiguous reads -> fail-closed
                return ANPRResult(
                    plate=None,
                    status=PlateStatus.OCR_UNCERTAIN,
                    confidence=0.0,
                    format_valid=False,
                    optical_quality=track_state.best_optical_quality,
                    observations=len(track_state.observations),
                    temporal_support=len(dominant_cluster)
                )

        reconstructed_text, cluster_evidence = self._align_and_vote(dominant_cluster)
        avg_evidence = cluster_evidence / len(dominant_cluster) if dominant_cluster else 0.0
        obs_count = len(dominant_cluster)
        
        is_valid = self.format_validator.is_valid(reconstructed_text)
        
        # Determine status
        if obs_count >= self.min_observations and avg_evidence >= self.min_evidence and is_valid:
            status = PlateStatus.CONFIRMED
            if track_state.first_confirmed_frame is None:
                track_state.first_confirmed_frame = current_frame
            if obs_count >= (self.min_observations + 2) and track_state.stable_confirmed_frame is None:
                track_state.stable_confirmed_frame = current_frame
        elif obs_count >= 1 and (is_valid or avg_evidence >= 0.05):
            status = PlateStatus.PROBABLE
        else:
            status = PlateStatus.UNKNOWN

        final_plate = reconstructed_text if status in (PlateStatus.CONFIRMED, PlateStatus.PROBABLE) else None
        conf = float(round(avg_evidence, 2)) if final_plate else 0.0

        track_state.current_status = status
        track_state.current_plate = final_plate
        track_state.current_confidence = conf

        return ANPRResult(
            plate=final_plate,
            status=status,
            confidence=conf,
            format_valid=is_valid,
            optical_quality=track_state.best_optical_quality,
            observations=len(track_state.observations),
            temporal_support=obs_count,
            first_confirmed_frame=track_state.first_confirmed_frame,
            stable_confirmed_frame=track_state.stable_confirmed_frame
        )
