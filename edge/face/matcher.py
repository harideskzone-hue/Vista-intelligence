"""
Matcher: Identity decision pipeline enforcing fail-closed invariants.
Logic: Quality -> Top-K Retrieval -> Threshold -> Margin -> Temporal Confirmation.
"""
from __future__ import annotations

import logging
import numpy as np

from edge.face.types import FaceMatchResult
from edge.face.watchlist import Watchlist, WatchlistIntegrityError
from edge.face.temporal_matcher import TemporalMatcher

log = logging.getLogger("matcher")

class FaceMatcher:
    def __init__(self, 
                 watchlist: Watchlist, 
                 temporal_matcher: TemporalMatcher,
                 operational_threshold: float = 0.55,  # DEMO_OPERATIONAL_THRESHOLD
                 margin_threshold: float = 0.1):       # DEMO_OPERATIONAL_MARGIN
        """
        Thresholds are operational/demo values and must be calibrated to the specific environment.
        """
        self.watchlist = watchlist
        self.temporal = temporal_matcher
        self.threshold = operational_threshold
        self.margin = margin_threshold

    def match(self, camera_id: str, track_id: str, embedding: np.ndarray, quality_pass: bool = True) -> FaceMatchResult:
        """
        Evaluates a single observation and determines the robust identity.
        Fails closed on any uncertainty.
        """
        # 1. Quality Gate Rejection
        if not quality_pass:
            state, _ = self.temporal.update(camera_id, track_id, None)
            return FaceMatchResult("UNKNOWN", None, None, self.threshold, state)

        # 2. Watchlist Empty Check
        if self.watchlist.index is None or self.watchlist.index.ntotal == 0:
            state, _ = self.temporal.update(camera_id, track_id, None)
            return FaceMatchResult("NO_WATCHLIST", None, None, self.threshold, state)

        # 3. Secure Retrieval (SQLite validated internally by Watchlist)
        try:
            candidates = self.watchlist.search(embedding, top_k=2)
        except WatchlistIntegrityError as e:
            # Watchlist DB integrity failed -> Fail closed
            log.error(f"Integrity failure during match: {e}")
            state, _ = self.temporal.update(camera_id, track_id, None)
            return FaceMatchResult("UNKNOWN", None, None, self.threshold, state)

        candidate_id = None
        best_sim = None

        if candidates:
            top1_person, _, top1_score = candidates[0]
            best_sim = float(top1_score)

            # 4. Absolute Similarity Threshold
            if top1_score >= self.threshold:
                # 5. Margin against Top-2
                if len(candidates) > 1:
                    top2_person, _, top2_score = candidates[1]
                    # Margin is only relevant to differentiate distinct identities
                    if top1_person != top2_person:
                        actual_margin = top1_score - top2_score
                        if actual_margin >= self.margin:
                            candidate_id = top1_person
                    else:
                        candidate_id = top1_person
                else:
                    candidate_id = top1_person

        # 6. Temporal Confirmation (Per-camera, Per-track)
        temporal_state, final_id = self.temporal.update(camera_id, track_id, candidate_id)

        # 7. Final Output Construction
        if final_id is None:
            status = "UNKNOWN" if candidate_id is None else "AMBIGUOUS"
            return FaceMatchResult(status, None, best_sim, self.threshold, temporal_state)

        return FaceMatchResult("MATCH", final_id, best_sim, self.threshold, temporal_state)
