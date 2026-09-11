from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


@dataclass
class PlateObservation:
    text: str
    ocr_confidence: float
    detector_confidence: float
    frame_index: int


@dataclass
class PlateDecision:
    text: str | None
    confidence: float
    status: str
    observations: int


class TemporalPlateVoter:

    def __init__(
        self,
        min_observations: int = 3,
        min_confidence: float = 0.50,
    ) -> None:

        self.min_observations = min_observations
        self.min_confidence = min_confidence

        self.observations: list[PlateObservation] = []

    def add(
        self,
        text: str,
        ocr_confidence: float,
        detector_confidence: float,
        frame_index: int,
    ) -> None:

        text = text.strip().upper()

        if not text:
            return

        self.observations.append(
            PlateObservation(
                text=text,
                ocr_confidence=ocr_confidence,
                detector_confidence=detector_confidence,
                frame_index=frame_index,
            )
        )

    def decide(self) -> PlateDecision:

        if not self.observations:
            return PlateDecision(
                text=None,
                confidence=0.0,
                status="UNKNOWN",
                observations=0,
            )

        scores = defaultdict(float)
        counts = defaultdict(int)

        for observation in self.observations:

            # Weighted evidence.
            score = (
                observation.ocr_confidence
                * observation.detector_confidence
            )

            scores[observation.text] += score
            counts[observation.text] += 1

        best_text = max(
            scores,
            key=scores.get,
        )

        best_score = scores[best_text]
        best_count = counts[best_text]

        average_score = (
            best_score / best_count
        )

        if (
            best_count >= self.min_observations
            and average_score >= self.min_confidence
        ):
            status = "CONFIRMED"

        elif best_count >= 2:
            status = "PROBABLE"

        else:
            status = "UNKNOWN"

        return PlateDecision(
            text=best_text,
            confidence=average_score,
            status=status,
            observations=best_count,
        )

    def reset(self) -> None:
        self.observations.clear()
