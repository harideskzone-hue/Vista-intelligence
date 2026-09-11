"""
Centralized types for the SIH26187 Face Module.
Includes both detection/pipeline types and database entity schemas.
"""
from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Literal

# --- PIPELINE TYPES ---

@dataclass(frozen=True)
class FaceDetection:
    """A single detected face from a video frame."""
    frame_index: int
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    # 5-point landmarks: [[left_eye_x, left_eye_y], [right_eye_x, ...], ...]
    landmarks: tuple[tuple[float, float], ...] | None


QualityStatus = Literal["PASS", "QUALITY_FAIL"]

@dataclass(frozen=True)
class QualityDecision:
    """Result of the face quality gate for one detection."""
    frame_index: int
    status: QualityStatus
    reason: str | None
    sharpness: float
    face_width_px: float
    face_height_px: float
    inter_ocular_distance: float | None


MatchStatus = Literal[
    "MATCH",
    "UNKNOWN",
    "AMBIGUOUS",
    "NO_WATCHLIST",
    "UNCALIBRATED",
]

@dataclass(frozen=True)
class FaceMatchResult:
    """Result of a 1:N watchlist search for one query embedding."""
    match_status: MatchStatus
    matched_identity_id: str | None
    similarity_score: float | None
    threshold_applied: float | None
    temporal_state: str = "UNKNOWN"


# --- DATABASE TYPES ---

PersonStatus = Literal["ACTIVE", "INACTIVE", "SUSPECT"]

@dataclass(frozen=True)
class Person:
    """An enrolled identity in the system."""
    person_id: str
    display_name: str
    category: str
    classification: str
    created_at: datetime.datetime
    status: PersonStatus


@dataclass(frozen=True)
class FaceEmbedding:
    """A single embedding vector for a person."""
    embedding_id: str
    person_id: str
    embedding_vector: bytes  # BLOB storage or array representation
    model_version: str
    created_at: datetime.datetime
    quality_score: float
    source: str


@dataclass(frozen=True)
class FaceImage:
    """A persisted face crop image metadata."""
    image_id: str
    person_id: str
    path: str
    camera_id: str
    frame_id: int
    timestamp: datetime.datetime
    sharpness: float
    pose_score: float | None
    lighting_score: float | None
    occlusion_score: float | None
    recognition_similarity: float
    quality_score: float
    hash_sha256: str
    source: str
    embedding_id: str | None = None
    embedding_id: str | None = None


@dataclass(frozen=True)
class CameraConfig:
    """Camera configuration within the identity system."""
    camera_id: str
    name: str
    stream_url: str
    enabled: bool
    status: str
    created_at: datetime.datetime


@dataclass(frozen=True)
class RecognitionEvent:
    """An event log of a face recognition decision."""
    event_id: str
    camera_id: str
    person_id: str | None  # None if UNKNOWN
    frame_id: int
    timestamp: datetime.datetime
    similarity: float | None
    quality_score: float
    decision: MatchStatus
    image_id: str | None
