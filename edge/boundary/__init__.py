"""
Boundary Intelligence Subsystem — Virtual Line Crossing & Anomaly Detection.
"""

from edge.boundary.geometry import (
    Point,
    Line,
    NormalizedLine,
    side_of_point,
    bottom_center,
    centroid,
    distance,
    bbox_aspect_ratio,
    point_in_polygon,
    angle_between_vectors
)
from edge.boundary.crossing_engine import (
    TrackObservation,
    CrossingEvent,
    CrossingEngine
)
from edge.boundary.anomaly_engine import (
    AnomalyEvent,
    AnomalyEngine,
    AnomalyThresholds,
    TrackMovementHistory
)
from edge.boundary.manager import (
    BoundaryManager,
    CameraBoundaryConfig,
    get_boundary_manager
)
from edge.boundary.adapter import (
    BoundaryTrackingAdapter,
    get_tracking_adapter
)

__all__ = [
    'Point',
    'Line',
    'NormalizedLine',
    'side_of_point',
    'bottom_center',
    'centroid',
    'distance',
    'bbox_aspect_ratio',
    'point_in_polygon',
    'angle_between_vectors',
    'TrackObservation',
    'CrossingEvent',
    'CrossingEngine',
    'AnomalyEvent',
    'AnomalyEngine',
    'AnomalyThresholds',
    'TrackMovementHistory',
    'BoundaryManager',
    'CameraBoundaryConfig',
    'get_boundary_manager',
    'BoundaryTrackingAdapter',
    'get_tracking_adapter',
]
