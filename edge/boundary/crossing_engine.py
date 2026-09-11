"""
CrossingEngine — Virtual Line Crossing & Restricted Zone State Machine.

Evaluates tracked object positions against configured boundary lines and zones:
- Computes ground-plane bottom-center crossings.
- Classifies directionality: INTRUDING vs RETREATING based on configured restricted_side.
- Enforces per-track debounce cooldowns to eliminate jitter/duplicate triggers.
- Automatically cleans up stale tracks.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import time
import numpy as np

from edge.boundary.geometry import Point, Line, side_of_point, bottom_center, point_in_polygon, point_to_segment_distance


@dataclass
class TrackObservation:
    track_id: int
    bbox: Tuple[float, float, float, float]  # (xmin, ymin, xmax, ymax)
    class_name: str = "person"
    confidence: float = 1.0
    timestamp: float = 0.0


@dataclass
class CrossingEvent:
    track_id: int
    class_name: str
    boundary_id: str
    from_side: int           # +1 (Side A) or -1 (Side B)
    to_side: int             # +1 or -1
    direction: str           # "INTRUDING", "RETREATING", "ZONE_INTRUDED", "ZONE_EXITED"
    timestamp: float
    location: Tuple[float, float]
    confidence: float = 1.0


class CrossingEngine:
    """
    Stateful boundary crossing detector operating on tracked object observations.
    """
    def __init__(self, cooldown_seconds: float = 2.0, stale_timeout: float = 10.0):
        self.cooldown_seconds = float(cooldown_seconds)
        self.stale_timeout = float(stale_timeout)

        # boundary_id -> track_id -> side (+1 or -1)
        self._prev_line_sides: Dict[str, Dict[int, int]] = {}
        # zone_id -> track_id -> is_inside (bool)
        self._prev_zone_states: Dict[str, Dict[int, bool]] = {}

        # (boundary_id, track_id) -> last_event_timestamp
        self._last_event_time: Dict[Tuple[str, int], float] = {}

        # (boundary_id, track_id) -> is_currently_touching
        self._in_touch_zone: Dict[Tuple[str, int], bool] = {}

        # track_id -> last_seen_timestamp
        self._last_seen: Dict[int, float] = {}

        # Metrics
        self.total_crossings = 0
        self.total_intrusions = 0
        self.total_retreats = 0
        self.total_zone_entries = 0
        self.total_zone_exits = 0

    def _is_on_cooldown(self, boundary_id: str, track_id: int, current_time: float) -> bool:
        last = self._last_event_time.get((boundary_id, track_id))
        if last is None:
            return False
        return (current_time - last) < self.cooldown_seconds

    def check_line_crossings(
        self,
        boundary_id: str,
        line: Line,
        restricted_side: int,
        observations: List[TrackObservation],
        timestamp: Optional[float] = None,
        touch_tolerance_px: float = 0.0
    ) -> List[CrossingEvent]:
        """
        Evaluates whether any tracked objects crossed or touched the line since the last observation.
        Uses bottom-center of the bbox as the ground contact anchor.
        """
        now = float(timestamp if timestamp is not None else time.time())
        events: List[CrossingEvent] = []

        if boundary_id not in self._prev_line_sides:
            self._prev_line_sides[boundary_id] = {}
        side_map = self._prev_line_sides[boundary_id]

        for obs in observations:
            tid = obs.track_id
            if tid < 0:
                continue

            self._last_seen[tid] = now
            anchor = bottom_center(obs.bbox)
            current_side = side_of_point(line, anchor)

            # Check border touch condition with deduplication
            if touch_tolerance_px > 0:
                dist = point_to_segment_distance(anchor, line)
                was_touching = self._in_touch_zone.get((boundary_id, tid), False)
                if dist <= touch_tolerance_px:
                    if not was_touching and not self._is_on_cooldown(boundary_id, tid, now):
                        self._in_touch_zone[(boundary_id, tid)] = True
                        self._last_event_time[(boundary_id, tid)] = now
                        touch_ev = CrossingEvent(
                            track_id=tid,
                            class_name=obs.class_name,
                            boundary_id=boundary_id,
                            from_side=current_side if current_side != 0 else restricted_side,
                            to_side=current_side if current_side != 0 else restricted_side,
                            direction="BORDER_TOUCH",
                            timestamp=now,
                            location=(anchor.x, anchor.y),
                            confidence=obs.confidence
                        )
                        events.append(touch_ev)
                elif dist > touch_tolerance_px * 1.5:
                    # Reset touch state once person moves comfortably away from boundary
                    self._in_touch_zone[(boundary_id, tid)] = False

            # Exactly on the line or indeterminable for side transitions
            if current_side == 0:
                continue

            prev_side = side_map.get(tid)
            side_map[tid] = current_side

            # First observation of this track — establish baseline without false crossing
            if prev_side is None:
                continue

            # Same side — no crossing
            if current_side == prev_side:
                continue

            # Check debounce cooldown
            if self._is_on_cooldown(boundary_id, tid, now):
                continue

            # Direction determination based on calibrated restricted_side
            if current_side == restricted_side:
                direction = "INTRUDING"
                self.total_intrusions += 1
            else:
                direction = "RETREATING"
                self.total_retreats += 1

            self.total_crossings += 1
            self._last_event_time[(boundary_id, tid)] = now

            ev = CrossingEvent(
                track_id=tid,
                class_name=obs.class_name,
                boundary_id=boundary_id,
                from_side=prev_side,
                to_side=current_side,
                direction=direction,
                timestamp=now,
                location=(anchor.x, anchor.y),
                confidence=obs.confidence
            )
            events.append(ev)

        self.cleanup_stale_tracks(now)
        return events

    def check_zone_transitions(
        self,
        zone_id: str,
        polygon_points: np.ndarray,
        observations: List[TrackObservation],
        timestamp: Optional[float] = None
    ) -> List[CrossingEvent]:
        """
        Evaluates whether any tracked objects entered or exited a polygon restricted zone.
        """
        now = float(timestamp if timestamp is not None else time.time())
        events: List[CrossingEvent] = []

        if zone_id not in self._prev_zone_states:
            self._prev_zone_states[zone_id] = {}
        zone_map = self._prev_zone_states[zone_id]

        for obs in observations:
            tid = obs.track_id
            if tid < 0:
                continue

            self._last_seen[tid] = now
            anchor = bottom_center(obs.bbox)
            inside = point_in_polygon(anchor, polygon_points)

            prev_inside = zone_map.get(tid)
            zone_map[tid] = inside

            if prev_inside is None:
                continue

            if inside == prev_inside:
                continue

            if self._is_on_cooldown(zone_id, tid, now):
                continue

            if inside:
                direction = "ZONE_INTRUDED"
                self.total_zone_entries += 1
            else:
                direction = "ZONE_EXITED"
                self.total_zone_exits += 1

            self.total_crossings += 1
            self._last_event_time[(zone_id, tid)] = now

            ev = CrossingEvent(
                track_id=tid,
                class_name=obs.class_name,
                boundary_id=zone_id,
                from_side=0 if prev_inside else 1,
                to_side=1 if inside else 0,
                direction=direction,
                timestamp=now,
                location=(anchor.x, anchor.y),
                confidence=obs.confidence
            )
            events.append(ev)

        self.cleanup_stale_tracks(now)
        return events

    def cleanup_stale_tracks(self, current_time: float) -> None:
        """Prunes tracking history for tracks that disappeared beyond stale_timeout."""
        stale_ids = [
            tid for tid, last_t in self._last_seen.items()
            if (current_time - last_t) > self.stale_timeout
        ]
        for tid in stale_ids:
            self._last_seen.pop(tid, None)
            for side_map in self._prev_line_sides.values():
                side_map.pop(tid, None)
            for zone_map in self._prev_zone_states.values():
                zone_map.pop(tid, None)
            # Prune cooldowns for this tid
            keys_to_del = [k for k in self._last_event_time if k[1] == tid]
            for k in keys_to_del:
                self._last_event_time.pop(k, None)

    def reset(self) -> None:
        """Resets all internal crossing engine state."""
        self._prev_line_sides.clear()
        self._prev_zone_states.clear()
        self._last_event_time.clear()
        self._last_seen.clear()
        self.total_crossings = 0
        self.total_intrusions = 0
        self.total_retreats = 0
        self.total_zone_entries = 0
        self.total_zone_exits = 0
