"""
AnomalyEngine — Rule-Based Behavioral Analytics & Threat Detector.

Evaluates tracked person trajectories for suspicious behaviors:
1. Loitering: Remaining within a compact radius for excessive duration.
2. Crawling: Prone body aspect ratio (w/h > 0.85) indicating low-crawling.
3. Running / Speed Anomaly: High-velocity movement toward or near boundaries.
4. Erratic Movement: Rapid zigzag directional changes attempting evasion.
5. Grouping: Dense clustering of 3+ persons near a checkpoint or boundary.
6. Zone Lingering: Remaining inside a restricted zone beyond a grace duration.

Note: All alerts produced are explicit rule-based anomaly indicators.
"""

from dataclasses import dataclass, field, asdict
from collections import deque
from typing import Dict, List, Tuple, Optional, Set, Any
import math
import time
import numpy as np

from edge.boundary.geometry import Point, distance, bbox_aspect_ratio, point_in_polygon, angle_between_vectors
from edge.boundary.crossing_engine import TrackObservation


@dataclass
class AnomalyThresholds:
    """Configurable thresholds for rule-based anomaly indicators."""
    loitering_time: float = 5.0
    loitering_radius: float = 60.0
    zone_lingering_time: float = 6.0
    running_speed_thresh: float = 180.0
    crawling_ratio_thresh: float = 0.85
    crawling_min_frames: int = 5
    erratic_angle_thresh: float = 80.0
    erratic_min_turns: int = 3
    group_min_size: int = 3
    group_distance_px: float = 80.0
    alert_cooldown_sec: float = 10.0
    stale_timeout: float = 12.0
    indicator_label: str = "Rule-based anomaly indicator"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnomalyThresholds':
        valid_keys = cls.__dataclass_fields__.keys()
        filtered = {k: data[k] for k in data if k in valid_keys}
        return cls(**filtered)


@dataclass
class AnomalyEvent:
    event_type: str          # "LOITERING", "CRAWLING", "RUNNING", "ERRATIC_MOVEMENT", "GROUPING", "ZONE_LINGERING"
    severity: str            # "INFO", "WARNING", "CRITICAL"
    track_id: int
    class_name: str
    location: Tuple[float, float]
    timestamp: float
    details: str
    duration: float = 0.0
    category: str = "Rule-based anomaly indicator"


@dataclass
class TrackMovementHistory:
    track_id: int
    class_name: str
    positions: deque = field(default_factory=lambda: deque(maxlen=300))  # (x, y, timestamp)
    bboxes: deque = field(default_factory=lambda: deque(maxlen=60))      # (bbox, timestamp)
    last_alert_times: Dict[str, float] = field(default_factory=dict)     # event_type -> timestamp
    zone_entry_times: Dict[str, float] = field(default_factory=dict)     # zone_id -> timestamp

    def add_point(self, x: float, y: float, bbox: Tuple[float, float, float, float], timestamp: float) -> None:
        self.positions.append((x, y, timestamp))
        self.bboxes.append((bbox, timestamp))

    def net_displacement(self, duration_sec: float) -> float:
        """Straight-line distance from position duration_sec ago to now."""
        if len(self.positions) < 2:
            return 0.0
        now = self.positions[-1][2]
        cutoff = now - duration_sec
        start = None
        for x, y, t in self.positions:
            if t >= cutoff:
                start = (x, y)
                break
        if start is None:
            return 0.0
        end = (self.positions[-1][0], self.positions[-1][1])
        return math.hypot(end[0] - start[0], end[1] - start[1])

    def total_path_length(self, duration_sec: float) -> float:
        """Accumulated path length traversed over the last duration_sec."""
        if len(self.positions) < 2:
            return 0.0
        now = self.positions[-1][2]
        cutoff = now - duration_sec
        total = 0.0
        prev = None
        for x, y, t in self.positions:
            if t < cutoff:
                continue
            if prev is not None:
                total += math.hypot(x - prev[0], y - prev[1])
            prev = (x, y)
        return total

    def speed_pixels_per_second(self, duration_sec: float = 1.0) -> float:
        """Average speed over the last duration_sec."""
        if len(self.positions) < 2:
            return 0.0
        now = self.positions[-1][2]
        cutoff = now - duration_sec
        points = [(x, y, t) for x, y, t in self.positions if t >= cutoff]
        if len(points) < 2:
            return 0.0
        dt = points[-1][2] - points[0][2]
        if dt <= 0.05:
            return 0.0
        dist = sum(math.hypot(points[i][0] - points[i-1][0], points[i][1] - points[i-1][1]) for i in range(1, len(points)))
        return dist / dt


class AnomalyEngine:
    """
    Configurable behavioral anomaly analyzer for border security operations.
    All outputs represent rule-based anomaly indicators.
    """
    def __init__(
        self,
        thresholds: Optional[AnomalyThresholds] = None,
        **kwargs
    ):
        if thresholds is not None:
            self.thresholds = thresholds
        else:
            self.thresholds = AnomalyThresholds()

        # Allow kwargs override
        for k, v in kwargs.items():
            if hasattr(self.thresholds, k):
                setattr(self.thresholds, k, type(getattr(self.thresholds, k))(v))

        # track_id -> TrackMovementHistory
        self._tracks: Dict[int, TrackMovementHistory] = {}
        self._last_seen: Dict[int, float] = {}

    def update_thresholds(self, new_thresholds: Dict[str, Any]) -> AnomalyThresholds:
        """Dynamically updates behavioral thresholds."""
        for k, v in new_thresholds.items():
            if hasattr(self.thresholds, k):
                setattr(self.thresholds, k, type(getattr(self.thresholds, k))(v))
        return self.thresholds

    def _should_alert(self, history: TrackMovementHistory, event_type: str, now: float) -> bool:
        if event_type not in history.last_alert_times:
            return True
        last = history.last_alert_times[event_type]
        return (now - last) >= self.thresholds.alert_cooldown_sec

    def update(
        self,
        observations: List[TrackObservation],
        timestamp: Optional[float] = None,
        zones: Optional[Dict[str, np.ndarray]] = None
    ) -> List[AnomalyEvent]:
        """
        Ingests current frame observations and returns any newly detected anomalies.
        """
        now = float(timestamp if timestamp is not None else time.time())
        anomalies: List[AnomalyEvent] = []

        # 1. Update movement histories
        for obs in observations:
            tid = obs.track_id
            if tid < 0:
                continue

            self._last_seen[tid] = now
            if tid not in self._tracks:
                self._tracks[tid] = TrackMovementHistory(track_id=tid, class_name=obs.class_name)

            th = self._tracks[tid]
            cx = (obs.bbox[0] + obs.bbox[2]) / 2.0
            cy = (obs.bbox[1] + obs.bbox[3]) / 2.0
            th.add_point(cx, cy, obs.bbox, now)

            # Check individual track anomalies
            # A. Crawling Detection (aspect ratio w/h > threshold over consecutive frames)
            crawl_event = self._check_crawling(th, now)
            if crawl_event:
                anomalies.append(crawl_event)

            # B. Running / Speed Anomaly
            run_event = self._check_running(th, now)
            if run_event:
                anomalies.append(run_event)

            # C. Loitering
            loiter_event = self._check_loitering(th, now)
            if loiter_event:
                anomalies.append(loiter_event)

            # D. Erratic / Zigzag Movement
            erratic_event = self._check_erratic(th, now)
            if erratic_event:
                anomalies.append(erratic_event)

            # E. Zone Lingering (if zones provided)
            if zones:
                zone_events = self._check_zone_lingering(th, zones, now)
                anomalies.extend(zone_events)

        # 2. Group Clustering Anomaly
        group_events = self._check_grouping(observations, now)
        anomalies.extend(group_events)

        # 3. Clean up stale tracks
        self._cleanup_stale(now)
        return anomalies

    def _check_crawling(self, th: TrackMovementHistory, now: float) -> Optional[AnomalyEvent]:
        min_frames = self.thresholds.crawling_min_frames
        if len(th.bboxes) < min_frames:
            return None

        recent_boxes = list(th.bboxes)[-min_frames:]
        high_ar_count = sum(1 for b, _ in recent_boxes if bbox_aspect_ratio(b) >= self.thresholds.crawling_ratio_thresh)

        if high_ar_count >= (min_frames - 1):
            if self._should_alert(th, "CRAWLING", now):
                th.last_alert_times["CRAWLING"] = now
                pos = th.positions[-1]
                return AnomalyEvent(
                    event_type="CRAWLING",
                    severity="CRITICAL",
                    track_id=th.track_id,
                    class_name=th.class_name,
                    location=(pos[0], pos[1]),
                    timestamp=now,
                    details=f"Track #{th.track_id} exhibits prone body aspect ratio (Rule-based anomaly indicator)",
                    category=self.thresholds.indicator_label
                )
        return None

    def _check_running(self, th: TrackMovementHistory, now: float) -> Optional[AnomalyEvent]:
        speed = th.speed_pixels_per_second(duration_sec=1.0)
        if speed >= self.thresholds.running_speed_thresh:
            if self._should_alert(th, "RUNNING", now):
                th.last_alert_times["RUNNING"] = now
                pos = th.positions[-1]
                return AnomalyEvent(
                    event_type="RUNNING",
                    severity="WARNING",
                    track_id=th.track_id,
                    class_name=th.class_name,
                    location=(pos[0], pos[1]),
                    timestamp=now,
                    details=f"Track #{th.track_id} moving at high velocity ({speed:.1f} px/s) (Rule-based anomaly indicator)",
                    category=self.thresholds.indicator_label
                )
        return None

    def _check_loitering(self, th: TrackMovementHistory, now: float) -> Optional[AnomalyEvent]:
        if len(th.positions) < 2:
            return None
        dt = now - th.positions[0][2]
        if dt < self.thresholds.loitering_time:
            return None

        disp = th.net_displacement(self.thresholds.loitering_time)
        if disp <= self.thresholds.loitering_radius:
            if self._should_alert(th, "LOITERING", now):
                th.last_alert_times["LOITERING"] = now
                pos = th.positions[-1]
                return AnomalyEvent(
                    event_type="LOITERING",
                    severity="WARNING",
                    track_id=th.track_id,
                    class_name=th.class_name,
                    location=(pos[0], pos[1]),
                    timestamp=now,
                    duration=round(dt, 1),
                    details=f"Track #{th.track_id} loitering within {disp:.1f}px radius for {dt:.1f}s (Rule-based anomaly indicator)",
                    category=self.thresholds.indicator_label
                )
        return None

    def _check_erratic(self, th: TrackMovementHistory, now: float) -> Optional[AnomalyEvent]:
        cutoff = now - 4.0
        pts = [(x, y) for x, y, t in th.positions if t >= cutoff]
        if len(pts) < 6:
            return None

        step = max(1, len(pts) // 8)
        sampled = pts[::step]
        if len(sampled) < 4:
            return None

        vectors = [
            np.array([sampled[i][0] - sampled[i-1][0], sampled[i][1] - sampled[i-1][1]], dtype=np.float32)
            for i in range(1, len(sampled))
        ]

        sharp_turns = 0
        for i in range(1, len(vectors)):
            ang = angle_between_vectors(vectors[i-1], vectors[i])
            if ang >= self.thresholds.erratic_angle_thresh:
                sharp_turns += 1

        if sharp_turns >= self.thresholds.erratic_min_turns:
            if self._should_alert(th, "ERRATIC_MOVEMENT", now):
                th.last_alert_times["ERRATIC_MOVEMENT"] = now
                pos = th.positions[-1]
                return AnomalyEvent(
                    event_type="ERRATIC_MOVEMENT",
                    severity="WARNING",
                    track_id=th.track_id,
                    class_name=th.class_name,
                    location=(pos[0], pos[1]),
                    timestamp=now,
                    details=f"Track #{th.track_id} executed {sharp_turns} abrupt zigzag direction changes (Rule-based anomaly indicator)",
                    category=self.thresholds.indicator_label
                )
        return None

    def _check_zone_lingering(
        self,
        th: TrackMovementHistory,
        zones: Dict[str, np.ndarray],
        now: float
    ) -> List[AnomalyEvent]:
        events = []
        if not th.positions:
            return events

        latest_pos = Point(th.positions[-1][0], th.positions[-1][1])
        for zid, poly in zones.items():
            inside = point_in_polygon(latest_pos, poly)
            if inside:
                if zid not in th.zone_entry_times:
                    th.zone_entry_times[zid] = now
                duration = now - th.zone_entry_times[zid]
                if duration >= self.thresholds.zone_lingering_time and self._should_alert(th, f"LINGER_{zid}", now):
                    th.last_alert_times[f"LINGER_{zid}"] = now
                    events.append(AnomalyEvent(
                        event_type="ZONE_LINGERING",
                        severity="CRITICAL",
                        track_id=th.track_id,
                        class_name=th.class_name,
                        location=(latest_pos.x, latest_pos.y),
                        timestamp=now,
                        duration=round(duration, 1),
                        details=f"Track #{th.track_id} lingering in restricted zone {zid} for {duration:.1f}s (Rule-based anomaly indicator)",
                        category=self.thresholds.indicator_label
                    ))
            else:
                th.zone_entry_times.pop(zid, None)

        return events

    def _check_grouping(
        self,
        observations: List[TrackObservation],
        now: float
    ) -> List[AnomalyEvent]:
        if len(observations) < self.thresholds.group_min_size:
            return []

        clusters: List[List[TrackObservation]] = []
        for obs in observations:
            placed = False
            c1 = ((obs.bbox[0] + obs.bbox[2]) / 2.0, (obs.bbox[1] + obs.bbox[3]) / 2.0)
            for cluster in clusters:
                c2 = ((cluster[0].bbox[0] + cluster[0].bbox[2]) / 2.0, (cluster[0].bbox[1] + cluster[0].bbox[3]) / 2.0)
                if math.hypot(c1[0] - c2[0], c1[1] - c2[1]) <= self.thresholds.group_distance_px:
                    cluster.append(obs)
                    placed = True
                    break
            if not placed:
                clusters.append([obs])

        events = []
        for cluster in clusters:
            if len(cluster) >= self.thresholds.group_min_size:
                lead_track = cluster[0]
                th = self._tracks.get(lead_track.track_id)
                if th and self._should_alert(th, "GROUPING", now):
                    th.last_alert_times["GROUPING"] = now
                    c_x = sum((o.bbox[0] + o.bbox[2]) / 2.0 for o in cluster) / len(cluster)
                    c_y = sum((o.bbox[1] + o.bbox[3]) / 2.0 for o in cluster) / len(cluster)
                    tids = [o.track_id for o in cluster]
                    events.append(AnomalyEvent(
                        event_type="GROUPING",
                        severity="WARNING",
                        track_id=lead_track.track_id,
                        class_name="group",
                        location=(c_x, c_y),
                        timestamp=now,
                        details=f"Group formation of {len(cluster)} persons clustered: tracks {tids} (Rule-based anomaly indicator)",
                        category=self.thresholds.indicator_label
                    ))
        return events

    def _cleanup_stale(self, now: float) -> None:
        stale_ids = [tid for tid, last_t in self._last_seen.items() if (now - last_t) > self.thresholds.stale_timeout]
        for tid in stale_ids:
            self._last_seen.pop(tid, None)
            self._tracks.pop(tid, None)

    def reset(self) -> None:
        self._tracks.clear()
        self._last_seen.clear()
