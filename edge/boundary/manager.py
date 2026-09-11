"""
BoundaryManager — Central Boundary, Line Crossing & Anomaly Intelligence Orchestrator.

Integrates:
- Normalized [0.0, 1.0] Canvas Line & Polygon Zone configurations per camera.
- Per-camera CrossingEngine and AnomalyEngine isolation.
- Full multi-object tracking integration (YOLO + ByteTrack adapter) for real uploaded videos & live streams.
- Strict semantic metric separation: Line Crossings (Total, Intruding, Retreating), Zone Events (Entered, Exited),
  and Rule-Based Anomaly Indicators (Total, By Type).
- Integration with EventHub (logging events as PENDING) and EventClipManager (asynchronous clip generation).
- Optional identity enrichment from Face AI without hard dependency.
"""

import os
import json
import time
import logging
import threading
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional, Any, Callable
import cv2
import numpy as np

from edge.boundary.geometry import Point, Line, NormalizedLine
from edge.boundary.crossing_engine import TrackObservation, CrossingEvent, CrossingEngine
from edge.boundary.anomaly_engine import AnomalyEvent, AnomalyEngine, AnomalyThresholds
from edge.events.event_hub import EventHub, SystemEvent
from edge.events.clip_manager import EventClipManager

log = logging.getLogger(__name__)


@dataclass
class CameraBoundaryConfig:
    camera_id: str
    line: Optional[Dict[str, float]] = None  # {x1, y1, x2, y2} normalized
    restricted_side: str = "A"              # "A" (+1) or "B" (-1)
    zones: Dict[str, List[List[float]]] = field(default_factory=dict)  # zone_id -> [[x, y], ...] normalized
    line_id: str = "main"
    anomaly_thresholds: Optional[Dict[str, Any]] = None
    active: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "line": self.line,
            "restricted_side": self.restricted_side,
            "zones": self.zones,
            "line_id": self.line_id,
            "anomaly_thresholds": self.anomaly_thresholds,
            "active": self.active
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CameraBoundaryConfig':
        return cls(
            camera_id=data.get("camera_id", "default"),
            line=data.get("line"),
            restricted_side=data.get("restricted_side", "A"),
            zones=data.get("zones", {}),
            line_id=data.get("line_id", "main"),
            anomaly_thresholds=data.get("anomaly_thresholds"),
            active=data.get("active", True)
        )


class BoundaryManager:
    """
    Per-camera Boundary and Anomaly Intelligence Coordinator.
    """
    def __init__(
        self,
        config_path: str = "data/boundary/config.json",
        event_hub: Optional[EventHub] = None,
        clip_manager: Optional[EventClipManager] = None
    ):
        self.config_path = config_path
        self.event_hub = event_hub
        self.clip_manager = clip_manager

        self._configs: Dict[str, CameraBoundaryConfig] = {}
        self._crossing_engines: Dict[str, CrossingEngine] = {}
        self._anomaly_engines: Dict[str, AnomalyEngine] = {}
        self._default_thresholds = AnomalyThresholds()
        self._lock = threading.Lock()

        # Ensure directory exists and load configs
        os.makedirs(os.path.dirname(self.config_path) or ".", exist_ok=True)
        self._load_configs()

    def _load_configs(self) -> None:
        if not os.path.exists(self.config_path):
            return
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
                for cam_id, cfg_data in raw.items():
                    self._configs[cam_id] = CameraBoundaryConfig.from_dict(cfg_data)
            log.info(f"Loaded boundary configurations for {len(self._configs)} cameras")
        except Exception as e:
            log.error(f"Failed to load boundary config from {self.config_path}: {e}")

    def _save_configs(self) -> None:
        try:
            temp_path = self.config_path + ".tmp"
            with open(temp_path, "w", encoding="utf-8") as f:
                dump_data = {cam_id: cfg.to_dict() for cam_id, cfg in self._configs.items()}
                json.dump(dump_data, f, indent=2)
            os.replace(temp_path, self.config_path)
        except Exception as e:
            log.error(f"Failed to save boundary config to {self.config_path}: {e}")

    def get_config(self, camera_id: str) -> Dict[str, Any]:
        with self._lock:
            if camera_id in self._configs:
                return self._configs[camera_id].to_dict()
            return CameraBoundaryConfig(camera_id=camera_id).to_dict()

    def list_configs(self) -> Dict[str, Any]:
        with self._lock:
            return {cam_id: cfg.to_dict() for cam_id, cfg in self._configs.items()}

    def set_boundary_line(
        self,
        camera_id: str,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        restricted_side: str = "A",
        line_id: str = "main",
        touch_tolerance: float = 0.03
    ) -> Dict[str, Any]:
        with self._lock:
            # Clamp normalized coordinates to [0.0, 1.0]
            x1 = max(0.0, min(1.0, float(x1)))
            y1 = max(0.0, min(1.0, float(y1)))
            x2 = max(0.0, min(1.0, float(x2)))
            y2 = max(0.0, min(1.0, float(y2)))
            side = restricted_side.upper() if restricted_side.upper() in ("A", "B") else "A"

            if camera_id not in self._configs:
                self._configs[camera_id] = CameraBoundaryConfig(camera_id=camera_id)

            cfg = self._configs[camera_id]
            cfg.line = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
            cfg.restricted_side = side
            cfg.line_id = line_id
            cfg.touch_tolerance = float(touch_tolerance)
            self._save_configs()
            return cfg.to_dict()

    def set_zone(
        self,
        camera_id: str,
        zone_id: str,
        polygon: List[List[float]]
    ) -> Dict[str, Any]:
        with self._lock:
            if camera_id not in self._configs:
                self._configs[camera_id] = CameraBoundaryConfig(camera_id=camera_id)

            # Validate polygon: list of [x, y] in [0.0, 1.0]
            valid_pts = []
            for pt in polygon:
                if len(pt) >= 2:
                    vx = max(0.0, min(1.0, float(pt[0])))
                    vy = max(0.0, min(1.0, float(pt[1])))
                    valid_pts.append([vx, vy])

            cfg = self._configs[camera_id]
            cfg.zones[zone_id] = valid_pts
            self._save_configs()
            return cfg.to_dict()

    def delete_zone(self, camera_id: str, zone_id: str) -> bool:
        with self._lock:
            if camera_id in self._configs and zone_id in self._configs[camera_id].zones:
                del self._configs[camera_id].zones[zone_id]
                self._save_configs()
                return True
            return False

    def get_anomaly_thresholds(self, camera_id: Optional[str] = None) -> Dict[str, Any]:
        with self._lock:
            if camera_id and camera_id in self._configs and self._configs[camera_id].anomaly_thresholds:
                return self._configs[camera_id].anomaly_thresholds
            return self._default_thresholds.to_dict()

    def set_anomaly_thresholds(self, camera_id: Optional[str], thresholds: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            if camera_id:
                if camera_id not in self._configs:
                    self._configs[camera_id] = CameraBoundaryConfig(camera_id=camera_id)
                self._configs[camera_id].anomaly_thresholds = thresholds
                if camera_id in self._anomaly_engines:
                    self._anomaly_engines[camera_id].update_thresholds(thresholds)
                self._save_configs()
                return self._configs[camera_id].anomaly_thresholds
            else:
                self._default_thresholds = AnomalyThresholds.from_dict(thresholds)
                for eng in self._anomaly_engines.values():
                    eng.update_thresholds(thresholds)
                return self._default_thresholds.to_dict()

    def get_or_create_crossing_engine(self, camera_id: str) -> CrossingEngine:
        with self._lock:
            if camera_id not in self._crossing_engines:
                self._crossing_engines[camera_id] = CrossingEngine()
            return self._crossing_engines[camera_id]

    def get_or_create_anomaly_engine(self, camera_id: str) -> AnomalyEngine:
        with self._lock:
            if camera_id not in self._anomaly_engines:
                # Load custom camera thresholds if configured
                cfg = self._configs.get(camera_id)
                t_dict = cfg.anomaly_thresholds if (cfg and cfg.anomaly_thresholds) else self._default_thresholds.to_dict()
                thresholds = AnomalyThresholds.from_dict(t_dict)
                self._anomaly_engines[camera_id] = AnomalyEngine(thresholds=thresholds)
            return self._anomaly_engines[camera_id]

    def process_frame(
        self,
        camera_id: str,
        frame: Optional[np.ndarray],
        observations: List[TrackObservation],
        timestamp: Optional[float] = None,
        frame_shape: Optional[Tuple[int, int]] = None,
        track_to_identity: Optional[Dict[int, Any]] = None
    ) -> Tuple[List[CrossingEvent], List[AnomalyEvent]]:
        """
        Processes track observations for a given camera frame.
        - frame: Optional numpy array (H, W, C). If provided and clip_manager is active, pushes to circular RAM buffer.
        - observations: List of current frame track observations.
        - timestamp: Monotonic or epoch timestamp.
        - frame_shape: (height, width) if frame is None.
        - track_to_identity: Optional map of track_id -> (person_id, display_name) or dict.
        """
        now = float(timestamp if timestamp is not None else time.time())

        # Determine frame dimensions
        if frame is not None:
            h, w = frame.shape[:2]
            if self.clip_manager is not None:
                self.clip_manager.add_frame(camera_id, frame, now)
        elif frame_shape is not None:
            h, w = frame_shape
        else:
            h, w = 1080, 1920  # Default canvas dimensions

        cfg = self._configs.get(camera_id)
        crossing_engine = self.get_or_create_crossing_engine(camera_id)
        anomaly_engine = self.get_or_create_anomaly_engine(camera_id)

        all_crossing_events: List[CrossingEvent] = []
        all_anomaly_events: List[AnomalyEvent] = []

        pixel_zones: Dict[str, np.ndarray] = {}

        if cfg is not None and cfg.active:
            # 1. Evaluate Line Crossings
            if cfg.line is not None:
                norm_line = NormalizedLine(
                    x1=cfg.line["x1"],
                    y1=cfg.line["y1"],
                    x2=cfg.line["x2"],
                    y2=cfg.line["y2"]
                )
                pixel_line = norm_line.to_pixel_line(w, h)
                restr_side = 1 if cfg.restricted_side.upper() == "A" else -1

                touch_tol_px = (cfg.touch_tolerance * min(w, h)) if getattr(cfg, "touch_tolerance", 0.0) > 0 else 0.0
                line_events = crossing_engine.check_line_crossings(
                    boundary_id=cfg.line_id,
                    line=pixel_line,
                    restricted_side=restr_side,
                    observations=observations,
                    timestamp=now,
                    touch_tolerance_px=touch_tol_px
                )
                all_crossing_events.extend(line_events)

            # 2. Evaluate Zone Transitions
            for zid, poly_pts in cfg.zones.items():
                if len(poly_pts) >= 3:
                    pixel_pts = np.array([[int(round(pt[0] * w)), int(round(pt[1] * h))] for pt in poly_pts], dtype=np.int32)
                    pixel_zones[zid] = pixel_pts
                    zone_events = crossing_engine.check_zone_transitions(
                        zone_id=zid,
                        polygon_points=pixel_pts,
                        observations=observations,
                        timestamp=now
                    )
                    all_crossing_events.extend(zone_events)

        # 3. Evaluate Behavioral Anomalies (Rule-based indicators)
        anomaly_events = anomaly_engine.update(
            observations=observations,
            timestamp=now,
            zones=pixel_zones if pixel_zones else None
        )
        all_anomaly_events.extend(anomaly_events)

        # 4. Integrate with EventHub & Trigger Event Clips
        self._dispatch_events_to_hub(
            camera_id=camera_id,
            crossing_events=all_crossing_events,
            anomaly_events=all_anomaly_events,
            track_to_identity=track_to_identity
        )

        return all_crossing_events, all_anomaly_events

    def _dispatch_events_to_hub(
        self,
        camera_id: str,
        crossing_events: List[CrossingEvent],
        anomaly_events: List[AnomalyEvent],
        track_to_identity: Optional[Dict[int, Any]] = None
    ) -> None:
        if self.event_hub is None:
            return

        def resolve_identity(tid: int) -> Tuple[Optional[str], Optional[str]]:
            if not track_to_identity or tid not in track_to_identity:
                return None, None
            val = track_to_identity[tid]
            if isinstance(val, (list, tuple)) and len(val) >= 2:
                return str(val[0]), str(val[1])
            elif isinstance(val, dict):
                return val.get("person_id"), val.get("display_name") or val.get("name")
            return str(val), None

        for ev in crossing_events:
            person_id, display_name = resolve_identity(ev.track_id)
            meta = {
                "boundary_id": ev.boundary_id,
                "from_side": ev.from_side,
                "to_side": ev.to_side,
                "location": list(ev.location)
            }
            if ev.direction == "BORDER_TOUCH":
                ev_type = "BORDER_TOUCH"
                details = f"Track #{ev.track_id} touched virtual border '{ev.boundary_id}'"
                severity = "WARNING"
            else:
                ev_type = "CROSSING"
                details = f"Track #{ev.track_id} crossed boundary '{ev.boundary_id}' ({ev.direction})"
                severity = "WARNING" if ev.direction in ("INTRUDING", "ZONE_INTRUDED") else "INFO"

            if display_name:
                details += f" [Identified: {display_name}]"

            rec = self.event_hub.log_event(
                event_type=ev_type,
                subsystem="BOUNDARY",
                camera_id=camera_id,
                timestamp=ev.timestamp,
                severity=severity,
                direction=ev.direction,
                track_id=ev.track_id,
                person_id=person_id,
                display_name=display_name,
                details=details,
                confidence=ev.confidence,
                generate_clip=True,
                metadata=meta
            )
            ev.event_id = rec.id if rec else None

        for a_ev in anomaly_events:
            person_id, display_name = resolve_identity(a_ev.track_id)
            meta = {
                "duration": a_ev.duration,
                "location": list(a_ev.location),
                "category": a_ev.category
            }
            details = a_ev.details
            if display_name:
                details += f" [Identified: {display_name}]"

            self.event_hub.log_event(
                event_type=a_ev.event_type,
                subsystem="BOUNDARY",
                camera_id=camera_id,
                timestamp=a_ev.timestamp,
                severity=a_ev.severity,
                track_id=a_ev.track_id,
                person_id=person_id,
                display_name=display_name,
                details=details,
                confidence=1.0,
                generate_clip=True,
                metadata=meta
            )

    def process_uploaded_video(
        self,
        video_path: str,
        camera_id: Optional[str] = None,
        line: Optional[Dict[str, float]] = None,
        restricted_side: str = "A",
        precomputed_frames_and_tracks: Optional[List[Tuple[float, np.ndarray, List[TrackObservation]]]] = None,
        fps_override: Optional[float] = None,
        adapter: Any = None
    ) -> Dict[str, Any]:
        """
        Executes Boundary and Anomaly detection over an uploaded video.
        Uses real multi-object tracking (YOLO + ByteTrack) when real video is supplied.
        Enforces strict semantic separation between Line Crossings, Zone Transitions, and Anomaly Indicators.
        """
        cam_id = camera_id or f"upload_{int(time.time())}"

        # Apply temporary line config if specified
        if line is not None:
            self.set_boundary_line(
                camera_id=cam_id,
                x1=line["x1"],
                y1=line["y1"],
                x2=line["x2"],
                y2=line["y2"],
                restricted_side=restricted_side,
                touch_tolerance=touch_tolerance
            )

        captured_frames: List[Tuple[float, np.ndarray]] = []
        recorded_crossings: List[CrossingEvent] = []
        recorded_anomalies: List[AnomalyEvent] = []

        total_frames = 0
        fps = 25.0

        if precomputed_frames_and_tracks is not None:
            # Deterministic / synthetic test sequence
            for ts, frame, obs_list in precomputed_frames_and_tracks:
                total_frames += 1
                captured_frames.append((ts, frame))
                crs, anm = self.process_frame(
                    camera_id=cam_id,
                    frame=frame,
                    observations=obs_list,
                    timestamp=ts
                )
                recorded_crossings.extend(crs)
                recorded_anomalies.extend(anm)
        else:
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")

            from edge.boundary.adapter import get_tracking_adapter
            tracking_adapter = adapter or get_tracking_adapter()
            if hasattr(tracking_adapter, "reset"):
                tracking_adapter.reset()

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")

            cap_fps = cap.get(cv2.CAP_PROP_FPS)
            if cap_fps and cap_fps > 1.0:
                fps = cap_fps
            if fps_override:
                fps = fps_override

            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                ts = frame_idx / fps
                captured_frames.append((ts, frame))

                # Run real detector + tracker on the frame
                observations = tracking_adapter.detect_and_track(frame, timestamp=ts)

                # Feed observations into the exact same BoundaryManager pipeline
                crs, anm = self.process_frame(
                    camera_id=cam_id,
                    frame=frame,
                    observations=observations,
                    timestamp=ts
                )
                recorded_crossings.extend(crs)
                recorded_anomalies.extend(anm)

                total_frames += 1
                frame_idx += 1
            cap.release()

        # Compile offline clips for all detected events via clip_manager if present
        if self.clip_manager is not None and self.event_hub is not None:
            events = self.event_hub.list_events(camera_id=cam_id, limit=500)
            for ev in events:
                if ev.clip_status == "PENDING":
                    clip_filename = f"event_{ev.id}.mp4"
                    clip_out_path = os.path.join(self.clip_manager.output_dir, clip_filename)
                    ok, generated_path = self.clip_manager.extract_offline_clip(
                        frames_with_timestamps=captured_frames,
                        event_time=ev.timestamp,
                        output_path=clip_out_path,
                        event_id=ev.id,
                        fps=fps
                    )
                    self.event_hub.update_clip_status(ev.id, generated_path, success=ok)

        # ── Explicit Semantic Separation ───────────────────────────────────────────
        # 1. Line Crossings (Only INTRUDING and RETREATING)
        line_crossings = [c for c in recorded_crossings if c.direction in ("INTRUDING", "RETREATING")]
        intruding_count = sum(1 for c in line_crossings if c.direction == "INTRUDING")
        retreating_count = sum(1 for c in line_crossings if c.direction == "RETREATING")

        # 2. Zone Transitions (Entered vs Exited)
        zone_events = [c for c in recorded_crossings if c.direction in ("ZONE_INTRUDED", "ZONE_EXITED")]
        entered_count = sum(1 for c in zone_events if c.direction == "ZONE_INTRUDED")
        exited_count = sum(1 for c in zone_events if c.direction == "ZONE_EXITED")

        # 3. Rule-Based Anomaly Indicators
        anomalies_by_type: Dict[str, int] = {}
        for a in recorded_anomalies:
            anomalies_by_type[a.event_type] = anomalies_by_type.get(a.event_type, 0) + 1

        duration = (captured_frames[-1][0] - captured_frames[0][0]) if len(captured_frames) > 1 else 0.0

        return {
            "status": "success",
            "camera_id": cam_id,
            "total_frames": total_frames,
            "duration_seconds": round(duration, 2),
            # Explicit semantic metric breakdowns
            "line_crossings": {
                "total": len(line_crossings),
                "intruding": intruding_count,
                "retreating": retreating_count
            },
            "zone_events": {
                "total": len(zone_events),
                "entered": entered_count,
                "exited": exited_count
            },
            "anomalies": {
                "total": len(recorded_anomalies),
                "by_type": anomalies_by_type,
                "category": "Rule-based anomaly indicators"
            },
            # Explicit separate event lists
            "crossing_events": [
                {
                    "track_id": c.track_id,
                    "direction": c.direction,
                    "timestamp": round(c.timestamp, 2),
                    "location": c.location,
                    "confidence": c.confidence
                }
                for c in line_crossings
            ],
            "zone_transition_events": [
                {
                    "zone_id": c.boundary_id,
                    "track_id": c.track_id,
                    "direction": c.direction,
                    "timestamp": round(c.timestamp, 2),
                    "location": c.location
                }
                for c in zone_events
            ],
            "anomaly_events": [
                {
                    "track_id": a.track_id,
                    "event_type": a.event_type,
                    "severity": a.severity,
                    "timestamp": round(a.timestamp, 2),
                    "details": a.details,
                    "category": a.category
                }
                for a in recorded_anomalies
            ],
            # Canonical dashboard counters (strictly line crossings, not zone transitions)
            "persons_crossed": len(line_crossings),
            "intruding": intruding_count,
            "retreating": retreating_count
        }




    def process_uploaded_video_stream(
        self,
        video_path: str,
        camera_id: Optional[str] = None,
        line: Optional[Dict[str, float]] = None,
        restricted_side: str = "A",
        touch_tolerance: float = 0.03,
        fps_override: Optional[float] = None,
        adapter: Any = None
    ):
        import cv2, json
        
        cam_id = camera_id or f"upload_{int(time.time())}"
        
        if line is not None:
            self.set_boundary_line(
                camera_id=cam_id,
                x1=line["x1"],
                y1=line["y1"],
                x2=line["x2"],
                y2=line["y2"],
                restricted_side=restricted_side,
                touch_tolerance=touch_tolerance
            )
            
        if not os.path.exists(video_path):
            yield json.dumps({"error": "Video file not found"}) + "\n"
            return
            
        from edge.boundary.adapter import get_tracking_adapter
        tracking_adapter = adapter or get_tracking_adapter()
        if hasattr(tracking_adapter, "reset"):
            tracking_adapter.reset()
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            yield json.dumps({"error": "Failed to open video"}) + "\n"
            return
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0: total_frames = 1000
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        if fps_override: fps = fps_override
        
        captured_frames = []
        recorded_crossings = []
        recorded_anomalies = []
        
        wall_start = time.time()
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            ts = frame_idx / fps
            captured_frames.append((ts, frame))
            
            observations = tracking_adapter.detect_and_track(frame, timestamp=ts)
            
            crs, anm = self.process_frame(
                camera_id=cam_id,
                frame=frame,
                observations=observations,
                timestamp=ts
            )
            recorded_crossings.extend(crs)
            recorded_anomalies.extend(anm)
            
            frame_idx += 1
            
            if frame_idx % 5 == 0 or crs or anm:
                progress = min(100, int((frame_idx / total_frames) * 100))
                logs = []
                for c in crs:
                    logs.append({
                        "timestamp": time.strftime("%H:%M:%S", time.gmtime(c.timestamp)),
                        "type": "Line Crossing",
                        "message": f"Track {c.track_id} {c.direction}"
                    })
                for a in anm:
                    logs.append({
                        "timestamp": time.strftime("%H:%M:%S", time.gmtime(a.timestamp)),
                        "type": "Anomaly",
                        "message": f"Track {a.track_id} {a.event_type}"
                    })
                if logs or frame_idx % 10 == 0:
                    yield json.dumps({"progress": progress, "logs": logs}) + "\n"
                
        cap.release()
        
        clip_status_map = {}
        if self.clip_manager is not None and self.event_hub is not None:
            events = self.event_hub.list_events(camera_id=cam_id, limit=500)
            for ev in events:
                if ev.clip_status == "PENDING":
                    clip_filename = f"event_{ev.id}.mp4"
                    clip_out_path = os.path.join(self.clip_manager.output_dir, clip_filename)
                    ok, generated_path = self.clip_manager.extract_offline_clip(
                        frames_with_timestamps=captured_frames,
                        event_time=ev.timestamp,
                        output_path=clip_out_path,
                        event_id=ev.id,
                        fps=fps
                    )
                    self.event_hub.update_clip_status(ev.id, generated_path if ok else None, success=ok)
                    clip_status_map[ev.id] = "READY" if ok else "FAILED"
                    
        line_crossings = [c for c in recorded_crossings if c.direction in ("INTRUDING", "RETREATING", "BORDER_TOUCH")]
        intruding_count = sum(1 for c in line_crossings if c.direction == "INTRUDING")
        retreating_count = sum(1 for c in line_crossings if c.direction == "RETREATING")
        touch_count = sum(1 for c in line_crossings if c.direction == "BORDER_TOUCH")
        
        zone_events = [c for c in recorded_crossings if c.direction in ("ZONE_INTRUDED", "ZONE_EXITED")]
        entered_count = sum(1 for c in zone_events if c.direction == "ZONE_INTRUDED")
        exited_count = sum(1 for c in zone_events if c.direction == "ZONE_EXITED")
        
        anomalies_by_type = {}
        for a in recorded_anomalies:
            anomalies_by_type[a.event_type] = anomalies_by_type.get(a.event_type, 0) + 1
            
        duration = (captured_frames[-1][0] - captured_frames[0][0]) if len(captured_frames) > 1 else 0.0

        crossing_payload = []
        for c in line_crossings:
            eid = getattr(c, "event_id", None)
            c_status = clip_status_map.get(eid, "UNAVAILABLE") if eid else "UNAVAILABLE"
            mins = int(c.timestamp // 60)
            secs = c.timestamp % 60
            ts_str = f"{mins:02d}:{secs:05.2f}"
            crossing_payload.append({
                "event_id": eid,
                "event_type": "BORDER_TOUCH" if c.direction == "BORDER_TOUCH" else "BORDER_LINE_CROSSING",
                "track_id": c.track_id,
                "direction": c.direction,
                "timestamp": round(c.timestamp, 2),
                "time_formatted": ts_str,
                "location": c.location,
                "confidence": round(c.confidence, 2),
                "clip_status": c_status,
                "clip_url": f"/api/events/{eid}/clip" if c_status == "READY" else None
            })
        
        res = {
            "progress": 100,
            "results": {
                "status": "success",
                "camera_id": cam_id,
                "total_frames": frame_idx,
                "duration_seconds": round(duration, 2),
                "line_crossings": {
                    "total": len(line_crossings),
                    "intruding": intruding_count,
                    "retreating": retreating_count,
                    "border_touch": touch_count
                },
                "zone_events": {
                    "total": len(zone_events),
                    "entered": entered_count,
                    "exited": exited_count
                },
                "anomalies": {
                    "total": len(recorded_anomalies),
                    "by_type": anomalies_by_type,
                    "category": "Rule-based anomaly indicators"
                },
                "crossing_events": crossing_payload,
                "zone_transition_events": [
                    {
                        "zone_id": c.boundary_id,
                        "track_id": c.track_id,
                        "direction": c.direction,
                        "timestamp": round(c.timestamp, 2),
                        "location": c.location
                    }
                    for c in zone_events
                ],
                "anomaly_events": [
                    {
                        "track_id": a.track_id,
                        "event_type": a.event_type,
                        "severity": a.severity,
                        "timestamp": round(a.timestamp, 2),
                        "details": a.details,
                        "category": a.category
                    }
                    for a in recorded_anomalies
                ],
                "persons_crossed": len(line_crossings),
                "intruding": intruding_count,
                "retreating": retreating_count
            }
        }
        yield json.dumps(res) + "\n"


_global_boundary_manager: Optional[BoundaryManager] = None


def get_boundary_manager() -> BoundaryManager:
    """Returns or instantiates the global singleton BoundaryManager."""
    global _global_boundary_manager
    if _global_boundary_manager is None:
        from edge.events.event_hub import get_event_hub
        hub = get_event_hub()
        _global_boundary_manager = BoundaryManager(
            event_hub=hub,
            clip_manager=hub.clip_manager
        )
    return _global_boundary_manager
