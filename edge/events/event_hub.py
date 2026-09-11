"""
EventHub — Centralized Event Domain Coordinator & Persistence Engine.

Coordinates event ingestion across Boundary, Anomaly, and Vehicle subsystems:
- Immediate non-blocking logging as PENDING when clips are requested.
- Asynchronous status progression (PENDING -> READY or FAILED).
- Isolated SQLite event storage.
- Thread-safe in-memory ring buffer for live UI subscriber dispatch.
- Integrated retention cleanup for event clips.
"""

import os
import sqlite3
import json
import time
import uuid
import logging
import threading
import concurrent.futures
from typing import Callable
from datetime import datetime, timedelta
from collections import deque
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict

from edge.events.clip_manager import EventClipManager

log = logging.getLogger("event_hub")


@dataclass
class SystemEvent:
    id: str
    event_type: str                  # e.g., "CROSSING", "LOITERING", "CRAWLING", "ANPR"
    subsystem: str                   # "BOUNDARY", "ANOMALY", "VEHICLE"
    camera_id: str
    timestamp: float
    datetime_str: str
    severity: str = "WARNING"        # "INFO", "WARNING", "CRITICAL"
    direction: Optional[str] = None  # "INTRUDING", "RETREATING", "IN", "OUT"
    track_id: Optional[int] = None
    person_id: Optional[str] = None  # Optional Face AI enrichment
    display_name: Optional[str] = None
    details: str = ""
    confidence: float = 1.0
    clip_status: str = "NONE"        # "NONE", "PENDING", "READY", "FAILED"
    clip_path: Optional[str] = None  # Internal disk path (never exposed in public API)
    snapshot_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_internal_paths: bool = False) -> Dict[str, Any]:
        d = asdict(self)
        if not include_internal_paths:
            d.pop("clip_path", None)
            d.pop("snapshot_path", None)
            d["has_clip"] = (self.clip_status == "READY")
            d["clip_url"] = f"/api/events/{self.id}/clip" if self.clip_status == "READY" else None
            d["snapshot_url"] = f"/api/events/{self.id}/snapshot" if self.snapshot_path else None
        return d


class EventHub:
    """
    Thread-safe coordinator for multi-modal events and clip lifecycles.
    """
    def __init__(
        self,
        db_path: Optional[str] = None,
        clip_manager: Optional[EventClipManager] = None,
        max_live_events: int = 100
    ):
        if db_path is None:
            base_dir = os.environ.get("SIH26187_DATA")
            if not base_dir:
                base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data"))
            db_path = os.path.join(base_dir, "db", "events.db")

        self.db_path = os.path.abspath(db_path)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        self.clip_manager = clip_manager or EventClipManager()
        self._live_events: deque = deque(maxlen=max_live_events)
        self._subscribers: List[Callable[["SystemEvent"], None]] = []
        self._dispatch_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="eh_pubsub")
        self._lock = threading.Lock()

        self._init_db()

    def subscribe(self, callback: Callable[["SystemEvent"], None]) -> None:
        with self._lock:
            if callback not in self._subscribers:
                self._subscribers.append(callback)

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock:
            with self._get_conn() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS system_events (
                        id TEXT PRIMARY KEY,
                        event_type TEXT NOT NULL,
                        subsystem TEXT NOT NULL,
                        camera_id TEXT NOT NULL,
                        timestamp REAL NOT NULL,
                        datetime_str TEXT NOT NULL,
                        severity TEXT NOT NULL,
                        direction TEXT,
                        track_id INTEGER,
                        person_id TEXT,
                        display_name TEXT,
                        details TEXT,
                        confidence REAL,
                        clip_status TEXT NOT NULL,
                        clip_path TEXT,
                        snapshot_path TEXT,
                        metadata TEXT
                    )
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_events_subsystem ON system_events(subsystem)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_events_camera ON system_events(camera_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON system_events(timestamp DESC)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_events_clip_status ON system_events(clip_status)")
                conn.commit()

    def log_event(
        self,
        event_type: str,
        subsystem: str,
        camera_id: str,
        timestamp: Optional[float] = None,
        severity: str = "WARNING",
        direction: Optional[str] = None,
        track_id: Optional[int] = None,
        person_id: Optional[str] = None,
        display_name: Optional[str] = None,
        details: str = "",
        confidence: float = 1.0,
        snapshot_path: Optional[str] = None,
        generate_clip: bool = True,
        pre_seconds: Optional[float] = None,
        post_seconds: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> SystemEvent:
        """
        Logs an event immediately to SQLite and live subscriber buffer.
        If generate_clip is True, triggers EventClipManager non-blockingly and sets
        clip_status='PENDING'. When the clip finishes encoding, it asynchronously updates
        to 'READY' (or 'FAILED').
        """
        now = float(timestamp if timestamp is not None else time.time())
        dt_str = datetime.fromtimestamp(now).isoformat()
        event_id = str(uuid.uuid4())
        meta = metadata or {}

        clip_status = "NONE"
        clip_path = None

        if generate_clip and self.clip_manager is not None:
            clip_status = "PENDING"
            # Trigger asynchronous clip collection & compilation
            self.clip_manager.trigger_clip(
                camera_id=camera_id,
                event_id=event_id,
                event_time=now,
                pre_seconds=pre_seconds,
                post_seconds=post_seconds,
                on_complete=self._on_clip_complete
            )

        event = SystemEvent(
            id=event_id,
            event_type=event_type,
            subsystem=subsystem,
            camera_id=camera_id,
            timestamp=now,
            datetime_str=dt_str,
            severity=severity,
            direction=direction,
            track_id=track_id,
            person_id=person_id,
            display_name=display_name,
            details=details,
            confidence=confidence,
            clip_status=clip_status,
            clip_path=clip_path,
            snapshot_path=snapshot_path,
            metadata=meta
        )

        # Persist to SQLite immediately
        with self._lock:
            with self._get_conn() as conn:
                conn.execute("""
                    INSERT INTO system_events
                    (id, event_type, subsystem, camera_id, timestamp, datetime_str,
                     severity, direction, track_id, person_id, display_name,
                     details, confidence, clip_status, clip_path, snapshot_path, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    event.id, event.event_type, event.subsystem, event.camera_id,
                    event.timestamp, event.datetime_str, event.severity, event.direction,
                    event.track_id, event.person_id, event.display_name,
                    event.details, event.confidence, event.clip_status,
                    event.clip_path, event.snapshot_path, json.dumps(event.metadata)
                ))
                conn.commit()

            # Add to in-memory live dispatch queue
            self._live_events.append(event)
            subs = list(self._subscribers)

        for sub in subs:
            self._dispatch_pool.submit(sub, event)

        log.info(f"Logged {subsystem} event {event_id} ({event_type}) on cam {camera_id} [clip={clip_status}]")
        return event

    def update_clip_status(self, event_id: str, clip_path: str, success: bool = True) -> None:
        """Updates clip status and path for an event."""
        self._on_clip_complete(event_id, clip_path, success)

    def _on_clip_complete(self, event_id: str, clip_path: str, success: bool) -> None:
        """Asynchronous callback executed by background clip worker."""
        new_status = "READY" if success else "FAILED"
        final_path = clip_path if success else None

        with self._lock:
            try:
                with self._get_conn() as conn:
                    conn.execute("""
                        UPDATE system_events
                        SET clip_status = ?, clip_path = ?
                        WHERE id = ?
                    """, (new_status, final_path, event_id))
                    conn.commit()

                # Also update in-memory instance if present
                for ev in self._live_events:
                    if ev.id == event_id:
                        ev.clip_status = new_status
                        ev.clip_path = final_path
                        break

                log.info(f"Event {event_id} clip updated to {new_status} (path={final_path})")
            except Exception as e:
                log.error(f"Failed to update clip status for event {event_id}: {e}")

    def get_event(self, event_id: str) -> Optional[SystemEvent]:
        """Fetches a single event by its UUID."""
        with self._lock:
            with self._get_conn() as conn:
                row = conn.execute("SELECT * FROM system_events WHERE id = ?", (event_id,)).fetchone()
                if not row:
                    return None
                return self._row_to_event(row)

    def list_events(
        self,
        subsystem: Optional[str] = None,
        camera_id: Optional[str] = None,
        event_type: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        search_query: Optional[str] = None,
        limit: int = 50
    ) -> List[SystemEvent]:
        """Lists events with filtering by subsystem, camera, type, time, and text search."""
        clauses = []
        params: List[Any] = []

        if subsystem:
            clauses.append("subsystem = ?")
            params.append(subsystem.upper())
        if camera_id:
            clauses.append("camera_id = ?")
            params.append(camera_id)
        if event_type:
            clauses.append("event_type = ?")
            params.append(event_type.upper())
        if start_time is not None:
            clauses.append("timestamp >= ?")
            params.append(start_time)
        if end_time is not None:
            clauses.append("timestamp <= ?")
            params.append(end_time)
        if search_query:
            clauses.append("(details LIKE ? OR metadata LIKE ?)")
            q = f"%{search_query}%"
            params.append(q)
            params.append(q)

        where_clause = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"SELECT * FROM system_events {where_clause} ORDER BY timestamp DESC LIMIT ?"
        params.append(int(limit))

        with self._lock:
            with self._get_conn() as conn:
                rows = conn.execute(query, params).fetchall()
                return [self._row_to_event(r) for r in rows]

    def get_live_events(self) -> List[SystemEvent]:
        """Returns the in-memory ring buffer of recent events for live polling."""
        with self._lock:
            return list(self._live_events)

    def cleanup_old_events(self, days: int) -> Tuple[int, int]:
        """
        Deletes events and their associated video clips older than N days.
        Returns: (deleted_events_count, deleted_clips_count)
        """
        cutoff = (datetime.now() - timedelta(days=days)).timestamp()
        deleted_events = 0
        deleted_clips = 0

        with self._lock:
            with self._get_conn() as conn:
                rows = conn.execute(
                    "SELECT id, clip_path, snapshot_path FROM system_events WHERE timestamp < ?",
                    (cutoff,)
                ).fetchall()

                for row in rows:
                    c_path = row["clip_path"]
                    s_path = row["snapshot_path"]
                    if c_path and os.path.exists(c_path):
                        try:
                            os.remove(c_path)
                            deleted_clips += 1
                        except Exception as e:
                            log.warning(f"Failed to remove clip {c_path}: {e}")

                    if s_path and os.path.exists(s_path):
                        try:
                            os.remove(s_path)
                        except Exception as e:
                            log.warning(f"Failed to remove snapshot {s_path}: {e}")

                    conn.execute("DELETE FROM system_events WHERE id = ?", (row["id"],))
                    deleted_events += 1

                conn.commit()

        log.info(f"Cleaned up {deleted_events} events and {deleted_clips} clips older than {days} days")
        return deleted_events, deleted_clips

    def _row_to_event(self, row: sqlite3.Row) -> SystemEvent:
        meta_str = row["metadata"]
        try:
            meta = json.loads(meta_str) if meta_str else {}
        except Exception:
            meta = {}

        return SystemEvent(
            id=row["id"],
            event_type=row["event_type"],
            subsystem=row["subsystem"],
            camera_id=row["camera_id"],
            timestamp=float(row["timestamp"]),
            datetime_str=row["datetime_str"],
            severity=row["severity"],
            direction=row["direction"],
            track_id=row["track_id"],
            person_id=row["person_id"],
            display_name=row["display_name"],
            details=row["details"] or "",
            confidence=float(row["confidence"] if row["confidence"] is not None else 1.0),
            clip_status=row["clip_status"],
            clip_path=row["clip_path"],
            snapshot_path=row["snapshot_path"],
            metadata=meta
        )


_global_event_hub: Optional[EventHub] = None


def get_event_hub() -> EventHub:
    global _global_event_hub
    if _global_event_hub is None:
        _global_event_hub = EventHub()
    return _global_event_hub
