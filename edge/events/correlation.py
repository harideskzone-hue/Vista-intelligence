import time
import logging
from typing import Dict, List
from collections import deque
import threading

from edge.events.event_hub import SystemEvent, get_event_hub

log = logging.getLogger("correlation_engine")

class CorrelationEngine:
    def __init__(self):
        self.lock = threading.Lock()
        self.camera_buffers: Dict[str, deque] = {}
        self.dedup_cache = {}
        
        self.FACE_VEHICLE_WINDOW = 5.0
        self.WATCHLIST_BOUNDARY_WINDOW = 10.0
        self.CORRELATION_DEDUP_WINDOW = 30.0
        
        hub = get_event_hub()
        hub.subscribe(self.process_event)

    def process_event(self, event: SystemEvent):
        if event.subsystem == "CORRELATION":
            return
            
        with self.lock:
            now = time.time()
            expired_keys = [k for k, v in self.dedup_cache.items() if now - v > self.CORRELATION_DEDUP_WINDOW]
            for k in expired_keys:
                del self.dedup_cache[k]
                
            camera_id = event.camera_id
            if camera_id not in self.camera_buffers:
                self.camera_buffers[camera_id] = deque(maxlen=200)
                
            buffer = self.camera_buffers[camera_id]
            buffer.append(event)
            
            self._evaluate_face_vehicle(event, buffer)
            self._evaluate_watchlist_boundary(event, buffer)

    def _evaluate_face_vehicle(self, current_event: SystemEvent, buffer: deque):
        if current_event.subsystem not in ("FACE", "VEHICLE"):
            return
            
        now = current_event.timestamp
        for historical_event in list(buffer):
            if historical_event.id == current_event.id:
                continue
                
            if abs(now - historical_event.timestamp) > self.FACE_VEHICLE_WINDOW:
                continue
                
            if (current_event.subsystem == "FACE" and historical_event.subsystem == "VEHICLE") or \
               (current_event.subsystem == "VEHICLE" and historical_event.subsystem == "FACE"):
                
                key = tuple(sorted([current_event.id, historical_event.id]))
                rule_key = (current_event.camera_id, key[0], key[1], "FACE_VEHICLE_PROXIMITY")
                if rule_key in self.dedup_cache:
                    continue
                    
                self.dedup_cache[rule_key] = time.time()
                
                spatial_evidence = False
                c_meta = current_event.metadata
                h_meta = historical_event.metadata
                if "bounding_box" in c_meta and "bounding_box" in h_meta:
                    spatial_evidence = True
                
                confidence_level = "WEAK"
                confidence = 0.4
                
                if c_meta.get("zone_id") and c_meta.get("zone_id") == h_meta.get("zone_id"):
                    confidence_level = "MEDIUM"
                    confidence = 0.6
                
                if spatial_evidence:
                    confidence_level = "STRONG"
                    confidence = 0.8
                
                if current_event.event_type in ("FACE_MATCH", "ANPR_MATCH", "VEHICLE_IDENTIFIED", "FACE_CONFIRMED") or \
                   historical_event.event_type in ("FACE_MATCH", "ANPR_MATCH", "VEHICLE_IDENTIFIED", "FACE_CONFIRMED"):
                    confidence += 0.1
                    if confidence_level == "WEAK":
                        confidence_level = "MEDIUM"
                
                self._emit_correlation(
                    rule_name="FACE_VEHICLE_PROXIMITY",
                    events=[current_event, historical_event],
                    confidence=confidence,
                    confidence_level=confidence_level,
                    camera_id=current_event.camera_id,
                    time_delta_ms=int(abs(now - historical_event.timestamp) * 1000)
                )

    def _evaluate_watchlist_boundary(self, current_event: SystemEvent, buffer: deque):
        if current_event.event_type not in ("ANPR_MATCH", "FACE_MATCH", "BOUNDARY_CROSSED", "ANOMALY", "PERSON_CROSSED"):
            return
            
        now = current_event.timestamp
        for historical_event in list(buffer):
            if historical_event.id == current_event.id:
                continue
                
            if abs(now - historical_event.timestamp) > self.WATCHLIST_BOUNDARY_WINDOW:
                continue
                
            types = {current_event.event_type, historical_event.event_type}
            match_types = {"ANPR_MATCH", "FACE_MATCH"}
            boundary_types = {"BOUNDARY_CROSSED", "ANOMALY", "PERSON_CROSSED"}
            
            if types & match_types and types & boundary_types:
                key = tuple(sorted([current_event.id, historical_event.id]))
                rule_key = (current_event.camera_id, key[0], key[1], "WATCHLIST_BOUNDARY_ALERT")
                if rule_key in self.dedup_cache:
                    continue
                    
                self.dedup_cache[rule_key] = time.time()
                
                self._emit_correlation(
                    rule_name="WATCHLIST_BOUNDARY_ALERT",
                    events=[current_event, historical_event],
                    confidence=0.9,
                    confidence_level="STRONG",
                    camera_id=current_event.camera_id,
                    time_delta_ms=int(abs(now - historical_event.timestamp) * 1000),
                    is_alert=True
                )

    def _emit_correlation(self, rule_name: str, events: List[SystemEvent], confidence: float, confidence_level: str, camera_id: str, time_delta_ms: int, is_alert: bool = False):
        hub = get_event_hub()
        
        identified_person = None
        identified_vehicle = None
        
        for e in events:
            if e.subsystem == "FACE" and e.person_id and e.person_id != "UNKNOWN":
                identified_person = e.person_id
            if e.subsystem == "VEHICLE" and e.metadata.get("plate_text") and e.metadata.get("plate_text") != "UNKNOWN":
                identified_vehicle = e.metadata.get("plate_text")

        details = f"{rule_name}: "
        if identified_person and identified_vehicle:
            details += f"Person {identified_person} associated with vehicle {identified_vehicle}."
        elif identified_person:
            details += f"Person {identified_person} associated with unknown vehicle/boundary."
        elif identified_vehicle:
            details += f"Vehicle {identified_vehicle} associated with unknown person/boundary."
        else:
            details += "Unknown person and unknown vehicle/boundary associated."
            
        event_type = "ALERT" if is_alert else "CORRELATION"
        
        meta = {
            "correlation_type": rule_name,
            "confidence": confidence,
            "confidence_level": confidence_level,
            "source_event_ids": [e.id for e in events],
            "time_delta_ms": time_delta_ms
        }
        
        hub.log_event(
            event_type=event_type,
            subsystem="CORRELATION",
            camera_id=camera_id,
            timestamp=time.time(),
            severity="CRITICAL" if is_alert else "INFO",
            details=details,
            confidence=confidence,
            generate_clip=False,
            metadata=meta
        )

_global_correlation_engine = None

def get_correlation_engine() -> CorrelationEngine:
    global _global_correlation_engine
    if _global_correlation_engine is None:
        _global_correlation_engine = CorrelationEngine()
    return _global_correlation_engine
