import os
import cv2
import time
import uuid
import numpy as np
from typing import Dict, List, Optional, Any

from edge.vehicle.schemas import VehicleObservation, VehicleWatchlistEntry, VehicleTestResponse
from edge.vehicle.adapter import VehicleIntelligenceAdapter
from edge.events.event_hub import get_event_hub

class VehicleManager:
    def __init__(self):
        self._adapters: Dict[str, VehicleIntelligenceAdapter] = {}
        self._watchlist: Dict[str, VehicleWatchlistEntry] = {}
        self.event_hub = get_event_hub()
        
        # Debouncing to avoid emitting VEHICLE_DETECTED or VEHICLE_IDENTIFIED every frame
        self._emitted_detection: Dict[str, set] = {}
        self._emitted_identification: Dict[str, set] = {}
        self._emitted_match: Dict[str, set] = {}
        
    def get_or_create_adapter(self, camera_id: str) -> VehicleIntelligenceAdapter:
        if camera_id not in self._adapters:
            self._adapters[camera_id] = VehicleIntelligenceAdapter(camera_id=camera_id)
            self._emitted_detection[camera_id] = set()
            self._emitted_identification[camera_id] = set()
            self._emitted_match[camera_id] = set()
        return self._adapters[camera_id]
        
    def add_to_watchlist(self, entry: VehicleWatchlistEntry):
        self._watchlist[entry.plate_text] = entry
        
    def remove_from_watchlist(self, plate_text: str):
        if plate_text in self._watchlist:
            del self._watchlist[plate_text]
            
    def get_watchlist(self) -> List[VehicleWatchlistEntry]:
        return list(self._watchlist.values())

    def _handle_events(self, camera_id: str, frame: np.ndarray, timestamp: float, obs: VehicleObservation):
        tid = obs.track_id
        
        if tid not in self._emitted_detection[camera_id]:
            self._emitted_detection[camera_id].add(tid)
            self.event_hub.log_event(
                event_type="VEHICLE_DETECTED",
                subsystem="VEHICLE",
                camera_id=camera_id,
                timestamp=timestamp,
                severity="INFO",
                track_id=tid,
                metadata={"vehicle_class": obs.vehicle_class, "bbox": obs.bbox},
                generate_clip=False
            )
            
        if obs.plate_status in ("PROBABLE", "CONFIRMED") and obs.plate_text:
            if tid not in self._emitted_identification[camera_id]:
                self._emitted_identification[camera_id].add(tid)
                severity = "INFO" if obs.plate_status == "PROBABLE" else "WARNING"
                self.event_hub.log_event(
                    event_type="VEHICLE_IDENTIFIED",
                    subsystem="VEHICLE",
                    camera_id=camera_id,
                    timestamp=timestamp,
                    severity=severity,
                    track_id=tid,
                    details=f"Plate {obs.plate_text} identified ({obs.plate_status})",
                    metadata={
                        "vehicle_class": obs.vehicle_class, 
                        "plate_text": obs.plate_text,
                        "plate_status": obs.plate_status,
                        "bbox": obs.bbox
                    },
                    generate_clip=True,
                    pre_seconds=5.0,
                    post_seconds=5.0
                )
            
            if obs.plate_text in self._watchlist:
                if tid not in self._emitted_match[camera_id]:
                    self._emitted_match[camera_id].add(tid)
                    entry = self._watchlist[obs.plate_text]
                    severity = entry.severity if entry.severity in ("INFO", "WARNING", "CRITICAL") else "WARNING"
                    self.event_hub.log_event(
                        event_type="ANPR_MATCH",
                        subsystem="VEHICLE",
                        camera_id=camera_id,
                        timestamp=timestamp,
                        severity=severity,
                        track_id=tid,
                        details=f"Watchlist match for plate {obs.plate_text}: {entry.reason}",
                        metadata={
                            "plate_text": obs.plate_text,
                            "reason": entry.reason,
                            "notes": entry.notes,
                            "bbox": obs.bbox
                        },
                        generate_clip=True,
                        pre_seconds=5.0,
                        post_seconds=5.0
                    )

    def process_frame(self, camera_id: str, frame: np.ndarray, timestamp: float) -> List[VehicleObservation]:
        adapter = self.get_or_create_adapter(camera_id)
        observations = adapter.process_frame(frame, timestamp)
        
        if self.event_hub.clip_manager:
            self.event_hub.clip_manager.add_frame(camera_id, frame, timestamp)
        
        for obs in observations:
            self._handle_events(camera_id, frame, timestamp, obs)
            
        return observations

    def process_uploaded_video(self, video_path: str, camera_id: str, fps_override: Optional[float] = None) -> VehicleTestResponse:
        import cv2
        import time
        from pathlib import Path

        p = Path(video_path)
        if not p.exists():
            return VehicleTestResponse(success=False, vehicles_detected=0, vehicles=[], total_frames=0, duration_seconds=0.0, clips_generated=0, error="Video file not found")

        cap = cv2.VideoCapture(str(p))
        if not cap.isOpened():
            return VehicleTestResponse(success=False, vehicles_detected=0, vehicles=[], total_frames=0, duration_seconds=0.0, clips_generated=0, error="Failed to open video")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if fps_override:
            fps = fps_override
            
        adapter = VehicleIntelligenceAdapter(camera_id=camera_id)
        self._adapters[camera_id] = adapter
        self._emitted_detection[camera_id] = set()
        self._emitted_identification[camera_id] = set()
        self._emitted_match[camera_id] = set()
        
        all_obs = []
        frame_idx = 0
        wall_start = time.time()
        start_time = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_idx += 1
                ts = start_time + (frame_idx / fps)
                
                obs_list = self.process_frame(camera_id, frame, timestamp=ts)
                
                for obs in obs_list:
                    all_obs.append(obs)
                    
        finally:
            cap.release()
            
        time.sleep(1.0)
        
        unique_obs = {}
        for obs in all_obs:
            unique_obs[obs.track_id] = obs
            
        final_vehicles = list(unique_obs.values())
        
        duration = time.time() - wall_start
        
        return VehicleTestResponse(
            success=True,
            vehicles_detected=len(final_vehicles),
            vehicles=final_vehicles,
            total_frames=frame_idx,
            duration_seconds=duration,
            clips_generated=len(self._emitted_identification[camera_id]) + len(self._emitted_match[camera_id]),
            error=None
        )


    def process_uploaded_video_stream(self, video_path: str, camera_id: str, fps_override: Optional[float] = None):
        import cv2, time, json
        from pathlib import Path
        
        p = Path(video_path)
        if not p.exists():
            yield json.dumps({"error": "Video file not found"}) + "\n"
            return
            
        cap = cv2.VideoCapture(str(p))
        if not cap.isOpened():
            yield json.dumps({"error": "Failed to open video"}) + "\n"
            return
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0: total_frames = 1000
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if fps_override: fps = fps_override
        
        adapter = self.get_or_create_adapter(camera_id)
        if camera_id not in self._emitted_identification:
            self._emitted_identification[camera_id] = set()
            
        wall_start = time.time()
        start_time = time.time()
        frame_idx = 0
        all_obs = []
        emitted_in_stream = set()

        try:
            while True:
                ret, frame = cap.read()
                if not ret: break
                
                frame_idx += 1
                ts = start_time + (frame_idx / fps)
                
                obs_list = self.process_frame(camera_id, frame, timestamp=ts)
                logs = []
                
                for obs in obs_list:
                    all_obs.append(obs)
                    is_reliable = obs.plate_text and obs.plate_text != "UNKNOWN" and obs.format_valid and obs.plate_status in ("CONFIRMED", "PROBABLE")
                    if is_reliable:
                        if obs.plate_text not in emitted_in_stream:
                            emitted_in_stream.add(obs.plate_text)
                            time_str = time.strftime("%H:%M:%S", time.gmtime(ts))
                            logs.append({"timestamp": time_str, "type": obs.vehicle_class.capitalize(), "plate": obs.plate_text})
                        
                if frame_idx % 5 == 0 or logs:
                    progress = min(100, int((frame_idx / total_frames) * 100))
                    yield json.dumps({"progress": progress, "logs": logs}) + "\n"
                    
        finally:
            cap.release()
            
        time.sleep(1.0)
        
        unique_obs = {}
        for obs in all_obs:
            if obs.track_id not in unique_obs:
                unique_obs[obs.track_id] = obs
            else:
                existing = unique_obs[obs.track_id]
                is_reliable = obs.plate_text and obs.plate_text != "UNKNOWN" and obs.format_valid and obs.plate_status in ("CONFIRMED", "PROBABLE")
                ex_reliable = existing.plate_text and existing.plate_text != "UNKNOWN" and existing.format_valid and existing.plate_status in ("CONFIRMED", "PROBABLE")
                
                if is_reliable:
                    unique_obs[obs.track_id] = obs
                elif not ex_reliable and (obs.plate_text and obs.plate_text != "UNKNOWN"):
                    unique_obs[obs.track_id] = obs
        final_vehicles = list(unique_obs.values())
        duration = time.time() - wall_start
        
        for v in final_vehicles:
            if v.plate_text and v.plate_text != "UNKNOWN":
                if not (v.format_valid and v.plate_status in ("CONFIRMED", "PROBABLE")):
                    v.plate_text = "UNKNOWN"
                    
        yield json.dumps({
            "progress": 100,
            "results": {
                "success": True,
                "vehicles_detected": len(final_vehicles),
                "total_frames": frame_idx,
                "duration_seconds": duration,
                "clips_generated": len(self._emitted_identification.get(camera_id, set()))
            }
        }) + "\n"

_vehicle_manager = None

def get_vehicle_manager() -> VehicleManager:
    global _vehicle_manager
    if _vehicle_manager is None:
        _vehicle_manager = VehicleManager()
    return _vehicle_manager
