"""
Temporal Confirmation: Maintains per-camera, per-track identity state.
Prevents identity oscillation and requires sustained evidence to confirm or switch identities.
State is kept in memory only and is cleared on process restart.
"""
from __future__ import annotations

import collections

import time

class TemporalState:
    """Tracks history and confirmed identity for a single face track."""
    
    def __init__(self, confirmation_frames: int, switch_frames: int):
        self.confirmation_frames = confirmation_frames
        self.switch_frames = switch_frames
        
        self.max_history = max(confirmation_frames, switch_frames)
        self.history: collections.deque[str | None] = collections.deque(maxlen=self.max_history)
        
        self.state = "UNKNOWN"
        self.confirmed_identity: str | None = None

    def update(self, candidate_id: str | None, recently_seen: bool) -> tuple[str, str | None]:
        """
        Process a new candidate observation and return the temporally confirmed identity.
        Returns: (state_name, confirmed_identity)
        """
        self.history.append(candidate_id)

        if self.state == "UNKNOWN":
            if candidate_id is not None:
                if recently_seen:
                    self.confirmed_identity = candidate_id
                    self.state = "CONFIRMED"
                else:
                    self.state = "CANDIDATE"
                    if self.confirmation_frames <= 1:
                        self.confirmed_identity = candidate_id
                        self.state = "CONFIRMED"
        
        elif self.state == "CANDIDATE":
            if candidate_id is not None:
                recent = list(self.history)[-self.confirmation_frames:]
                if len(recent) >= self.confirmation_frames and all(x == candidate_id for x in recent):
                    self.confirmed_identity = candidate_id
                    self.state = "CONFIRMED"
            else:
                self.state = "UNKNOWN"
                self.history.clear()
                
        elif self.state == "CONFIRMED":
            self.state = "LOCKED"
            # Fall through to LOCKED logic
            
        if self.state == "LOCKED":
            if candidate_id is not None and candidate_id != self.confirmed_identity:
                recent = list(self.history)[-self.switch_frames:]
                if len(recent) >= self.switch_frames and all(x == candidate_id for x in recent):
                    self.confirmed_identity = candidate_id
                    self.state = "CONFIRMED"

        if self.state in ("CONFIRMED", "LOCKED"):
            return self.state, self.confirmed_identity
        else:
            return self.state, None


class TemporalMatcher:
    """
    Manages temporal confirmation state across all active cameras and tracks.
    State is ephemeral (in-memory only).
    """
    
    def __init__(self, confirmation_frames: int = 3, switch_frames: int = 5, occlusion_memory_seconds: float = 15.0):
        self.confirmation_frames = confirmation_frames
        self.switch_frames = switch_frames
        self.occlusion_memory_seconds = occlusion_memory_seconds
        
        # (camera_id, track_id) -> TemporalState
        self.tracks: dict[tuple[str, str], TemporalState] = {}
        
        # (camera_id, person_id) -> last_seen_time (float)
        # Decouples person_id from track_id for robust identity stability
        self.recently_seen_identities: dict[tuple[str, str], float] = {}

    def update(self, camera_id: str, track_id: str, candidate_id: str | None) -> str | None:
        """Update track history and get current confirmed identity."""
        key = (camera_id, track_id)
        if key not in self.tracks:
            self.tracks[key] = TemporalState(self.confirmation_frames, self.switch_frames)
            
        now = time.time()
        
        # Check if candidate was recently seen on this camera
        recently_seen = False
        if candidate_id is not None:
            mem_key = (camera_id, candidate_id)
            last_seen = self.recently_seen_identities.get(mem_key, 0)
            if now - last_seen <= self.occlusion_memory_seconds:
                recently_seen = True
                
        state_name, confirmed = self.tracks[key].update(candidate_id, recently_seen)
        
        # Update the decoupled identity memory
        if confirmed is not None:
            self.recently_seen_identities[(camera_id, confirmed)] = now
            
        return state_name, confirmed

    def remove_track(self, camera_id: str, track_id: str) -> None:
        """Clear state when a track finishes."""
        key = (camera_id, track_id)
        self.tracks.pop(key, None)

    def clear_all(self) -> None:
        """Clear all temporal state (e.g., on restart/reconnect)."""
        self.tracks.clear()
        self.recently_seen_identities.clear()
