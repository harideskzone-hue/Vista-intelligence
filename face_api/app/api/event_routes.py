"""
Event Routes — Secure REST endpoints for multi-modal events and video clip streaming.

Security Guarantees:
1. No filesystem paths exposed to the frontend.
2. Direct resolution of clips through the event database record.
3. Strict path traversal verification (must reside inside allowed clips directory).
4. Range-compatible MP4 streaming for HTML5 video players.
"""

import os
import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse

from edge.events.event_hub import get_event_hub

log = logging.getLogger("event_routes")
router = APIRouter(prefix="/api/events", tags=["events"])


@router.get("")
async def list_events(
    subsystem: Optional[str] = Query(None, description="Subsystem filter: BOUNDARY, ANOMALY, VEHICLE"),
    camera_id: Optional[str] = Query(None, description="Camera ID filter"),
    event_type: Optional[str] = Query(None, description="Event type filter"),
    limit: int = Query(50, ge=1, le=100, description="Max events to return")
):
    """Lists recent system events without exposing internal server paths."""
    hub = get_event_hub()
    events = hub.list_events(subsystem=subsystem, camera_id=camera_id, event_type=event_type, limit=limit)
    return JSONResponse([ev.to_dict(include_internal_paths=False) for ev in events])


@router.get("/live_stream")
async def get_live_events():
    """Returns recent in-memory events for fast dashboard polling."""
    hub = get_event_hub()
    events = hub.get_live_events()
    return JSONResponse([ev.to_dict(include_internal_paths=False) for ev in events])


@router.get("/{event_id}")
async def get_event_details(event_id: str):
    """Returns single event details by UUID without internal disk paths."""
    hub = get_event_hub()
    ev = hub.get_event(event_id)
    if not ev:
        raise HTTPException(status_code=404, detail="Event not found")
    return JSONResponse(ev.to_dict(include_internal_paths=False))


@router.get("/{event_id}/clip")
async def stream_event_clip(event_id: str):
    """
    Securely streams the generated MP4 event clip.
    Resolves the clip through the database record and enforces directory boundary checks.
    """
    hub = get_event_hub()
    ev = hub.get_event(event_id)
    if not ev:
        raise HTTPException(status_code=404, detail="Event not found")

    if ev.clip_status == "PENDING":
        raise HTTPException(status_code=202, detail="Event clip is currently being compiled")

    if ev.clip_status != "READY" or not ev.clip_path:
        raise HTTPException(status_code=404, detail="No video clip available for this event")

    clip_path = os.path.abspath(ev.clip_path)
    allowed_dir = os.path.abspath(hub.clip_manager.output_dir)

    # Path traversal defense: ensure file is strictly inside the designated clips directory
    if not clip_path.startswith(allowed_dir):
        log.error(f"Security Alert: Path traversal attempt blocked for event {event_id}: {clip_path}")
        raise HTTPException(status_code=403, detail="Forbidden file access")

    if not os.path.exists(clip_path) or os.path.getsize(clip_path) == 0:
        raise HTTPException(status_code=404, detail="Event clip file not found on disk")

    return FileResponse(
        clip_path,
        media_type="video/mp4",
        filename=f"event_{event_id}.mp4",
        headers={"Accept-Ranges": "bytes"}
    )


@router.get("/{event_id}/snapshot")
async def get_event_snapshot(event_id: str):
    """Securely streams the event frame snapshot."""
    hub = get_event_hub()
    ev = hub.get_event(event_id)
    if not ev or not ev.snapshot_path:
        raise HTTPException(status_code=404, detail="Snapshot not found")

    snap_path = os.path.abspath(ev.snapshot_path)
    if not os.path.exists(snap_path):
        raise HTTPException(status_code=404, detail="Snapshot file not found on disk")

    return FileResponse(snap_path, media_type="image/jpeg")
