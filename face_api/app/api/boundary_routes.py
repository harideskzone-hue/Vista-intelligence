"""
Boundary API Routes — RESTful Endpoints for Boundary Line, Zone Configuration & Testing.

Features:
- Configures normalized [0.0, 1.0] Canvas boundary lines and polygon zones.
- Configures rule-based anomaly indicator thresholds.
- Enforces strict parameter validation (coordinates bounded within [0.0, 1.0]).
- Queries boundary crossing and anomaly events with zero filesystem path exposure.
- Processes uploaded video files with real YOLO multi-object tracking and exact event clip pipeline.
- Returns semantically separated metrics: Line Crossings vs Zone Transitions vs Rule-based Anomalies.
"""

import os
import shutil
import tempfile
import logging
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, Query, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from edge.boundary.manager import get_boundary_manager
from edge.events.event_hub import get_event_hub

log = logging.getLogger("boundary_routes")
router = APIRouter(prefix="/api/boundary", tags=["boundary"])


class LineCoordinates(BaseModel):
    x1: float = Field(..., ge=0.0, le=1.0, description="Normalized start X in [0.0, 1.0]")
    y1: float = Field(..., ge=0.0, le=1.0, description="Normalized start Y in [0.0, 1.0]")
    x2: float = Field(..., ge=0.0, le=1.0, description="Normalized end X in [0.0, 1.0]")
    y2: float = Field(..., ge=0.0, le=1.0, description="Normalized end Y in [0.0, 1.0]")


class BoundaryLineConfigRequest(BaseModel):
    camera_id: str = Field(..., min_length=1, description="Target camera identifier")
    line: LineCoordinates
    restricted_side: str = Field("A", description="Restricted side: 'A' (+1) or 'B' (-1)")
    line_id: Optional[str] = Field("main", description="Identifier for the line boundary")


class ZoneConfigRequest(BaseModel):
    camera_id: str = Field(..., min_length=1, description="Target camera identifier")
    zone_id: str = Field(..., min_length=1, description="Identifier for the polygon zone")
    polygon: List[List[float]] = Field(..., min_length=3, description="List of [x, y] coordinates in [0.0, 1.0]")


class AnomalyThresholdsRequest(BaseModel):
    camera_id: Optional[str] = Field(None, description="Camera ID or None for system default")
    loitering_time: Optional[float] = Field(None, ge=1.0, le=120.0)
    loitering_radius: Optional[float] = Field(None, ge=10.0, le=500.0)
    zone_lingering_time: Optional[float] = Field(None, ge=1.0, le=120.0)
    running_speed_thresh: Optional[float] = Field(None, ge=50.0, le=1000.0)
    crawling_ratio_thresh: Optional[float] = Field(None, ge=0.5, le=3.0)
    crawling_min_frames: Optional[int] = Field(None, ge=2, le=30)
    erratic_angle_thresh: Optional[float] = Field(None, ge=30.0, le=150.0)
    erratic_min_turns: Optional[int] = Field(None, ge=2, le=10)
    group_min_size: Optional[int] = Field(None, ge=2, le=20)
    group_distance_px: Optional[float] = Field(None, ge=20.0, le=500.0)
    alert_cooldown_sec: Optional[float] = Field(None, ge=1.0, le=60.0)


@router.get("/config")
async def get_boundary_config(camera_id: Optional[str] = Query(None, description="Camera ID")):
    """Returns boundary configuration for a specific camera or all cameras."""
    bm = get_boundary_manager()
    if camera_id:
        return JSONResponse(bm.get_config(camera_id))
    return JSONResponse(bm.list_configs())


@router.post("/config")
async def set_boundary_config(req: BoundaryLineConfigRequest):
    """Saves or updates normalized boundary line configuration for a camera."""
    side = req.restricted_side.upper()
    if side not in ("A", "B"):
        raise HTTPException(status_code=400, detail="restricted_side must be either 'A' or 'B'")

    bm = get_boundary_manager()
    updated = bm.set_boundary_line(
        camera_id=req.camera_id,
        x1=req.line.x1,
        y1=req.line.y1,
        x2=req.line.x2,
        y2=req.line.y2,
        restricted_side=side,
        line_id=req.line_id or "main"
    )
    return JSONResponse({"status": "success", "config": updated})


@router.post("/zones")
async def set_boundary_zone(req: ZoneConfigRequest):
    """Adds or updates a polygon restricted zone for a camera."""
    for pt in req.polygon:
        if len(pt) < 2 or not (0.0 <= pt[0] <= 1.0 and 0.0 <= pt[1] <= 1.0):
            raise HTTPException(status_code=400, detail="All polygon points must be [x, y] between 0.0 and 1.0")

    bm = get_boundary_manager()
    updated = bm.set_zone(
        camera_id=req.camera_id,
        zone_id=req.zone_id,
        polygon=req.polygon
    )
    return JSONResponse({"status": "success", "config": updated})


@router.delete("/zones/{camera_id}/{zone_id}")
async def delete_boundary_zone(camera_id: str, zone_id: str):
    """Deletes a polygon restricted zone."""
    bm = get_boundary_manager()
    success = bm.delete_zone(camera_id, zone_id)
    if not success:
        raise HTTPException(status_code=404, detail="Zone or camera configuration not found")
    return JSONResponse({"status": "success", "message": f"Zone {zone_id} deleted"})


@router.get("/anomalies/config")
async def get_anomaly_config(camera_id: Optional[str] = Query(None, description="Camera ID")):
    """Returns rule-based anomaly indicator thresholds."""
    bm = get_boundary_manager()
    thresholds = bm.get_anomaly_thresholds(camera_id)
    return JSONResponse({
        "camera_id": camera_id or "default",
        "thresholds": thresholds,
        "classification": "Rule-based anomaly indicators"
    })


@router.post("/anomalies/config")
async def set_anomaly_config(req: AnomalyThresholdsRequest):
    """Updates configurable rule-based anomaly indicator thresholds."""
    bm = get_boundary_manager()
    # Extract non-None fields
    dump_fn = getattr(req, 'model_dump', getattr(req, 'dict', None))
    updates = {k: v for k, v in dump_fn().items() if v is not None and k != "camera_id"}
    saved = bm.set_anomaly_thresholds(req.camera_id, updates)
    return JSONResponse({
        "status": "success",
        "camera_id": req.camera_id or "default",
        "thresholds": saved,
        "classification": "Rule-based anomaly indicators"
    })


@router.get("/events")
async def list_boundary_events(
    camera_id: Optional[str] = Query(None, description="Filter by camera ID"),
    event_type: Optional[str] = Query(None, description="Filter by event type (e.g. CROSSING, LOITERING, RUNNING)"),
    limit: int = Query(50, ge=1, le=100, description="Limit of records to retrieve")
):
    """Lists boundary events with zero filesystem path exposure."""
    hub = get_event_hub()
    events = hub.list_events(
        subsystem="BOUNDARY",
        camera_id=camera_id,
        event_type=event_type,
        limit=limit
    )
    return JSONResponse([ev.to_dict(include_internal_paths=False) for ev in events])


@router.post("/test-video")
async def test_uploaded_video(
    file: UploadFile = File(...),
    camera_id: Optional[str] = Form(None),
    x1: float = Form(0.2),
    y1: float = Form(0.5),
    x2: float = Form(0.8),
    y2: float = Form(0.5),
    restricted_side: str = Form("A"),
    touch_tolerance: float = Form(0.03)
):
    """
    Executes boundary crossing and anomaly detection on an uploaded video file.
    Uses real YOLO + ByteTrack multi-object tracking to detect and follow subjects.
    Returns semantically separated metrics for Line Crossings, Zone Events, and Anomalies.
    """
    allowed_exts = (".mp4", ".avi", ".mov", ".mkv")
    filename = file.filename or "uploaded_video.mp4"
    ext = os.path.splitext(filename)[1].lower()
    if ext not in allowed_exts:
        raise HTTPException(status_code=400, detail=f"Unsupported video extension. Allowed: {allowed_exts}")

    bm = get_boundary_manager()
    temp_dir = tempfile.mkdtemp(prefix="boundary_test_")
    temp_video_path = os.path.join(temp_dir, filename)

    try:
        with open(temp_video_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        def event_generator():
            try:
                for chunk in bm.process_uploaded_video_stream(
                    video_path=temp_video_path,
                    camera_id=camera_id or f"test_video_{os.path.basename(filename)}",
                    line={"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    restricted_side=restricted_side,
                    touch_tolerance=touch_tolerance
                ):
                    yield chunk
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)
                
        from fastapi.responses import StreamingResponse
        return StreamingResponse(event_generator(), media_type="application/x-ndjson")

    except Exception as e:
        log.error(f"Error starting video stream: {e}", exc_info=True)
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(e))
