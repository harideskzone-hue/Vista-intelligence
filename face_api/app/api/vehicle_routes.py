import os
import shutil
from fastapi import APIRouter, File, UploadFile, BackgroundTasks, HTTPException, Query
from typing import List, Dict, Any

from edge.vehicle.manager import get_vehicle_manager
from edge.vehicle.schemas import VehicleWatchlistEntry, VehicleTestResponse
from edge.events.event_hub import get_event_hub

vehicle_router = APIRouter(prefix="/api/vehicle", tags=["Vehicle Intelligence"])

# For uploaded video testing
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@vehicle_router.post("/test-video", response_model=VehicleTestResponse)
async def test_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    camera_id: str = Query("CAM_TEST"),
    fps_override: float = Query(None)
):
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
        raise HTTPException(status_code=400, detail="Unsupported file format")

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        manager = get_vehicle_manager()
        # This is synchronous and could block the event loop, 
        # but is fine for the testing endpoints as per previous boundary implementation.
        response = manager.process_uploaded_video(file_path, camera_id=camera_id, fps_override=fps_override)
        return response
    finally:
        # Clean up file in background
        def cleanup(path):
            if os.path.exists(path):
                os.remove(path)
        background_tasks.add_task(cleanup, file_path)


from fastapi.responses import StreamingResponse
import json

@vehicle_router.post("/test-video-stream")
async def test_video_stream(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    camera_id: str = Query("CAM_TEST"),
    fps_override: float = Query(None)
):
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
        raise HTTPException(status_code=400, detail="Unsupported file format")

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        import shutil
        shutil.copyfileobj(file.file, buffer)

    manager = get_vehicle_manager()
    
    def event_stream():
        try:
            for chunk in manager.process_uploaded_video_stream(file_path, camera_id=camera_id, fps_override=fps_override):
                yield chunk
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")

@vehicle_router.get("/events")
async def get_vehicle_events(
    limit: int = 50,
    camera_id: str = None,
    event_type: str = None,
    start_time: float = None,
    end_time: float = None,
    search_query: str = None
):
    event_hub = get_event_hub()
    events = event_hub.list_events(
        subsystem="VEHICLE",
        camera_id=camera_id,
        event_type=event_type,
        start_time=start_time,
        end_time=end_time,
        search_query=search_query,
        limit=limit
    )
    
    formatted = []
    for e in events:
        d = e.to_dict()
        formatted.append(d)
        
    return formatted

@vehicle_router.get("/watchlist")
async def get_watchlist():
    manager = get_vehicle_manager()
    return manager.get_watchlist()

@vehicle_router.post("/watchlist")
async def add_to_watchlist(entry: VehicleWatchlistEntry):
    manager = get_vehicle_manager()
    manager.add_to_watchlist(entry)
    return {"status": "success", "plate": entry.plate_text}

@vehicle_router.delete("/watchlist/{plate_text}")
async def remove_from_watchlist(plate_text: str):
    manager = get_vehicle_manager()
    manager.remove_from_watchlist(plate_text)
    return {"status": "success", "removed": plate_text}

@vehicle_router.get("/cameras/{camera_id}/status")
async def get_camera_status(camera_id: str):
    manager = get_vehicle_manager()
    # If the adapter exists, we can get some pipeline metrics
    if camera_id in manager._adapters:
        adapter = manager._adapters[camera_id]
        return {
            "camera_id": camera_id,
            "status": "active",
            "frames_processed": adapter.pipeline.frames_processed,
            "metrics": adapter.pipeline.metrics.dict() if hasattr(adapter.pipeline.metrics, "dict") else adapter.pipeline.metrics.model_dump()
        }
    return {"camera_id": camera_id, "status": "inactive"}

@vehicle_router.get("/watchlist")
async def get_watchlist():
    manager = get_vehicle_manager()
    return manager.get_watchlist()

@vehicle_router.post("/watchlist")
async def add_to_watchlist(entry: VehicleWatchlistEntry):
    manager = get_vehicle_manager()
    manager.add_to_watchlist(entry)
    return {"status": "success", "plate": entry.plate_text}
