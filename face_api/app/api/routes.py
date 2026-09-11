"""
FastAPI Routes
New API structure: /search, /add, /name, /topn
Refactored to use FaceService.
"""
from typing import List, Optional
from fastapi import APIRouter, UploadFile, Form, HTTPException, Body, Query, Request  # type: ignore
from fastapi.responses import JSONResponse, Response  # type: ignore
from pydantic import BaseModel  # type: ignore
import os, json, uuid, tempfile, time, threading

from app.services.face_service import FaceService  # type: ignore

api = APIRouter(prefix="/api")
face_service = FaceService()

# ── Shared cameras.json path ───────────────────────────────────────────────────
# Stored at project root so both face_api and face_engine can read/write it.
_CAMERAS_JSON = os.environ.get(
    "CAMERAS_JSON_PATH",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "cameras.json")
)
MAX_CAMERAS = 4
_cam_lock = threading.Lock()

# ── Remote Camera IPC State ────────────────────────────────────────────────────
_CAMERA_STATE_FILE = os.path.join(
    os.environ.get("SIH26187_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "data")),
    "camera_state.json"
)
_camera_state_lock = threading.Lock()

def _get_camera_state():
    if not os.path.exists(_CAMERA_STATE_FILE):
        return {"active": False}
    try:
        with open(_CAMERA_STATE_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {"active": False}

def _set_camera_state(active: bool):
    with _camera_state_lock:
        os.makedirs(os.path.dirname(_CAMERA_STATE_FILE), exist_ok=True)
        # Atomic-like write for cross-process safety
        tmp = _CAMERA_STATE_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"active": active, "updated_at": time.time()}, f)
        os.replace(tmp, _CAMERA_STATE_FILE)
        return {"active": active}

# Pydantic Models
class BatchAddImage(BaseModel):
    img: str
    score: float = 0.0
    name: Optional[str] = None
    cropped: bool = False

class BatchAddRequest(BaseModel):
    images: List[BatchAddImage]

class CreatePersonRequest(BaseModel):
    name: str

class SearchRequest(BaseModel):
    img: Optional[str] = None
    cropped: bool = False

# Endpoints

@api.get('/health')
async def health():
    """Health check endpoint."""
    return face_service.get_health_stats()

@api.post('/system/cleanup')
async def system_cleanup(request: Request):
    """Manually trigger the storage cleanup."""
    try:
        raw_body = await request.body()
        body = await request.json() if raw_body else {}
    except Exception:
        body = {}
    retention_days = body.get('retention_days', None) if isinstance(body, dict) else None
    result = face_service.run_storage_cleanup(retention_days)
    if result.get('success'):
        return JSONResponse(result)
    else:
        return JSONResponse(result, status_code=500)

# ── Debug / Observability ─────────────────────────────────────────────────────
_last_upload_info: dict = {}

@api.get('/debug/last_upload')
async def debug_last_upload():
    """Return metadata about the most recent upload for observability."""
    if not _last_upload_info:
        return JSONResponse({"status": "no_uploads_yet"})
    return JSONResponse(_last_upload_info)

@api.get('/camera/status')
async def camera_status():
    """Return the intended hardware state of the camera and the background processing queue size."""
    import typing
    state: dict[str, typing.Any] = _get_camera_state()  # type: ignore
    try:
        import sqlite3
        db_path = os.path.join(
            os.environ.get("SIH26187_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "data")),
            "captured_videos", "jobs.db"
        )
        if os.path.exists(db_path):
            with sqlite3.connect(db_path, timeout=2.0) as conn:
                state["pending_jobs"] = conn.execute("SELECT COUNT(*) FROM jobs WHERE status = 'PENDING'").fetchone()[0]
        else:
            state["pending_jobs"] = 0
    except Exception as e:
        state["pending_jobs"] = 0
    return JSONResponse(state)

@api.post('/camera/start')
async def camera_start():
    """Request the scorer daemon to allocate hardware and begin."""
    return JSONResponse(_set_camera_state(True))

@api.post('/camera/stop')
async def camera_stop():
    """Request the scorer daemon to release hardware and enter standby."""
    return JSONResponse(_set_camera_state(False))

@api.post('/search')
async def search_post(request: SearchRequest):
    """
    Search by face image (POST).
    """
    result = face_service.search_image(request.img, request.cropped)
    status_code = 200 if result.get('success', False) else 400
    if not result.get('success') and result.get('match') is False:
         status_code = 200 # No match is a valid result
    
    # Handle specific error cases that were returned as 500 in original code if appropriate, 
    # but 400 is generally safer for client errors. 
    # Original code returned 500 for 'Person data not found' logic error.
    if result.get('error') == 'Person data not found':
        status_code = 500
        
    return JSONResponse(result, status_code=status_code)

@api.get('/search')
async def search_get(
    id: Optional[str] = Query(None),
    name: Optional[str] = Query(None)
):
    """
    Search by ID or name (GET).
    """
    if not id and not name:
        return JSONResponse({'success': False, 'error': 'No search parameter provided (id or name)'}, status_code=400)
    
    result = face_service.get_person_by_id_or_name(id, name)
    status_code = 200 if result.get('success') else 404
    return JSONResponse(result, status_code=status_code)

@api.post('/enroll')
async def enroll_person(request: Request):
    """
    Explicitly enroll a person by name and image(s).
    Expects multipart/form-data with 'images' (or 'image'), 'name', 'role', 'classification'.
    """
    try:
        form = await request.form()
        name = str(form.get('name', 'Unknown Person'))
        role = str(form.get('role', 'STAFF'))
        classification = str(form.get('classification', 'normal'))
        
        # Support both 'image' and 'images' keys
        upload_files = form.getlist('images')
        if not upload_files:
            upload_files = form.getlist('image')
            
        images_data = []
        for file in upload_files:
            if hasattr(file, 'read'):
                data = await file.read()
                if data:
                    images_data.append(data)
                    
        if not images_data:
            return JSONResponse({'success': False, 'error': 'No image provided'}, status_code=400)
            
        # We use enroll_person_bulk for both single and multiple
        result = face_service.enroll_person_bulk(name, role, images_data, classification=classification)
        
        status_code = 200 if result['success'] else 400
        return JSONResponse(result, status_code=status_code)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({'success': False, 'error': str(e)}, status_code=400)



import collections
import base64

_live_events = collections.deque(maxlen=50)
_unknown_throttles = {}
_person_alert_throttles = {}  # (camera_id, person_id) -> last_alert_time

@api.post('/recognize_live')
async def recognize_live(request: Request):
    """
    Live frame recognition endpoint.
    Expects multipart/form-data with 'image', 'camera_id', and 'track_id'.
    """
    try:
        form = await request.form()
        camera_id = str(form.get('camera_id', 'unknown_cam'))
        track_id = str(form.get('track_id', 'unknown_track'))
        enhancement_mode = str(form.get('enhancement_mode', 'AUTO'))
        
        upload_file = form.get('image')
        if upload_file is not None and hasattr(upload_file, 'read'):
            img_data = await upload_file.read()
        else:
            return JSONResponse({'success': False, 'error': 'No image provided'}, status_code=400)
            
        result = face_service.recognize_live_frame(img_data, camera_id, track_id, enhancement_mode=enhancement_mode)
        
        # Log event for dashboard
        if result.get('success'):
            status = result.get('match_status')
            action = result.get('action')
            person_id = result.get('matched_identity_id')
            
            should_log = False
            img_b64 = None
            now = time.time()
            
            if status == 'MATCH' and person_id and result.get('temporal_state') in ('CONFIRMED', 'LOCKED'):
                alert_key = (camera_id, person_id)
                last_alert = _person_alert_throttles.get(alert_key, 0)
                
                # We log if this is a brand new event (>15s since last seen) OR if we captured better evidence
                if action == 'saved_evidence':
                    should_log = True
                    img_b64 = base64.b64encode(img_data).decode('utf-8')
                    _person_alert_throttles[alert_key] = now
                elif now - last_alert > 15:
                    should_log = True
                    _person_alert_throttles[alert_key] = now
                    # Try to fetch the best frame for this person since we are alerting but didn't save this frame
                    p_info = face_service.get_person_by_id_or_name(person_id)
                    if p_info.get('success') and p_info['person'].get('representative_image_id'):
                        best_bytes = face_service.get_image_bytes(p_info['person']['representative_image_id'])
                        if best_bytes:
                            img_b64 = base64.b64encode(best_bytes).decode('utf-8')
                    
                    if not img_b64:
                        img_b64 = base64.b64encode(img_data).decode('utf-8') # Fallback to current
            else:
                throttle_key = f"{camera_id}_{track_id}"
                if now - _unknown_throttles.get(throttle_key, 0) > 30:
                    _unknown_throttles[throttle_key] = now
                    should_log = True
                    
            if should_log:
                _live_events.appendleft({
                    'timestamp': now,
                    'camera_id': camera_id,
                    'match_status': status,
                    'temporal_state': result.get('temporal_state', 'UNKNOWN'),
                    'matched_identity_id': person_id,
                    'matched_identity_name': result.get('matched_identity_name'),
                    'action': action,
                    'embedding_distance': result.get('embedding_distance'),
                    'image': img_b64
                })
            
        status_code = 200 if result['success'] else 400
        return JSONResponse(result, status_code=status_code)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({'success': False, 'error': str(e)}, status_code=400)

@api.get('/events/live')
async def get_live_events(n: int = Query(12)):
    """Return the N most recent live recognition events."""
    events = list(_live_events)[:n]
    return JSONResponse({'success': True, 'events': events})

@api.post('/batch_add')
async def batch_add_images(request: BatchAddRequest):
    """
    Add multiple images in batch.
    """
    images = request.images
    
    if not images:
        return JSONResponse({'success': False, 'error': 'List of images required'}, status_code=400)
    
    results = []
    success_count: int = 0
    fail_count: int = 0
    
    print(f"DEBUG: Batch processing {len(images)} images...")
    
    for i, img_data in enumerate(images):
        result = face_service.process_and_add_image(img_data.img, img_data.score, img_data.cropped)
        result['index'] = i
        
        if result['success']:
            name = img_data.name
            # If name provided, try to assign it. This logic is slightly coupled 
            # but fits 'process_and_add' extension or calling assign_name separately.
            if name and result.get('person_id'):
                face_service.assign_name(result['person_id'], name)
                result['name_assigned'] = name
            success_count = success_count + 1  # type: ignore
        else:
            print(f"DEBUG: Batch item {i} failed: {result.get('error')}")
            fail_count = fail_count + 1  # type: ignore
        
        results.append(result)
        
    return {
        'success': True,
        'total': len(images),
        'added': success_count,
        'failed': fail_count,
        'results': results
    }

@api.delete('/reset')
async def reset_database():
    """Clear all data from the database."""
    # This involves multiple services (storage, vector_db). 
    # For now, it's safer to keep this special administrative action here 
    # or add a 'reset_all' to service. 
    # Let's add it to service for consistency in next step or use dependency access here?
    # Better: Use dependencies directly as this is admin function, OR strictly use service.
    # Service doesn't have reset() yet. 
    # Implementation detail: I'll use the service deps pattern or imports. 
    # To follow Clean Code: add reset to service.
    try:
        from app.core.storage import get_storage  # type: ignore
        from app.core.vector_db import get_vector_db  # type: ignore
        storage = get_storage()
        vector_db = get_vector_db()
        storage.reset()
        vector_db.reset()
        
        return {
            'success': True,
            'message': 'Database successfully cleared'
        }
    except Exception as e:
        return JSONResponse({'success': False, 'error': str(e)}, status_code=500)

@api.post('/name')
async def assign_name(request: Request):
    """Assign a name to a person."""
    try:
        content_type = request.headers.get('content-type', '')
        final_pid = None
        final_name = None
        
        if 'application/json' in content_type:
            data = await request.json()
            final_pid = data.get('person_id')
            final_name = data.get('name')
        elif 'multipart/form-data' in content_type or 'application/x-www-form-urlencoded' in content_type:
            form = await request.form()
            final_pid = form.get('person_id')
            final_name = form.get('name')
    except Exception:
        return JSONResponse({'success': False, 'error': 'Invalid request body'}, status_code=400)
    
    if not final_pid or not final_name:
        return JSONResponse({'success': False, 'error': 'person_id and name required'}, status_code=400)
    
    result = face_service.assign_name(final_pid, final_name)
    status_code = 200 if result['success'] else 400
    if result.get('error') == 'Person not found':
        status_code = 404
        
    return JSONResponse(result, status_code=status_code)

@api.get('/topn')
async def get_top_images():
    """Get top 3 images globally by score."""
    return face_service.get_top_images(n=3)

@api.get('/recent')
async def get_recent_images(n: int = Query(12)):
    """Get N most recent images globally."""
    # We add a new method to FaceService for this
    return face_service.get_recent_images(n=n)

@api.get('/person/{person_id}')
async def get_person(person_id: str):
    """Get person info by ID."""
    result = face_service.get_person_by_id_or_name(pid=person_id, name=None)
    
    if result.get('success'):
         return {
             'success': True,
             'person': result['person']
         }
    else:
        return JSONResponse(result, status_code=404)


@api.get('/persons')
async def get_all_persons(classification: Optional[str] = Query(None)):
    result = face_service.get_all_persons_edge()
    if not result['success']:
        return JSONResponse(result, status_code=500)
    
    persons = result['persons']
    if classification:
        persons = [p for p in persons if p.get('classification') == classification]
        
    return JSONResponse({'success': True, 'persons': persons})

class PersonUpdate(BaseModel):
    display_name: Optional[str] = None
    category: Optional[str] = None
    classification: Optional[str] = None
    status: Optional[str] = None

@api.post('/persons/{person_id}/images')
async def add_reference_images(person_id: str, request: Request):
    """Add reference images to an existing person."""
    try:
        form = await request.form()
        upload_files = form.getlist('images')
        if not upload_files:
            upload_files = form.getlist('image')
            
        images_data = []
        for file in upload_files:
            if hasattr(file, 'read'):
                data = await file.read()
                if data:
                    images_data.append(data)
                    
        if not images_data:
            return JSONResponse({'success': False, 'error': 'No image provided'}, status_code=400)
            
        result = face_service.add_reference_images_to_person(person_id, images_data)
        status_code = 200 if result['success'] else 400
        return JSONResponse(result, status_code=status_code)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({'success': False, 'error': str(e)}, status_code=400)

@api.put('/persons/{person_id}')
async def update_person(person_id: str, update: PersonUpdate):
    updates = {k: v for k, v in update.dict(exclude_unset=True).items()}
    result = face_service.update_person_edge(person_id, updates)
    status_code = 200 if result['success'] else 400
    return JSONResponse(result, status_code=status_code)

@api.delete('/persons/{person_id}')

async def delete_person(person_id: str):
    """Delete a person and all their images."""
    result = face_service.delete_person(person_id)
    status_code = 200 if result['success'] else 404
    return JSONResponse(result, status_code=status_code)

@api.delete('/image/{image_id}')
async def delete_image(image_id: str):
    """Delete a reference image and its associated embedding."""
    result = face_service.delete_reference_image(image_id)
    status_code = 200 if result['success'] else 400
    return JSONResponse(result, status_code=status_code)

@api.get('/image/{image_id}')
async def get_image(image_id: str):
    """Serve an image file by ID (raw JPEG bytes)."""
    image_bytes = face_service.get_image_bytes(image_id)
    
    if not image_bytes:
        return JSONResponse({'success': False, 'error': 'Image not found'}, status_code=404)
    
    return Response(image_bytes, media_type='image/jpeg')


# ══════════════════════════════════════════════════════════════════════════════
# Camera Management API
# ══════════════════════════════════════════════════════════════════════════════

class CameraEntry(BaseModel):
    id: str | None = None
    source: str           # "0", "1", "rtsp://...", etc.
    label: str = ""
    enabled: bool = True

class CameraUpdate(BaseModel):
    source: Optional[str] = None
    label: Optional[str] = None
    enabled: Optional[bool] = None


def _read_cameras() -> list:
    """Read cameras.json safely. Returns [] on any error."""
    with _cam_lock:
        try:
            with open(_CAMERAS_JSON, 'r') as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except Exception:
            return []


def _write_cameras(cameras: list) -> None:
    """
    Write cameras.json atomically using write-to-temp-then-rename.
    This prevents race conditions where live_scorer reads a half-written file.
    """
    dir_ = os.path.dirname(os.path.abspath(_CAMERAS_JSON))
    with _cam_lock:
        fd, tmp_path = tempfile.mkstemp(dir=dir_, suffix='.tmp')
        try:
            with os.fdopen(fd, 'w') as f:
                json.dump(cameras, f, indent=2)
            os.replace(tmp_path, _CAMERAS_JSON)   # atomic on POSIX
        except Exception:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
            raise


def _validate_source(source: str) -> tuple[bool, str]:
    """
    Quick validation of a camera source without keeping the stream open.
    Tries to open the stream, reads one frame, then releases immediately.
    Returns (ok, error_message).
    """
    try:
        import cv2  # type: ignore
        src = int(source) if source.strip().lstrip('-').isdigit() else source.strip()
        cap = cv2.VideoCapture(src)
        opened = cap.isOpened()
        if opened:
            ret, _ = cap.read()
            cap.release()
            if not ret:
                return False, f"Camera opened but could not read frame from '{source}'"
            return True, ""
        else:
            cap.release()
            return False, f"Could not open camera source '{source}'"
    except ImportError:
        return False, "opencv-python (cv2) not installed on server"
    except Exception as e:
        return False, str(e)


import re

def _mask_source(source: str) -> str:
    # Match rtsp://username:password@IP...
    # Replace username:password@ with ***:***@
    return re.sub(r'rtsp://[^:]+:[^@]+@', 'rtsp://***:***@', source)

@api.get('/cameras')
async def list_cameras():
    """Return all configured cameras, masking sensitive credentials."""
    cameras = _read_cameras()
    for c in cameras:
        c['source'] = _mask_source(str(c['source']))
    return JSONResponse({'success': True, 'cameras': cameras})


@api.post('/cameras')
async def add_camera(entry: CameraEntry):
    """Add a new camera. Validates source before saving. Max 4 cameras."""
    cameras = _read_cameras()
    if len([c for c in cameras if c.get('enabled')]) >= MAX_CAMERAS:
        return JSONResponse({'success': False,
                             'error': f'Maximum of {MAX_CAMERAS} cameras supported'},
                            status_code=400)

    ok, err = _validate_source(entry.source)
    if not ok:
        return JSONResponse(
            {'success': False, 'error': f'Source validation failed: {err}', 'source': entry.source},
            status_code=400
        )

    # Using split avoiding slice syntax for strict type checker
    raw_uuid = str(uuid.uuid4()).split('-')[0]
    cam = {'id': entry.id or f'cam_{raw_uuid}',
            'source': entry.source,
            'label': entry.label or f'Camera {len(cameras)}',
            'enabled': entry.enabled}
    cameras.append(cam)
    _write_cameras(cameras)
    ret_cam = dict(cam)
    ret_cam['source'] = _mask_source(str(cam['source']))
    return JSONResponse({'success': True, 'camera': ret_cam})


@api.put('/cameras/{cam_id}')
async def update_camera(cam_id: str, update: CameraUpdate):
    """Update source/label/enabled for a specific camera."""
    cameras = _read_cameras()
    cam = next((c for c in cameras if c['id'] == cam_id), None)
    if cam is None:
        return JSONResponse({'success': False, 'error': 'Camera not found'}, status_code=404)

    src = update.source
    if src is not None and src != cam['source']:
        # Help Pyre2 infer that src is strictly str here
        src_str: str = str(src)
        ok, err = _validate_source(src_str)
        if not ok:
            return JSONResponse(
                {'success': False, 'error': f'Source validation failed: {err}'},
                status_code=400
            )
        cam['source'] = src_str

    if update.label is not None:
        cam['label'] = update.label
    if update.enabled is not None:
        cam['enabled'] = update.enabled

    _write_cameras(cameras)
    ret_cam = dict(cam)
    ret_cam['source'] = _mask_source(str(cam['source']))
    return JSONResponse({'success': True, 'camera': ret_cam})


@api.delete('/cameras/{cam_id}')
async def delete_camera(cam_id: str):
    """Remove a camera from the config. live_scorer will stop the stream within 10s."""
    cameras = _read_cameras()
    orig_len = len(cameras)
    cameras = [c for c in cameras if c['id'] != cam_id]
    if len(cameras) == orig_len:
        return JSONResponse({'success': False, 'error': 'Camera not found'}, status_code=404)
    _write_cameras(cameras)
    return JSONResponse({'success': True, 'removed_id': cam_id})


@api.post('/cameras/test')
async def test_camera(entry: CameraEntry):
    """
    Test whether a camera source is reachable.
    Does NOT save the camera — purely a connectivity check.
    """
    ok, err = _validate_source(entry.source)
    if ok:
        return JSONResponse({'success': True, 'source': entry.source,
                             'message': 'Camera is reachable and streaming'})
    return JSONResponse({'success': False, 'source': entry.source, 'error': err},
                       status_code=400)

# ── Recordings ────────────────────────────────────────────────────────────────

@api.get('/recordings')
async def list_recordings(limit: int = 100):
    """List recorded video segments."""
    try:
        from app.core.storage import Storage
        storage = Storage()
        records = storage.get_recordings(limit=limit)
        return JSONResponse({'success': True, 'recordings': records})
    except Exception as e:
        log.error(f"Failed to list recordings: {e}")
        return JSONResponse({'success': False, 'error': str(e)}, status_code=500)

@api.get('/recordings/{rec_id}/stream')
async def stream_recording(rec_id: str):
    """Stream a specific recording file."""
    try:
        from app.core.storage import Storage
        storage = Storage()
        rec = storage.get_recording(rec_id)
        if not rec:
            return JSONResponse({'success': False, 'error': 'Recording not found in DB'}, status_code=404)
        
        file_path = os.path.join(storage.recordings_dir, rec['file_path'])
        if not os.path.exists(file_path):
            return JSONResponse({'success': False, 'error': 'Recording file missing from disk'}, status_code=404)
            
        from fastapi.responses import FileResponse
        return FileResponse(file_path, media_type="video/mp4")
    except Exception as e:
        log.error(f"Failed to stream recording {rec_id}: {e}")
        return JSONResponse({'success': False, 'error': str(e)}, status_code=500)

@api.delete('/recordings/{rec_id}')
async def delete_recording(rec_id: str):
    """Delete a recording from DB and disk."""
    try:
        from app.core.storage import Storage
        storage = Storage()
        rec = storage.get_recording(rec_id)
        if not rec:
            return JSONResponse({'success': False, 'error': 'Recording not found'}, status_code=404)
        
        file_path = os.path.join(storage.recordings_dir, rec['file_path'])
        if os.path.exists(file_path):
            os.remove(file_path)
            
        storage.delete_recording(rec_id)
        return JSONResponse({'success': True})
    except Exception as e:
        log.error(f"Failed to delete recording {rec_id}: {e}")
        return JSONResponse({'success': False, 'error': str(e)}, status_code=500)

