"""
state_api_routes.py — State Government Integration API

Public API endpoints authenticated via X-API-Key header (not session cookie).
These endpoints are the integration surface for State Government systems.

Endpoints:
  GET  /api/v1/health               Connectivity + auth check
  POST /api/v1/wanted               Upload a wanted-person image
  GET  /api/v1/wanted/{person_id}   Check upload status

Authentication:
  Header:  X-API-Key: vsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
  (Preferred over ?token= to avoid secrets in server logs)

Idempotency:
  Header:  Idempotency-Key: <unique-request-uuid>
  If the same key is sent twice, the original result is returned without
  creating a duplicate record.

Rate limiting:
  100 requests / minute per API key (in-memory sliding window).
"""

import logging
import time
from collections import defaultdict, deque
from typing import Optional

from fastapi import APIRouter, Request, HTTPException, Header, File, UploadFile, Form
from fastapi.responses import JSONResponse

log = logging.getLogger("state_api")
router = APIRouter(prefix="/api/v1", tags=["state_api"])

# ── Rate limiter (in-memory, per API key) -------------------------------------
_RATE_WINDOW   = 60        # seconds
_RATE_LIMIT    = 100       # max requests per window
_rate_store: dict[str, deque] = defaultdict(deque)

def _check_rate_limit(key_id: str) -> bool:
    """Sliding window rate limiter. Returns False if limit exceeded."""
    now = time.monotonic()
    window = _rate_store[key_id]
    while window and window[0] < now - _RATE_WINDOW:
        window.popleft()
    if len(window) >= _RATE_LIMIT:
        return False
    window.append(now)
    return True


# ── Auth dependency -----------------------------------------------------------

def _get_storage(request: Request):
    from app.services.face_service import face_service  # type: ignore
    return face_service._storage


async def _authenticate(
    request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> dict:
    """
    Validate the X-API-Key header.
    Logs the request, checks rate limit, returns the key record.
    Raises HTTP 401 / 429 on failure.
    """
    storage = _get_storage(request)

    if not x_api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing X-API-Key header. Obtain a key from the Vista AI administrator.",
        )

    key_record = storage.validate_api_key(x_api_key)
    if key_record is None:
        raise HTTPException(status_code=401, detail="Invalid or revoked API key.")

    if not _check_rate_limit(key_record["id"]):
        storage.log_api_request(
            api_key_id=key_record["id"],
            endpoint=request.url.path,
            method=request.method,
            status_code=429,
            source_ip=request.client.host if request.client else None,
        )
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Maximum {_RATE_LIMIT} requests per minute.",
        )

    return key_record


# ── Endpoints -----------------------------------------------------------------

@router.get("/health")
async def health_check(
    request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
):
    """
    Connectivity and authentication check for State Government systems.
    Returns 200 with integration identity info if the key is valid.
    """
    key_record = await _authenticate(request, x_api_key)
    return JSONResponse({
        "ok":           True,
        "status":       "connected",
        "integration":  key_record["name"],
        "state":        key_record["state"],
        "server":       "Vista AI — Central Dashboard",
        "api_version":  "v1",
    })


@router.post("/wanted")
async def upload_wanted_person(
    request:         Request,
    image:           UploadFile = File(..., description="JPEG or PNG face image"),
    name:            str        = Form(..., description="Full name of the wanted person"),
    crime:           str        = Form(default=None),
    case_number:     str        = Form(default=None),
    state:           str        = Form(default=None),
    notes:           str        = Form(default=None),
    x_api_key:       Optional[str] = Header(default=None, alias="X-API-Key"),
    idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
):
    """
    Upload a wanted-person image from a State Government system.

    The person is enrolled into the Vista AI face database, automatically
    classified as 'wanted', and tagged with provenance metadata.

    Idempotency: send the same Idempotency-Key to safely retry without
    creating duplicate records.
    """
    storage = _get_storage(request)

    # 1. Authenticate
    key_record = await _authenticate(request, x_api_key)
    key_id     = key_record["id"]
    source_state = key_record["state"] or state  # trust key's state over form field

    # 2. Idempotency check
    if idempotency_key:
        existing = storage.check_idempotency(idempotency_key)
        if existing:
            log.info(f"[STATE API] Idempotent replay: {idempotency_key}")
            return JSONResponse(existing)

    # 3. Validate input
    if not name or not name.strip():
        _log_and_fail(storage, key_id, request, 400, "name is required")
        raise HTTPException(status_code=400, detail="name is required")

    image_bytes = await image.read()
    if len(image_bytes) < 1024:
        _log_and_fail(storage, key_id, request, 400, "Image too small or empty")
        raise HTTPException(status_code=400, detail="Image too small or empty")

    content_type = image.content_type or ""
    if content_type not in ("image/jpeg", "image/png", "image/jpg") \
       and not image.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        raise HTTPException(status_code=400, detail="Only JPEG and PNG images are accepted")

    # 4. Enroll the person via existing face pipeline
    try:
        from app.services.face_service import face_service  # type: ignore
        result = face_service.enroll_person_bulk(
            name=name.strip(),
            role="WANTED",
            images_data=[image_bytes],
            classification="wanted",
        )
    except Exception as e:
        log.error(f"[STATE API] Enrollment error: {e}")
        _log_and_fail(storage, key_id, request, 500, str(e))
        raise HTTPException(status_code=500, detail=f"Enrollment failed: {e}")

    if not result.get("success"):
        err = result.get("error", "Enrollment failed")
        _log_and_fail(storage, key_id, request, 400, err)
        raise HTTPException(status_code=400, detail=err)

    person_id = result["person_id"]

    # 5. Tag with source provenance (STATE_API + originating state + key)
    storage.update_person_source(
        person_id=person_id,
        source_type="STATE_API",
        source_state=source_state,
        source_key_id=key_id,
        crime=crime,
        case_number=case_number,
    )

    # 6. Audit log
    storage.log_api_request(
        api_key_id=key_id,
        endpoint="/api/v1/wanted",
        method="POST",
        status_code=201,
        source_ip=request.client.host if request.client else None,
        request_id=idempotency_key,
    )

    # 7. Build response
    response = {
        "ok":          True,
        "person_id":   person_id,
        "name":        name.strip(),
        "source_type": "STATE_API",
        "source_state": source_state,
        "classification": "wanted",
        "message":     "Wanted person registered successfully in Vista AI.",
    }

    # 8. Store idempotency record
    if idempotency_key:
        storage.store_idempotency(idempotency_key, key_id, response)

    log.info(
        f"[STATE API] Wanted person registered: {name} | "
        f"id={person_id} | state={source_state} | key={key_record['key_prefix']}..."
    )
    return JSONResponse(response, status_code=201)


@router.get("/wanted/{person_id}")
async def get_upload_status(
    request:   Request,
    person_id: str,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
):
    """
    Check the status of a previously uploaded wanted person.
    State Gov can use this to confirm the upload was processed.
    """
    key_record = await _authenticate(request, x_api_key)
    storage    = _get_storage(request)

    from app.services.face_service import face_service  # type: ignore
    person = face_service._storage.get_person(person_id)

    if person is None:
        raise HTTPException(status_code=404, detail="Person not found")

    # Only expose records from this key's state (basic access control)
    if person.get("source_key_id") != key_record["id"]:
        raise HTTPException(
            status_code=403,
            detail="You do not have permission to view this record."
        )

    return JSONResponse({
        "ok":           True,
        "person_id":    person["id"],
        "name":         person.get("name"),
        "classification": person.get("classification"),
        "source_state": person.get("source_state"),
        "crime":        person.get("crime"),
        "case_number":  person.get("case_number"),
        "created_at":   person.get("created_at"),
    })


# ── Helpers -------------------------------------------------------------------

def _log_and_fail(storage, key_id, request, status_code, error_msg):
    try:
        storage.log_api_request(
            api_key_id=key_id,
            endpoint=request.url.path,
            method=request.method,
            status_code=status_code,
            source_ip=request.client.host if request.client else None,
            error_message=error_msg,
        )
    except Exception:
        pass
