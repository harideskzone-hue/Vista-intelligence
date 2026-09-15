"""
apikey_routes.py — Central Government: API Key Administration

Endpoints (session-authenticated — Central Gov admin only):
  GET    /api/admin/keys            List all API keys
  POST   /api/admin/keys            Generate a new key (full key returned ONCE)
  DELETE /api/admin/keys/{key_id}   Revoke / soft-delete a key
  PATCH  /api/admin/keys/{key_id}   Enable / disable a key
  GET    /api/admin/keys/{key_id}/logs  Audit log for a key
  GET    /api/admin/uploads/recent  Last N wanted-person uploads via STATE_API
"""

import logging
from typing import Optional
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

log = logging.getLogger("apikey_admin")
router = APIRouter(prefix="/api/admin", tags=["api_keys"])


def _get_storage(request=None):
    """Retrieve the shared Storage singleton via get_storage()."""
    from app.core.storage import get_storage  # type: ignore
    return get_storage()


# ── Request / Response models -------------------------------------------------

class CreateKeyRequest(BaseModel):
    name:  str
    state: Optional[str] = None


class PatchKeyRequest(BaseModel):
    is_active: bool


# ── Endpoints -----------------------------------------------------------------

@router.get("/keys")
async def list_keys(request: Request):
    """List all API keys (prefix + metadata only — full key never returned)."""
    storage = _get_storage(request)
    keys = storage.list_api_keys()
    return JSONResponse({"ok": True, "keys": keys, "count": len(keys)})


@router.post("/keys")
async def create_key(request: Request, body: CreateKeyRequest):
    """
    Generate a new API key.
    The full key string is returned ONCE in this response.
    It is never stored in plaintext and cannot be retrieved again.
    """
    if not body.name or not body.name.strip():
        raise HTTPException(status_code=400, detail="name is required")

    storage = _get_storage(request)
    full_key, record = storage.create_api_key(
        name=body.name.strip(),
        state=body.state.strip() if body.state else None
    )
    log.info(f"[API KEY] Created: {record['key_prefix']}... for '{body.name}'")

    return JSONResponse({
        "ok":      True,
        "key":     full_key,       # shown ONCE — not stored in plaintext
        "record":  record,
        "warning": "This is the only time this key will be shown. Copy it now.",
    })


@router.delete("/keys/{key_id}")
async def delete_or_revoke_key(request: Request, key_id: str):
    """
    DELETE /api/admin/keys/{key_id}

    By default permanently deletes (clears) the API key and all its audit logs.
    If ?mode=soft or ?mode=revoke is passed (or X-Delete-Mode: soft), it revokes the key instead.
    """
    storage = _get_storage(request)
    mode = (
        request.query_params.get("mode")
        or request.headers.get("x-delete-mode", "hard")
    ).lower()

    if mode in ("soft", "revoke"):
        # Soft revoke — stops service, record kept
        success = storage.revoke_api_key(key_id)
        if not success:
            raise HTTPException(status_code=404, detail="API key not found")
        log.info(f"[API KEY] Revoked (service stopped): {key_id}")
        return JSONResponse({"ok": True, "key_id": key_id, "status": "revoked"})
    else:
        # Permanent hard delete — removes key row and all api_request_logs
        success = storage.delete_api_key(key_id)
        if not success:
            raise HTTPException(status_code=404, detail="API key not found")
        log.info(f"[API KEY] Hard-deleted (cleared): {key_id}")
        return JSONResponse({"ok": True, "key_id": key_id, "status": "deleted"})


@router.post("/keys/{key_id}/revoke")
async def revoke_key_endpoint(request: Request, key_id: str):
    """Explicit endpoint to stop API service (revoke key)."""
    storage = _get_storage(request)
    success = storage.revoke_api_key(key_id)
    if not success:
        raise HTTPException(status_code=404, detail="API key not found")
    log.info(f"[API KEY] Revoked: {key_id}")
    return JSONResponse({"ok": True, "key_id": key_id, "status": "revoked"})


@router.post("/keys/{key_id}/enable")
async def enable_key_endpoint(request: Request, key_id: str):
    """Explicit endpoint to resume API service (enable key)."""
    storage = _get_storage(request)
    success = storage.enable_api_key(key_id)
    if not success:
        raise HTTPException(status_code=404, detail="API key not found")
    log.info(f"[API KEY] Enabled: {key_id}")
    return JSONResponse({"ok": True, "key_id": key_id, "status": "active"})


@router.patch("/keys/{key_id}")
async def patch_key(request: Request, key_id: str, body: PatchKeyRequest):
    """Enable or disable an API key."""
    storage = _get_storage(request)
    if body.is_active:
        success = storage.enable_api_key(key_id)
    else:
        success = storage.revoke_api_key(key_id)

    if not success:
        raise HTTPException(status_code=404, detail="API key not found")

    status = "active" if body.is_active else "revoked"
    log.info(f"[API KEY] {status}: {key_id}")
    return JSONResponse({"ok": True, "key_id": key_id, "status": status})


@router.get("/keys/{key_id}/logs")
async def get_key_logs(request: Request, key_id: str, limit: int = 50):
    """Retrieve the audit log for a specific API key."""
    storage = _get_storage(request)
    logs = storage.get_api_key_logs(key_id, limit=limit)
    return JSONResponse({"ok": True, "key_id": key_id, "logs": logs, "count": len(logs)})


@router.get("/uploads/recent")
async def recent_api_uploads(request: Request, limit: int = 10):
    """Last N wanted-person records uploaded via State Government API."""
    storage = _get_storage(request)
    uploads = storage.get_recent_api_uploads(limit=limit)
    return JSONResponse({"ok": True, "uploads": uploads, "count": len(uploads)})
