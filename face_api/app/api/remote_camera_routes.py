"""
remote_camera_routes.py — WebSocket Frame Receiver & VirtualCameraStream Registry

Enables cameras on any network to stream to this server via an outbound
WebSocket connection from camera_relay.py (the edge gateway agent).

Architecture:
    camera_relay.py  (any network)
         | WSS/WS binary JPEG frames
         v
    /api/camera/ws/{cam_id}         <- WebSocket endpoint (this module)
         |
         v
    VirtualCameraStream             <- same .read() interface as CameraStream
         |
    GET /api/camera/frame/{cam_id} <- live_scorer polls this HTTP endpoint
         |
    RemoteCameraSource (in live_scorer process)
         |
    CameraProcessor (face recog, boundary, recording)
"""

import os
import time
import logging
import threading
import numpy as np
import cv2
from collections import deque
from typing import Dict, Optional, Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, HTTPException
from fastapi.responses import JSONResponse

log = logging.getLogger("remote_camera")
router = APIRouter(prefix="/api/camera", tags=["remote_camera"])

# -- Auth ---------------------------------------------------------------------
_SESSION_SECRET_PATH = os.environ.get(
    "SESSION_SECRET_PATH",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))))),
        "data", ".session_secret"
    )
)

def _load_secret() -> Optional[str]:
    try:
        if os.path.exists(_SESSION_SECRET_PATH):
            return open(_SESSION_SECRET_PATH).read().strip()
    except Exception:
        pass
    return None

def _check_token(token: Optional[str], auth_header: Optional[str] = None) -> bool:
    """
    Returns True if:
      - No session secret is configured (open-access on local net)
      - ?token= query param matches the secret
      - Authorization: Bearer <token> header matches the secret
    Prefers Authorization header over URL param to avoid secrets in logs.
    """
    secret = _load_secret()
    if secret is None:
        return True
    if auth_header:
        parts = auth_header.split(None, 1)
        if len(parts) == 2 and parts[0].lower() == "bearer":
            return parts[1].strip() == secret
    return token == secret


# -- VirtualCameraStream ------------------------------------------------------
class VirtualCameraStream:
    """
    Drop-in replacement for CameraStream that receives frames pushed
    over WebSocket instead of reading from a local camera.

    Interface matches CameraStream exactly:
        .read()  -> (ret: bool, frame: np.ndarray | None, frame_id: int)
        .stopped -> bool
        .start() -> self
        .stop()  -> None
    """

    STALE_TIMEOUT   = 30.0
    OFFLINE_TIMEOUT = 60.0

    def __init__(self, cam_id: str):
        self.cam_id    = cam_id
        self.stopped   = False

        self._frame:    Optional[np.ndarray] = None
        self._frame_id: int   = 0
        self._lock      = threading.Lock()
        self._last_seen: float = 0.0

        self.frames_received: int   = 0
        self.bytes_received:  int   = 0
        self.connect_time:    float = time.time()
        self.last_fps:        float = 0.0
        self._fps_window:     deque = deque(maxlen=30)

    def push_jpeg(self, jpeg_bytes: bytes) -> None:
        """Decode JPEG and store as latest frame. Called by WebSocket handler."""
        try:
            nparr = np.frombuffer(jpeg_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                return
        except Exception:
            return

        now = time.time()
        with self._lock:
            self._frame    = frame
            self._frame_id += 1
            self._last_seen = now
            self.frames_received += 1
            self.bytes_received  += len(jpeg_bytes)
            self._fps_window.append(now)
            if len(self._fps_window) >= 2:
                span = self._fps_window[-1] - self._fps_window[0]
                self.last_fps = (len(self._fps_window) - 1) / span if span > 0 else 0.0

        # Also store raw JPEG bytes for cross-process HTTP frame endpoint
        with _REMOTE_FRAME_LOCK:
            _REMOTE_FRAME_IDS[self.cam_id] = (jpeg_bytes, now, self._frame_id)

    # CameraStream interface
    def read(self):
        with self._lock:
            if self._frame is None:
                return False, None, self._frame_id
            return True, self._frame.copy(), self._frame_id

    def start(self) -> "VirtualCameraStream":
        return self

    def stop(self) -> None:
        self.stopped = True

    @property
    def status(self) -> str:
        if self._last_seen == 0:
            return "WAITING"
        age = time.time() - self._last_seen
        if age > self.OFFLINE_TIMEOUT:
            return "OFFLINE"
        if age > self.STALE_TIMEOUT:
            return "STALE"
        return "LIVE"

    def to_dict(self) -> Dict[str, Any]:
        age = round(time.time() - self._last_seen, 1) if self._last_seen else None
        return {
            "camera_id":       self.cam_id,
            "status":          self.status,
            "frames_received": self.frames_received,
            "bytes_received":  self.bytes_received,
            "fps":             round(self.last_fps, 1),
            "last_seen_ago_s": age,
            "uptime_s":        round(time.time() - self.connect_time, 0),
        }


# -- Frame ID tracking (monotonic counter per camera) -------------------------
_REMOTE_FRAME_IDS: dict = {}   # cam_id -> int frame_id (increments each new push)
_REMOTE_FRAME_LOCK = threading.Lock()

# -- Global Registry ----------------------------------------------------------
_REMOTE_STREAMS: Dict[str, VirtualCameraStream] = {}
_registry_lock  = threading.Lock()


def get_remote_stream(cam_id: str) -> Optional[VirtualCameraStream]:
    return _REMOTE_STREAMS.get(cam_id)


def get_all_remote_streams() -> Dict[str, VirtualCameraStream]:
    with _registry_lock:
        return dict(_REMOTE_STREAMS)


def register_remote_stream(cam_id: str) -> VirtualCameraStream:
    with _registry_lock:
        if cam_id not in _REMOTE_STREAMS:
            _REMOTE_STREAMS[cam_id] = VirtualCameraStream(cam_id)
            log.info(f"[REMOTE] Registered stream: {cam_id}")
        return _REMOTE_STREAMS[cam_id]


def unregister_remote_stream(cam_id: str):
    with _registry_lock:
        stream = _REMOTE_STREAMS.pop(cam_id, None)
        if stream:
            stream.stop()
            log.info(f"[REMOTE] Unregistered stream: {cam_id}")


# -- WebSocket Endpoint -------------------------------------------------------
@router.websocket("/ws/{camera_id}")
async def camera_ws_endpoint(
    websocket: WebSocket,
    camera_id: str,
    token: Optional[str] = Query(default=None),
):
    # Support Authorization: Bearer <token> header (preferred over URL param)
    auth_header = websocket.headers.get("authorization")
    """
    WebSocket endpoint for camera_relay.py agents.
    Accepts binary JPEG frames from any remote device.
    Auth: ?token=SECRET  (skipped if no session secret configured)
    """
    if not _check_token(token, auth_header):
        await websocket.close(code=4003, reason="Unauthorized")
        log.warning(f"[REMOTE] Rejected unauthorized relay for {camera_id}")
        return

    await websocket.accept()
    log.info(f"[REMOTE] Relay connected: {camera_id} from {websocket.client}")

    stream = register_remote_stream(camera_id)
    stream.stopped = False
    frames_this_session = 0

    try:
        while True:
            try:
                data = await websocket.receive_bytes()
            except Exception:
                break

            if not data:
                continue

            stream.push_jpeg(data)
            frames_this_session += 1

    except WebSocketDisconnect:
        log.info(f"[REMOTE] Relay disconnected: {camera_id} ({frames_this_session} frames)")
    except Exception as e:
        log.error(f"[REMOTE] Error for {camera_id}: {e}")
    finally:
        log.info(f"[REMOTE] Session ended: {camera_id}")




# -- Status Endpoints ---------------------------------------------------------
@router.get("/remote/status")
async def get_all_remote_status():
    """List all registered remote camera streams with health info."""
    streams = get_all_remote_streams()
    return JSONResponse({
        "count":   len(streams),
        "cameras": [s.to_dict() for s in streams.values()],
    })


@router.get("/remote/{camera_id}/status")
async def get_remote_camera_status(camera_id: str):
    """Health info for a specific remote camera."""
    stream = get_remote_stream(camera_id)
    if stream is None:
        raise HTTPException(status_code=404, detail=f"No remote stream: {camera_id}")
    return JSONResponse(stream.to_dict())


@router.delete("/remote/{camera_id}")
async def remove_remote_camera(camera_id: str):
    """Disconnect and unregister a remote camera."""
    unregister_remote_stream(camera_id)
    return JSONResponse({"ok": True, "camera_id": camera_id})


# -- Cross-Process HTTP Frame Endpoint ----------------------------------------
# live_scorer (separate process) polls this to get frames from remote cameras.
# Returns: JPEG bytes with X-Frame-Id and X-Frame-Timestamp response headers.

from fastapi import Response as FastAPIResponse

@router.get("/frame/{camera_id}")
async def get_remote_frame(camera_id: str):
    """
    Returns the latest JPEG frame received from a remote camera relay.
    Used by live_scorer.RemoteCameraSource to poll frames across the process boundary.

    Response headers:
      X-Frame-Id        monotonic frame counter (int)
      X-Frame-Timestamp epoch timestamp of when frame was received
      X-Camera-Status   LIVE / STALE / OFFLINE / WAITING
    """
    with _REMOTE_FRAME_LOCK:
        entry = _REMOTE_FRAME_IDS.get(camera_id)

    stream = get_remote_stream(camera_id)
    status = stream.status if stream else "NOT_REGISTERED"

    if entry is None:
        return FastAPIResponse(
            content=b"",
            status_code=204,  # No Content — camera registered but no frames yet
            headers={
                "X-Frame-Id":        "0",
                "X-Frame-Timestamp": "0",
                "X-Camera-Status":   status,
            }
        )

    jpeg_bytes, ts, frame_id = entry
    return FastAPIResponse(
        content=jpeg_bytes,
        media_type="image/jpeg",
        headers={
            "X-Frame-Id":        str(frame_id),
            "X-Frame-Timestamp": str(ts),
            "X-Camera-Status":   status,
        }
    )

