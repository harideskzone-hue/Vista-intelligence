import os
import sys
import json
import cv2  # type: ignore
import time
import subprocess
import threading
import os

LAZY_ENCODING = os.environ.get('LAZY_ENCODING', 'false').lower() == 'true'
OPTIMIZE_SCHEDULING = os.environ.get('OPTIMIZE_SCHEDULING', 'false').lower() == 'true'
INFERENCE_MODE = os.environ.get('INFERENCE_MODE', 'legacy')
INFERENCE_BATCH_SIZE = int(os.environ.get('INFERENCE_BATCH_SIZE', '4'))

import queue
import datetime
import uuid
import sys
import json
import os

# Add face_api to sys path to import Storage
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "face_api"))
try:
    from app.core.storage import Storage
    _storage = Storage()
except Exception as e:
    print(f"[CAM] Failed to initialize storage: {e}")
    _storage = None

from ultralytics import YOLO  # type: ignore
import numpy as np  # type: ignore
from collections import deque
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, Response
from collections import deque

# Add current dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from edge.face.enhancement import AdaptiveEnhancer

# Load .env
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
except ImportError:
    pass

def _parse_camera_source(val: str):
    """Parse a camera source string: numeric string -> int (webcam), URL -> str."""
    if val.strip().lstrip('-').isdigit():
        return int(val.strip())
    return val.strip()


# ── MJPEG Stream Server ───────────────────────────────────────────────────────
_STREAM_FRAMES = {}  # cam_id -> (frame_version, bytes)
_STREAM_VERSIONS = {} # cam_id -> int
_STREAM_BUFFERS = {} # cam_id -> deque of (timestamp, bytes)
_stream_app = FastAPI()

import threading
import time
import cv2

DASHBOARD_OPTIMIZATIONS_ENABLED = True

class SharedJPEGEncoder:
    def __init__(self):
        self._lock = threading.Lock()
        self.last_encoded_version = -1
        self.cached_jpeg = None
        self.last_encoded_time = 0.0
        self.jpeg_encodes = 0
        self.unique_frames = 0
        self.clients_served = 0
        self.stale_skips = 0

    def get_jpeg(self, frame_version: int, frame) -> bytes:
        import time
        with self._lock:
            self.clients_served += 1
            if self.last_encoded_version == frame_version and self.cached_jpeg is not None:
                return self.cached_jpeg
            
            now = time.monotonic()
            
            if DASHBOARD_OPTIMIZATIONS_ENABLED:
                if self.cached_jpeg is not None and (now - self.last_encoded_time) < 0.2:
                    if self.last_encoded_version != -1 and frame_version > self.last_encoded_version:
                        self.stale_skips += (frame_version - self.last_encoded_version)
                    # Need to sync the version so we don't count skips twice for the same version
                    self.last_encoded_version = frame_version
                    return self.cached_jpeg
            
            # Not cached, or rate limit allows encoding a new one
            encode_frame = frame
            if DASHBOARD_OPTIMIZATIONS_ENABLED:
                import cv2
                encode_frame = cv2.resize(frame, (320, 240))
            
            _, buf = cv2.imencode('.jpg', encode_frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
            self.cached_jpeg = buf.tobytes()
            self.last_encoded_time = now
            
            if self.last_encoded_version != -1 and frame_version > self.last_encoded_version + 1:
                pass
                
            self.last_encoded_version = frame_version
            self.jpeg_encodes += 1
            self.unique_frames += 1
            return self.cached_jpeg

_SHARED_ENCODERS = {}
_ENCODERS_LOCK = threading.Lock()

def get_shared_encoder(cam_id: str) -> SharedJPEGEncoder:
    with _ENCODERS_LOCK:
        if cam_id not in _SHARED_ENCODERS:
            _SHARED_ENCODERS[cam_id] = SharedJPEGEncoder()
        return _SHARED_ENCODERS[cam_id]


def _make_placeholder_jpeg(text: str = "Connecting...") -> bytes:
    """Generate a minimal black JPEG with status text using OpenCV."""
    import numpy as np  # type: ignore
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.putText(img, text, (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 80, 80), 2)
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 60])
    return buf.tobytes()


_PLACEHOLDER_JPEG = None  # lazy-initialised on first request


@_stream_app.get("/video_buffer/{cam_id}")
async def get_buffered_frame(cam_id: str, offset: float = 0.0):
    import time
    buf = _STREAM_BUFFERS.get(cam_id)
    if not buf or len(buf) == 0:
        return Response(status_code=404)
    target_time = time.time() + offset
    best_frame = buf[-1][1]
    min_diff = float('inf')
    # search backwards for closest frame
    for ts, frame_bytes in reversed(buf):
        diff = abs(ts - target_time)
        if diff <= min_diff:
            min_diff = diff
            best_frame = frame_bytes
        else:
            break
    if isinstance(best_frame, np.ndarray):
        _, buf_enc = cv2.imencode(".jpg", best_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        best_frame = buf_enc.tobytes()
    return Response(content=best_frame, media_type="image/jpeg")

@_stream_app.get("/video_feed/{cam_id}")
async def video_feed(cam_id: str):
    encoder = get_shared_encoder(cam_id)
    def generate():
        global _PLACEHOLDER_JPEG
        if _PLACEHOLDER_JPEG is None:
            _PLACEHOLDER_JPEG = _make_placeholder_jpeg("Connecting...")
        while True:
            entry = _STREAM_FRAMES.get(cam_id)
            out = _PLACEHOLDER_JPEG
            if entry is not None and isinstance(entry, tuple) and len(entry) == 2:
                frame_version, frame = entry
                if isinstance(frame, np.ndarray): # Lazy encoding
                    out = encoder.get_jpeg(frame_version, frame)
                else:
                    out = frame
            elif entry is not None:
                out = entry
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + out + b'\r\n')
            time.sleep(0.04)  # ~25 fps
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

threading.Thread(target=lambda: uvicorn.run(_stream_app, host="127.0.0.1", port=5002, log_level="warning"), daemon=True).start()

# ── Preflight Validation ──────────────────────────────────────────────────────
def _preflight():
    """
    Validate all critical dependencies before starting the pipeline.
    Returns True if all required checks pass.
    """
    # Make sure we import model_manager so it can locate the right directory
    from model_manager import MODELS_DIR
    _MODELS_DIR = MODELS_DIR

    checks = []

    # 1. Model files
    required_models = ["yolo26n-face.pt"]
    optional_models = []

    for m in required_models:
        path = os.path.join(_MODELS_DIR, m)
        exists = os.path.isfile(path)
        size_mb = os.path.getsize(path) / 1e6 if exists else 0
        checks.append(("✓" if exists else "✗", m, f"{size_mb:.1f} MB" if exists else "MISSING (REQUIRED)"))
        if not exists:
            # Attempt auto-download
            try:
                sys.path.insert(0, _ENGINE_DIR)
                from model_manager import download_missing  # type: ignore
                download_missing()
            except Exception:
                pass

    for m in optional_models:
        path = os.path.join(_MODELS_DIR, m)
        exists = os.path.isfile(path)
        size_mb = os.path.getsize(path) / 1e6 if exists else 0
        checks.append(("✓" if exists else "⚠", m, f"{size_mb:.1f} MB" if exists else "missing (optional)"))

    # 2. Camera test
    cam_ok = False
    cam_detail = "UNAVAILABLE"
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cam_ok = True
            cam_detail = f"{w}×{h}"
            cap.release()
        else:
            cap.release()
    except Exception:
        pass
    checks.append(("✓" if cam_ok else "⚠", "Camera 0", cam_detail))

    # 3. API health
    api_ok = False
    api_detail = "NOT RESPONDING"
    try:
        import urllib.request
        API_PORT = int(os.environ.get("PORT", 5001))
        with urllib.request.urlopen(f"http://localhost:{API_PORT}/api/health", timeout=3) as resp:
            if resp.status == 200:
                import json as _json
                data = _json.loads(resp.read())
                api_ok = True
                api_detail = f"200 OK, {data.get('person_count', 0)} persons"
    except Exception:
        pass
    checks.append(("✓" if api_ok else "⚠", "API health", api_detail))

    # 4. Session token
    token_ok = False
    if getattr(sys, 'frozen', False):
        _base = os.path.dirname(sys.executable)
    else:
        _base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
    _usr_dir = os.environ.get("SIH26187_DATA", os.path.join(_base, "data"))
    _secret = os.path.join(_usr_dir, ".session_secret")
    if os.path.isfile(_secret):
        token_ok = True
    checks.append(("✓" if token_ok else "⚠", "Session token", "loaded" if token_ok else "missing (uploads may fail)"))

    # 5. Disk space
    import shutil
    total, used, free = shutil.disk_usage(os.path.expanduser("~"))
    free_gb = free / (1024**3)
    disk_ok = free_gb >= 0.5
    checks.append(("✓" if disk_ok else "⚠", "Disk space", f"{free_gb:.1f} GB free"))

    # Print table
    print("\n┌─ PREFLIGHT ──────────────────────────────────────────┐")
    for sym, name, detail in checks:
        print(f"│ {sym} {name:<28} ({detail:<20}) │")
    print("└──────────────────────────────────────────────────────┘\n")

    # Check for fatal failures
    fatal = any(sym == "✗" for sym, _, _ in checks)
    if fatal:
        print("[PREFLIGHT] ✗ Some required checks failed. See above.")
        return False
    return True


# ── Threaded camera reader ────────────────────────────────────────────────────
class CameraStream:
    def __init__(self, src):
        if isinstance(src, str) and src.isdigit():
            src = int(src)
        self.src = src
        # For local USB/built-in cameras (integer index), explicitly use the
        # AVFoundation backend on macOS to prevent the OBSENSOR (Orbbec depth
        # camera) backend from intercepting the device and failing.
        if isinstance(src, int):
            self.stream = cv2.VideoCapture(src, cv2.CAP_AVFOUNDATION)
            self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            # ── USB BANDWIDTH FIX ──────────────────────────────────────────────────────
            # macOS AVFoundation frequently ignores the MJPEG FOURCC request and falls
            # back to uncompressed YUV420. At 1280×720 @ 30fps each camera consumes
            # ~55 MB/s. Two cameras (~110 MB/s) saturate the USB controller, leaving
            # Camera 2 with zero bandwidth. It negotiates (isOpened=True) but every
            # subsequent read() returns empty => "Failed to read initial frame from 2."
            #
            # Fix: request 640×480. Even as raw YUV this is only ~18 MB/s per camera,
            # so 6+ cameras comfortably share one USB controller. The inference engine
            # already downscales to 640×360 internally, so AI accuracy is unaffected.
            self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            # DO NOT set CAP_PROP_FPS on macOS AVFoundation!
            # Setting FPS on identical external USB cameras causes a kernel-level
            # hardware lockup which hangs cv2.VideoCapture indefinitely!
        else:
            self.stream = cv2.VideoCapture(src)
            self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            
        if self.stream.isOpened():
            w = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.stream.get(cv2.CAP_PROP_FPS)
            fourcc_val = int(self.stream.get(cv2.CAP_PROP_FOURCC))
            def decode_fourcc(v):
                return "".join([chr((int(v) >> 8 * i) & 0xFF) for i in range(4)])
            fourcc_str = decode_fourcc(fourcc_val) if fourcc_val > 0 else "UNKNOWN"
            print(f"  [CAM {src}] Negotiated: {w}x{h} @ {fps}fps, codec: {fourcc_str}")

        if not self.stream.isOpened():
            print(f"Failed to open {src}.")
            self.stopped = True
            return

        # Keep OpenCV's internal buffer as small as possible so we always
        # get the newest frame rather than reading stale buffered frames.
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # ── NON-BLOCKING WARMUP ────────────────────────────────────────────────
        # Do NOT attempt a synchronous read() here to "verify" the camera.
        # When other cameras are already streaming, their grab threads compete
        # for USB time, causing Camera 2/3/etc. synchronous reads to always
        # fail -> "Failed to read initial frame from 2."
        #
        # isOpened() is sufficient proof of hardware acceptance.
        # The background grab thread (started by CameraStream.start()) will
        # deliver the first real frame within ~300ms once the USB bus settles.
        # If the grab loop finds the stream truly dead, it stops itself via the
        # max_consecutive_failures / max_stale_time guard already in _update().
        self.ret = True   # Assume alive; grab loop will flip this if truly dead
        self.frame = None # No frame yet; process_frame() guards against None

        self.stopped = False
        self.frame_id = 0
        self._lock = threading.Lock()
        self._thread: "threading.Thread | None" = None
        
        # Telemetry
        from collections import deque
        self.capture_fps = 0.0
        self._frame_times = deque(maxlen=30)
        self.dropped_frames = 0
        self._frame_read = True

    def start(self):
        self._thread = threading.Thread(target=self._update, daemon=True)  # type: ignore
        self._thread.start()  # type: ignore
        return self

    def _update(self):
        """Drain OpenCV's internal network buffer with grab() in a tight loop.
        We only decode (retrieve) when the main thread actually wants a frame.
        This keeps the buffer empty so read() always returns the freshest frame."""
        import time
        consecutive_failures = 0
        last_success_time = time.time()
        max_consecutive_failures = 300  # ~1.5 seconds if grab returns instantly
        max_stale_time = 15.0  # seconds until we assume the stream is completely dead
        
        while not self.stopped:
            # grab() signals the camera / advances the buffer pointer (cheap)
            is_file = isinstance(self.src, str) and self.src.lower().endswith(('.mp4', '.avi', '.mov'))
            
            grabbed = False
            if is_file:
                grabbed = self.stream.grab()
            else:
                # DRAIN THE OS BUFFER (Fixes progressive lag on physical cameras)
                # If grab() is < 10ms, it was buffered. If > 10ms, we waited for hardware!
                while True:
                    t0 = time.time()
                    grabbed = self.stream.grab()
                    if not grabbed or (time.time() - t0) * 1000 > 12.0:
                        break

            if not grabbed:
                consecutive_failures += 1
                elapsed = time.time() - last_success_time
                if consecutive_failures > max_consecutive_failures or elapsed > max_stale_time:
                    print(f"  [CAM] Stream {self.src} disconnected (grab failed {consecutive_failures} times over {elapsed:.1f}s)")
                    self.stopped = True
                    break
                time.sleep(0.005)
                continue
            
            consecutive_failures = 0
            last_success_time = time.time()
            # Decode the grabbed frame and expose it to the main thread
            ret, frame = self.stream.retrieve()
            if ret:
                with self._lock:
                    if self.frame_id > 0 and not getattr(self, '_frame_read', True):
                        self.dropped_frames += 1
                        
                    self.ret = ret
                    self.frame = frame
                    self.frame_id += 1
                    self._frame_read = False
                    
                    self._frame_times.append(time.time())
                    if len(self._frame_times) > 1:
                        self.capture_fps = len(self._frame_times) / (self._frame_times[-1] - self._frame_times[0])
            
            # Throttle if reading from a local video file to simulate real-time stream
            if isinstance(self.src, str) and self.src.lower().endswith(('.mp4', '.avi', '.mov')):
                time.sleep(1/30.0)
                    
        # Safely release the camera inside the thread that owns it to prevent Segfaults
        self.stream.release()

    def read(self):
        with self._lock:
            self._frame_read = True
            return self.ret, self.frame, self.frame_id

    def stop(self):
        self.stopped = True
        if hasattr(self, '_thread') and self._thread.is_alive():  # type: ignore
            self._thread.join(timeout=2.0)  # type: ignore



class CentralInferenceWorker:
    def __init__(self, model_path, batch_size=4, use_mps=True):
        self.batch_size = batch_size
        self.use_mps = use_mps
        
        import torch
        from ultralytics import YOLO
        
        # Load model explicitly with correct device
        self.device = 'mps' if (self.use_mps and torch.backends.mps.is_available()) else 'cpu'
        print(f"[CentralInferenceWorker] Initializing YOLO on {self.device} with batch size {self.batch_size}")
        self.model = YOLO(model_path)
        
        self.frames_lock = threading.Lock()
        self.latest_frames = {} # cam_id -> frame
        self.results = {}       # cam_id -> raw bounding boxes
        
        # Tracking states
        self.tracks = {}
        self.next_track_ids = {}
        
        self._stopped = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        
    def submit(self, cam_id, frame):
        with self.frames_lock:
            self.latest_frames[cam_id] = frame
            if cam_id not in self.tracks:
                self.tracks[cam_id] = {}
                self.next_track_ids[cam_id] = 1
                self.results[cam_id] = []
                
    def get_result(self, cam_id):
        # Tracking logic ported here to preserve state per camera
        boxes = self.results.get(cam_id, [])
        tracks = self.tracks.get(cam_id, {})
        next_id = self.next_track_ids.get(cam_id, 1)
        
        new_tracks = {}
        matched = set()
        
        def _iou(boxA, boxB):
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])
            interArea = max(0, xB - xA) * max(0, yB - yA)
            boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
            boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
            return interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0
            
        for tid, last_box in tracks.items():
            best_iou = 0
            best_idx = -1
            for idx, det in enumerate(boxes):
                if idx in matched: continue
                iou = _iou(last_box, det)
                if iou > best_iou:
                    best_iou, best_idx = iou, idx
            if best_iou > 0.3:
                new_tracks[tid] = boxes[best_idx]
                matched.add(best_idx)
                
        for idx, det in enumerate(boxes):
            if idx not in matched:
                new_tracks[next_id] = det
                next_id += 1
                
        self.tracks[cam_id] = new_tracks
        self.next_track_ids[cam_id] = next_id
        
        return [{'track_id': str(tid), 'box': b} for tid, b in new_tracks.items()]
        
    def _run(self):
        import cv2
        while not self._stopped:
            import time
            batch = []
            cam_ids = []
            orig_shapes = []
            
            with self.frames_lock:
                for cid, frame in list(self.latest_frames.items()):
                    if frame is not None:
                        orig_shapes.append(frame.shape[:2]) # (h, w)
                        # Resize for inference (16:9 ratio) to 640x360 as benchmarked
                        inference_frame = cv2.resize(frame, (640, 360))
                        batch.append(inference_frame)
                        cam_ids.append(cid)
                        self.latest_frames[cid] = None # Clear after taking
                    if len(batch) >= self.batch_size:
                        break
                        
            if not batch:
                time.sleep(0.01)
                continue
                
            try:
                # YOLO batch inference
                out = self.model(batch, verbose=False, conf=0.5, device=self.device)
                
                for i, res in enumerate(out):
                    cid = cam_ids[i]
                    orig_h, orig_w = orig_shapes[i]
                    scale_x = orig_w / 640.0
                    scale_y = orig_h / 360.0
                    
                    boxes = []
                    if len(res.boxes) > 0:
                        for box in res.boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            # Scale back to original resolution
                            x1, x2 = x1 * scale_x, x2 * scale_x
                            y1, y2 = y1 * scale_y, y2 * scale_y
                            boxes.append((x1, y1, x2, y2))
                    self.results[cid] = boxes
                    
            except Exception as e:
                print(f"[CentralInferenceWorker] Batch inference error: {e}")
                
            # Yield GIL to UI thread (prevents UI starving when inference is at 100% duty cycle)
            time.sleep(0.02)
                
    def stop(self):
        self._stopped = True

_global_central_worker = None

# ── Non-blocking Face inference thread ───────────────────────────────────────

class FaceDetectorThread:
    """
    Runs face detection and basic IoU tracking in a dedicated thread.
    """
    def __init__(self, face_model, conf):
        self.model = face_model
        self.conf = conf

        # Queue of depth-1: newest frame only
        self._q = queue.Queue(maxsize=1)
        self.result = []          # list of dicts: {'track_id': str, 'box': (x1,y1,x2,y2)}
        self._stopped = False
        self._busy = False
        self._thread = None
        
        self.next_track_id = 1
        self.tracks = {} # track_id -> (x1, y1, x2, y2)

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def submit(self, frame):
        """Submit a frame for detection. Non-blocking."""
        try:
            self._q.get_nowait()
        except queue.Empty:
            pass
        try:
            self._q.put_nowait(frame)
        except queue.Full:
            pass

    @property
    def is_busy(self):
        return self._busy

    def _iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea + 1e-5)

    def _run(self):
        while not self._stopped:
            try:
                frame = self._q.get(timeout=0.1)
            except queue.Empty:
                continue

            self._busy = True
            try:
                # Primary: Face detection
                results = self.model(frame, conf=self.conf, verbose=False)
                boxes = results[0].boxes
                
                current_detections = []
                if len(boxes) > 0:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        current_detections.append((x1, y1, x2, y2))
                        
                # Simple IoU matching for track stability
                new_tracks = {}
                matched_detections = set()
                
                for track_id, last_box in self.tracks.items():
                    best_iou = 0
                    best_idx = -1
                    for idx, det_box in enumerate(current_detections):
                        if idx in matched_detections:
                            continue
                        iou = self._iou(last_box, det_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_idx = idx
                            
                    if best_iou > 0.3:
                        new_tracks[track_id] = current_detections[best_idx]
                        matched_detections.add(best_idx)
                        
                for idx, det_box in enumerate(current_detections):
                    if idx not in matched_detections:
                        new_tracks[self.next_track_id] = det_box
                        self.next_track_id += 1
                        
                self.tracks = new_tracks
                self.result = [{'track_id': str(tid), 'box': box} for tid, box in self.tracks.items()]
                
            except Exception:
                pass
            finally:
                self._busy = False

    def stop(self):
        self._stopped = True
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)



# ── RemoteCameraSource ────────────────────────────────────────────────────────
class RemoteCameraSource:
    """
    Camera stream that polls face_api's HTTP frame bridge endpoint.

    This correctly crosses the process boundary:
      Process 1 (face_api):    camera_relay.py → WebSocket → _REMOTE_FRAME_IDS
                               GET /api/camera/frame/{cam_id}
      Process 2 (live_scorer): RemoteCameraSource polls that endpoint

    Interface matches CameraStream exactly:
        .read()   → (ret: bool, frame: np.ndarray | None, frame_id: int)
        .stopped  → bool
        .start()  → self
        .stop()   → None
    """

    POLL_INTERVAL   = 0.066   # ~15 fps polling
    STALE_TIMEOUT   = 30.0    # seconds without new frame → STALE
    OFFLINE_TIMEOUT = 60.0    # seconds without new frame → OFFLINE / stopped

    def __init__(self, cam_id: str, face_api_base: str = f"http://127.0.0.1:{os.environ.get('PORT', 5001)}"):
        self.cam_id         = cam_id
        self._base_url      = f"{face_api_base}/api/camera/frame/{cam_id}"
        self.stopped        = False

        self._frame:    "np.ndarray | None" = None
        self._frame_id: int   = 0
        self._lock      = threading.Lock()
        self._last_frame_id_seen: int = -1   # detect new frames vs stale
        self._last_frame_time:    float = 0.0
        self._thread: "threading.Thread | None" = None

    def start(self) -> "RemoteCameraSource":
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self.stopped = True

    def read(self):
        """Returns (ret, frame, frame_id) — same interface as CameraStream."""
        with self._lock:
            if self._frame is None or self.status in ("OFFLINE",):
                return False, None, self._frame_id
            return True, self._frame.copy(), self._frame_id

    @property
    def status(self) -> str:
        if self._last_frame_time == 0:
            return "WAITING"
        age = time.time() - self._last_frame_time
        if age > self.OFFLINE_TIMEOUT:
            return "OFFLINE"
        if age > self.STALE_TIMEOUT:
            return "STALE"
        return "LIVE"

    def _poll_loop(self):
        import urllib.request
        log_prefix = f"[REMOTE {self.cam_id}]"
        logged_state = None

        while not self.stopped:
            try:
                req = urllib.request.Request(self._base_url, method="GET")
                # Load session token for auth (mirrors how live_scorer calls face_api)
                try:
                    _secret_path = os.path.join(
                        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", ".session_secret"
                    )
                    if os.path.exists(_secret_path):
                        token = open(_secret_path).read().strip()
                        req.add_header("x-internal-token", token)
                except Exception:
                    pass

                with urllib.request.urlopen(req, timeout=3) as resp:
                    if resp.status == 204:
                        # Camera registered in face_api but no frames yet
                        if logged_state != "WAITING":
                            print(f"{log_prefix} Waiting for relay to connect...")
                            logged_state = "WAITING"
                        time.sleep(self.POLL_INTERVAL)
                        continue

                    server_frame_id = int(resp.headers.get("X-Frame-Id", 0))

                    # Only decode if this is a new frame (not the same one again)
                    if server_frame_id > self._last_frame_id_seen:
                        jpeg_bytes = resp.read()
                        nparr = np.frombuffer(jpeg_bytes, np.uint8)
                        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        if frame is not None:
                            with self._lock:
                                self._frame    = frame
                                self._frame_id += 1
                                self._last_frame_id_seen = server_frame_id
                                self._last_frame_time    = time.time()

                            if logged_state != "LIVE":
                                print(f"{log_prefix} ✓ Stream LIVE")
                                logged_state = "LIVE"
                    else:
                        # Same frame as before — camera may be paused or slow
                        resp.read()  # drain

                    new_status = self.status
                    if new_status != logged_state and new_status in ("STALE", "OFFLINE"):
                        print(f"{log_prefix} Stream {new_status}")
                        logged_state = new_status
                        if new_status == "OFFLINE":
                            self.stopped = True
                            break

            except Exception as e:
                if logged_state != "ERROR":
                    print(f"{log_prefix} Poll error: {e}")
                    logged_state = "ERROR"
                time.sleep(2.0)

            time.sleep(self.POLL_INTERVAL)


class CameraProcessor:
    def __init__(self, cam_id, src, face_det_model, enhancement_mode="AUTO", stream=None):
        self.cam_id = cam_id
        self.src = src
        # Accept a pre-built stream (e.g. VirtualCameraStream for remote cameras)
        if stream is not None:
            self.stream = stream
        else:
            self.stream = CameraStream(src)
        if self.stream.stopped:
            return
        self.stream.start()

        self.use_legacy = (INFERENCE_MODE == "legacy")
        if self.use_legacy:
            self.face_thread = FaceDetectorThread(face_det_model, conf=0.5)
            self.face_thread.start()
        else:
            global _global_central_worker
            if _global_central_worker is None:
                model_path = os.path.join(os.environ.get('SIH26187_MODELS', os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")), "yolo26n-face.pt")
                _global_central_worker = CentralInferenceWorker(model_path, batch_size=INFERENCE_BATCH_SIZE)
            self.face_thread = None
        
        self.enhancement_mode = enhancement_mode
        self._enhancer = AdaptiveEnhancer()
        
        self.frame_times = deque(maxlen=30)
        self.display_fps = 15.0
        self.last_processed_frame_id = -1
        
        # Output frame for rendering
        self.viz_frame = None
        import os; self.recording_enabled = os.environ.get('ENABLE_RECORDING', 'false').lower() == 'true'
        self.video_writer = None
        self.current_recording_id = None
        self.recording_start_time = None
        self.recording_frames = 0
        self.recording_path = None
        self.cam_label = getattr(self, 'label', self.cam_id)
        
    def _start_new_segment(self, w, h, fps):
        if not _storage:
            return
        self._close_segment()
        self.recording_id = str(uuid.uuid4())
        rec_dir = _storage.recordings_dir
        filename = f"{self.cam_id}_{self.recording_id}.mp4"
        self.recording_path = os.path.join(rec_dir, filename)
        
        # Use hardware-accelerated AVC1 (H.264) instead of CPU-heavy MP4V
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        self.video_writer = cv2.VideoWriter(self.recording_path, fourcc, fps, (w, h))
        self.recording_start_time = time.time()
        self.recording_frames = 0
        
        # Offload encoding to a background thread to prevent blocking the main pipeline
        import queue, threading
        self.record_queue = queue.Queue(maxsize=120)
        self.record_thread_active = True
        
        def _writer_thread():
            while self.record_thread_active or not self.record_queue.empty():
                try:
                    frm = self.record_queue.get(timeout=0.2)
                    if frm is not None and self.video_writer is not None:
                        self.video_writer.write(frm)
                    self.record_queue.task_done()
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"[Encoder] Thread error: {e}")
                    
        self.record_thread = threading.Thread(target=_writer_thread, daemon=True)
        self.record_thread.start()
        
        # Insert metadata
        try:
            _storage.start_recording(self.cam_id, self.cam_label, filename)
        except Exception as e:
            print(f"[CAM] DB Error starting recording: {e}")
            
    def _close_segment(self):
        if hasattr(self, 'record_thread_active'):
            self.record_thread_active = False
        if hasattr(self, 'record_thread') and self.record_thread:
            self.record_thread.join(timeout=1.0)
            self.record_thread = None
            
        if hasattr(self, 'video_writer') and self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            if self.recording_start_time and hasattr(self, 'recording_id') and _storage:
                duration = time.time() - self.recording_start_time
                size = os.path.getsize(self.recording_path) if os.path.exists(self.recording_path) else 0
                end_time = datetime.datetime.now().isoformat()
                try:
                    _storage.finish_recording(self.recording_id, end_time, duration, size)
                except Exception as e:
                    print(f"[CAM] DB Error finishing recording: {e}")
            self.recording_start_time = None

    def process_frame(self) -> None:
        if self.stream.stopped:
            return

        # Lifecycle guard: pause processing for STALE/OFFLINE remote cameras
        # to avoid spinning the AI pipeline on a frozen last-frame
        if isinstance(self.stream, RemoteCameraSource):
            status = self.stream.status
            if status == "OFFLINE":
                self.stream.stop()
                return
            if status in ("STALE", "WAITING"):
                time.sleep(0.1)
                return

        ret, frame, frame_id = self.stream.read()
        if not ret or frame is None:
            return

        if frame_id == self.last_processed_frame_id:
            return
        self.last_processed_frame_id = frame_id

        # Copy and flip
        frame = cv2.flip(frame, 1).copy()
        
        # Phase 3: Intelligent Scheduling
        import os
        from collections import deque
        INFERENCE_FPS_CAP = int(os.environ.get('INFERENCE_FPS_CAP', '10'))
        
        now_ts = time.time()
        do_inference = True
        
        if not hasattr(self, 'last_yolo_time'):
            self.last_yolo_time = 0
            self.inference_times = deque(maxlen=30)
            self.inference_fps = 0.0
            
        if INFERENCE_FPS_CAP > 0:
            if now_ts - self.last_yolo_time < (1.0 / INFERENCE_FPS_CAP):
                do_inference = False
            else:
                self.last_yolo_time = now_ts
                self.inference_times.append(now_ts)
                if len(self.inference_times) > 1:
                    self.inference_fps = len(self.inference_times) / (self.inference_times[-1] - self.inference_times[0])

        if do_inference:
            # Adaptive Enhancement ONLY on the inference frame to save massive CPU load on 1080p video
            enhanced_frame = self._enhancer.enhance(frame, self.enhancement_mode)
            if self.use_legacy:
                self.face_thread.submit(enhanced_frame)
            else:
                _global_central_worker.submit(self.cam_id, enhanced_frame)

        # FPS tracking
        now = time.time()
        self.frame_times.append(now)
        if len(self.frame_times) >= 10 and len(self.frame_times) % 10 == 0:
            recent = list(self.frame_times)
            self.display_fps = max(1.0, min(60.0, 9.0 / (recent[-1] - recent[0])))
            
        # Telemetry Logging
        if not hasattr(self, 'last_telemetry_print'):
            self.last_telemetry_print = time.time()
        
        if time.time() - self.last_telemetry_print > 5.0:
            self.last_telemetry_print = time.time()
            capture_fps = getattr(self.stream, 'capture_fps', 0.0)
            dropped = getattr(self.stream, 'dropped_frames', 0)
            print(f"[TELEMETRY CAM {self.cam_id}] Capture: {capture_fps:.1f}fps | Inference: {self.inference_fps:.1f}fps | Display: {self.display_fps:.1f}fps | Dropped: {dropped}")

        faces = self.face_thread.result if self.use_legacy else _global_central_worker.get_result(self.cam_id)

        # HUD Overlay
        viz = frame.copy()
        w_f, h_f = frame.shape[1], frame.shape[0]
        pw, ph = min(500, w_f), min(120, h_f)
        region = viz[0:ph, 0:pw].copy()
        dark   = np.full_like(region, (8, 10, 16))
        viz[0:ph, 0:pw] = cv2.addWeighted(dark, 0.78, region, 0.22, 0)

        cv2.putText(viz, f"CAM {self.cam_id} - LIVE", (15, 30), 0, 0.8, (0, 255, 255), 2)
        status_text = f"FACES: {len(faces)}"
        color = (0, 255, 0) if len(faces) > 0 else (0, 0, 255)
        cv2.putText(viz, status_text, (15, 65), 0, 0.6, color, 2)
        
        for face in faces:
            tid = face['track_id']
            x1, y1, x2, y2 = map(int, face['box'])
            cv2.rectangle(viz, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(viz, f"ID: {tid}", (x1, max(0, y1 - 10)), 0, 0.6, (0, 255, 0), 2)
            
            # Send to API in a background thread to not block the live feed
            def send_to_api(face_frame, t_id):
                import requests
                _, img_encoded = cv2.imencode('.jpg', face_frame)
                files = {'image': ('frame.jpg', img_encoded.tobytes(), 'image/jpeg')}
                data = {
                    'camera_id': self.cam_id, 
                    'track_id': t_id,
                    'enhancement_mode': self.enhancement_mode
                }
                try:
                    headers={"x-internal-token": open("/Users/hariharans/Documents/SIH26187/data/.session_secret").read().strip()} if __import__("os").path.exists("/Users/hariharans/Documents/SIH26187/data/.session_secret") else {}
                    API_PORT = int(os.environ.get("PORT", 5001))
                    res = requests.post(f"http://localhost:{API_PORT}/api/recognize_live", files=files, data=data, headers=headers, timeout=2.0)
                    if res.status_code == 200:
                        rj = res.json()
                        status = rj.get('match_status')
                        if status == 'MATCH':
                            print(f"[CAM {self.cam_id}] Track {t_id} MATCH: {rj.get('matched_identity_id')}")
                        elif status == 'UNKNOWN':
                            print(f"[CAM {self.cam_id}] Track {t_id} UNKNOWN")
                except Exception as e:
                    pass
            
            # Extract crop to send. Expanding bbox slightly could be good, but we send the face
            # We could just send the full frame or crop. The user said: 
            # "live_scorer.py -> face bbox + camera_id + track_id"
            # We will send the full frame and let face_service detect it to keep it simple,
            # or send a crop to save bandwidth. Sending full frame with bbox metadata is safer for alignment.
            # For MVP bandwidth, let's send the full frame. We throttle to 1 request per second per track.
            
            # Throttle requests
            if not hasattr(self, 'last_api_send'):
                self.last_api_send = {}
            if now - self.last_api_send.get(tid, 0) > 1.0:
                self.last_api_send[tid] = now
                threading.Thread(target=send_to_api, args=(frame.copy(), tid), daemon=True).start()

        cv2.putText(viz, f"{self.display_fps:.1f} FPS", (w_f - 180, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 1, cv2.LINE_AA)

        self.viz_frame = viz
        self.has_new_viz = True
        
        # Handle Recording
        if self.recording_enabled and _storage:
            w_f, h_f = frame.shape[1], frame.shape[0]
            fps = self.display_fps if self.display_fps > 0 else 15.0
            
            if self.video_writer is None:
                self._start_new_segment(w_f, h_f, fps)
                
            if self.video_writer is not None and hasattr(self, 'record_queue'):
                try:
                    self.record_queue.put_nowait(viz.copy())
                    self.recording_frames += 1
                except Exception:
                    pass # Drop frame if recording IO is completely saturated
                
                # Close segment after ~45 seconds
                if time.time() - self.recording_start_time > 45.0:
                    self._close_segment()

    def stop(self):
        self._close_segment()
        if hasattr(self, 'face_thread') and self.face_thread:
            self.face_thread.stop()
        self.stream.stop()

# ── Remote Camera IPC State ────────────────────────────────────────────────    
_CAMERA_STATE_FILE = os.environ.get(
    "CAMERAS_STATE_PATH",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "camera_state.json")
)

def _is_system_active() -> bool:
    if not os.path.exists(_CAMERA_STATE_FILE):
        return False
    try:
        import json
        with open(_CAMERA_STATE_FILE, "r") as f:
            data = json.load(f)
            return data.get("active", False)
    except Exception:
        return False

# ── Camera config from cameras.json ───────────────────────────────────────────
_CAMERAS_JSON = os.environ.get(
    "CAMERAS_JSON_PATH",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cameras.json")
)
_CAM_RELOAD_INTERVAL = 2.0   # seconds between hot-reload checks
_CAM_RETRY_DELAYS    = [2, 5, 10, 30]  # back-off seconds on stream failure


def _load_cameras_json() -> list:
    """Read cameras.json safely. Returns [] on any error."""
    try:
        with open(_CAMERAS_JSON, 'r') as f:
            data = json.load(f)
        return [c for c in data if isinstance(c, dict) and c.get('enabled', True)]
    except Exception:
        return []


def _env_camera_sources() -> list:
    """Fallback: read CAMERA_0/1/2 from env vars (backward compat)."""
    srcs = []
    for key in ("CAMERA_0", "CAMERA_1", "CAMERA_2"):
        val = os.environ.get(key, "DISABLED")
        if val.strip().upper() != "DISABLED":
            srcs.append({"id": key.lower(), "source": val.strip(), "label": key, "enabled": True})
    return srcs


def _get_active_cameras() -> list:
    """
    Return the list of enabled camera config dicts to use right now.
    Prefers cameras.json; falls back to CAMERA_x env vars.
    """
    configs = _load_cameras_json()
    if not configs:
        configs = _env_camera_sources()
    return configs


def _start_processor(cam_cfg: dict, face_det) -> "CameraProcessor | None":
    """
    Attempt to start a CameraProcessor for a config entry.
    Returns None (and logs) if the stream cannot be opened.

    Special sources:
      "ws://remote" or "wss://remote" -> uses VirtualCameraStream (remote relay)
      numeric string, rtsp://, http:// -> uses CameraStream (local camera)
    """
    source_str = str(cam_cfg['source'])
    label = cam_cfg.get('label', cam_cfg['id'])
    cam_id = cam_cfg['id']
    mode = cam_cfg.get('enhancement_mode', 'AUTO')

    # -- Remote camera via WebSocket relay ------------------------------------
    # Source "ws://remote" or "wss://..." means this camera's frames arrive via
    # camera_relay.py -> face_api WebSocket -> HTTP frame bridge.
    # We use RemoteCameraSource which polls http://localhost:{os.environ.get('PORT', 5001)}/api/camera/frame/{cam_id}
    # This correctly crosses the process boundary without shared memory.
    if source_str.startswith(("ws://", "wss://")):
        print(f"  [CAM] Remote camera (WebSocket relay): {label} ({cam_id})")
        rstream = RemoteCameraSource(cam_id)
        proc = CameraProcessor(cam_id, source_str, face_det, enhancement_mode=mode, stream=rstream)
        print(f"  [CAM] ✓ Remote camera registered: {label} — waiting for relay to connect")
        return proc

    # -- Local camera --------------------------------------------------------
    src = _parse_camera_source(source_str)
    print(f"  [CAM] Connecting → {label} ({src}) [Mode: {mode}] ...")
    proc = CameraProcessor(cam_id, src, face_det, enhancement_mode=mode)
    if proc.stream.stopped:
        print(f"  [CAM] ✗ Failed to open {label} ({src})")
        return None
    print(f"  [CAM] ✓ Connected: {label}")
    return proc


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("--- Initializing Multi-Camera Auto-Recording Pipeline ---")

    print("Loading YOLO models (this may take a few seconds)...")
    try:
        face_det = YOLO(os.path.join(os.environ.get('SIH26187_MODELS', os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")), "yolo26n-face.pt"))
        print("  ✓ Face model loaded")
    except Exception as e:
        face_det = None
        print(f"  ⚠ Face model unavailable: {e}")
        sys.exit(1)

    # ── Initial camera load ─────────────────────────────────────────────────
    active_cfg   = _get_active_cameras()
    processors: dict[str, "CameraProcessor"] = {}   # cam_id → processor
    retry_times: dict[str, float]            = {}   # cam_id → next retry timestamp
    retry_counts: dict[str, int]             = {}   # cam_id → number of retries
    _connecting_cams = set()

    def _sync_processors(configs: list):
        """
        Reconcile running processors with the desired config list.
        Starts new cameras, stops removed ones. Modifies dict in-place.
        """
        desired_ids = {c['id'] for c in configs}

        # Stop processors for removed cameras
        removed_ids = set(processors.keys()) - desired_ids
        for cid in removed_ids:
            print(f"  [CAM] Stopping removed camera: {cid}")
            try:
                processors[cid].stop()
            except Exception as e:
                print(f"  [CAM] Warning during stop of {cid}: {e}")
            processors.pop(cid, None)
            retry_times.pop(cid, None)
            retry_counts.pop(cid, None)

        # Start processors for new cameras (not already running)
        for cam_cfg in configs:
            cid = cam_cfg['id']
            if cid in processors or cid in _connecting_cams:
                continue  # already running or currently connecting
            now = time.time()
            if cid in retry_times and now < retry_times[cid]:
                continue  # still in back-off
                
            _connecting_cams.add(cid)
            
            def _connect_worker(c_cfg, c_id):
                proc = _start_processor(c_cfg, face_det)
                if proc:
                    processors[c_id] = proc
                    retry_times.pop(c_id, None)
                    retry_counts.pop(c_id, None)
                else:
                    count = retry_counts.get(c_id, 0)
                    delay = _CAM_RETRY_DELAYS[min(count, len(_CAM_RETRY_DELAYS) - 1)]
                    retry_counts[c_id] = count + 1
                    retry_times[c_id] = time.time() + delay
                    print(f"  [CAM] Will retry {c_id} in {delay}s")
                _connecting_cams.remove(c_id)
                
            threading.Thread(target=_connect_worker, args=(cam_cfg, cid), daemon=True).start()
            time.sleep(1.5) # Stagger initialization to prevent USB bus lockup


    print("\n--- Multi-Camera System Ready (IDLE) ---")
    print(f"Awaiting Start signal from Dashboard http://localhost:{os.environ.get('PORT', 5001)}/")

    last_reload = 0.0
    system_active = False

    try:
        while True:
            target_active = _is_system_active()
            
            if not target_active:
                if system_active:
                    print("\n  [SYSTEM] Remote stop signal received. Shutting down hardware...")
                    for proc in processors.values():
                        proc.stop()
                    processors.clear()
                    retry_times.clear()
                    retry_counts.clear()

                    system_active = False
                    print("  [SYSTEM] Hardware released. Entering IDLE state.")
                    time.sleep(1.0) # Pause an extra beat to let hardware catch up
                

                
                time.sleep(0.5)
                continue
                
            else:
                if not system_active:
                    print("\n  [SYSTEM] Remote start signal received. Allocating hardware...")

                    system_active = True
                    last_reload = 0 # Force instant sync of cameras.json
                
            # ── Hot-reload cameras.json every _CAM_RELOAD_INTERVAL seconds ──
            now = time.time()
            if now - last_reload >= _CAM_RELOAD_INTERVAL:  # type: ignore
                new_cfg = _get_active_cameras()
                _sync_processors(new_cfg)
                last_reload = now

            # ── Check for crashed streams and attempt recovery ───────────────
            for cid, proc in list(processors.items()):
                if proc.stream.stopped:
                    print(f"  [CAM] Stream {cid} stopped unexpectedly. Scheduling retry...")
                    proc.stop()
                    processors.pop(cid, None)
                    retry_times[cid] = time.time() + _CAM_RETRY_DELAYS[0]

            # ── Process frames (Threaded for parallel multi-cam) ─────────
            def _process_one(proc):
                proc.process_frame()
                if getattr(proc, 'has_new_viz', False) and proc.viz_frame is not None:
                    proc.has_new_viz = False
                    if LAZY_ENCODING:
                        return proc.cam_id, proc.viz_frame
                    else:
                        _, buffer = cv2.imencode('.jpg', proc.viz_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        return proc.cam_id, buffer.tobytes()
                return None, None

            # Async encoder pool for event buffers to prevent memory bloat
            import queue
            import concurrent.futures
            
            if not hasattr(_sync_processors, 'enc_queue'):
                _sync_processors.enc_queue = queue.Queue(maxsize=15)
                
                def _encoder_worker():
                    while True:
                        try:
                            cid, frm, ts = _sync_processors.enc_queue.get()
                            if isinstance(frm, np.ndarray):
                                h, w = frm.shape[:2]
                                if w > 1280:
                                    frm = cv2.resize(frm, (1280, int(1280 * h / w)))
                                _, buf = cv2.imencode('.jpg', frm, [cv2.IMWRITE_JPEG_QUALITY, 70])
                                frm = buf.tobytes()
                            if cid not in _STREAM_BUFFERS:
                                _STREAM_BUFFERS[cid] = deque(maxlen=450)
                            _STREAM_BUFFERS[cid].append((ts, frm))
                            _sync_processors.enc_queue.task_done()
                        except Exception as e:
                            print(f"[Encoder] Error: {e}")
                            
                for _ in range(4):
                    threading.Thread(target=_encoder_worker, daemon=True).start()

            if not hasattr(_sync_processors, 'executor'):
                _sync_processors.executor = concurrent.futures.ThreadPoolExecutor(max_workers=16)
                
            results = _sync_processors.executor.map(_process_one, list(processors.values()))

            for cam_id, frame_data in results:
                if cam_id and frame_data is not None:
                    # Update live stream immediately (raw numpy array if LAZY, else bytes)
                    ver = _STREAM_VERSIONS.get(cam_id, 0) + 1
                    _STREAM_VERSIONS[cam_id] = ver
                    _STREAM_FRAMES[cam_id] = (ver, frame_data)
                    
                    # Offload the buffer encoding asynchronously using bounded queue
                    try:
                        _sync_processors.enc_queue.put_nowait((cam_id, frame_data, time.time()))
                    except queue.Full:
                        print(f"  [CAM] Encoder queue full (dropping frame for {cam_id} to save memory)")
            
            # Keep CPU from spinning too fast if no frames
            time.sleep(0.01)

    except KeyboardInterrupt:
        pass
    finally:
        for proc in processors.values():
            proc.stop()

if __name__ == "__main__":
    import json  # needed for _load_cameras_json

    # Run preflight checks
    if not _preflight():
        print("[FATAL] Preflight checks failed. Fix issues above and retry.")
        sys.exit(1)

    main()

