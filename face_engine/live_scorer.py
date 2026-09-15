import os
import sys
import cv2  # type: ignore
import time
import subprocess
import threading
import queue
import datetime
import uuid
import sys
import os

# Add face_api to sys path to import Storage
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from face_api.app.core.storage import Storage
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
_STREAM_FRAMES = {}  # cam_id -> bytes
_STREAM_BUFFERS = {} # cam_id -> deque of (timestamp, bytes)
_stream_app = FastAPI()

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
    return Response(content=best_frame, media_type="image/jpeg")

@_stream_app.get("/video_feed/{cam_id}")
async def video_feed(cam_id: str):
    def generate():
        while True:
            frame = _STREAM_FRAMES.get(cam_id)
            if frame is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.04)  # ~25 fps
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

threading.Thread(target=lambda: uvicorn.run(_stream_app, host="127.0.0.1", port=5002, log_level="warning"), daemon=True).start()

# ── Preflight Validation ──────────────────────────────────────────────────────
def _preflight():
    """
    Validate all critical dependencies before starting the pipeline.
    Returns True if all required checks pass.
    """
    _ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))
    _MODELS_DIR = os.path.join(_ENGINE_DIR, "models")

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
    checks.append(("✓" if cam_ok else "✗", "Camera 0", cam_detail))

    # 3. API health
    api_ok = False
    api_detail = "NOT RESPONDING"
    try:
        import urllib.request
        with urllib.request.urlopen("http://localhost:5001/api/health", timeout=3) as resp:
            if resp.status == 200:
                import json as _json
                data = _json.loads(resp.read())
                api_ok = True
                api_detail = f"200 OK, {data.get('person_count', 0)} persons"
    except Exception:
        pass
    checks.append(("✓" if api_ok else "✗", "API health", api_detail))

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
        self.src = src
        # For local USB/built-in cameras (integer index), explicitly use the
        # AVFoundation backend on macOS to prevent the OBSENSOR (Orbbec depth
        # camera) backend from intercepting the device and failing.
        if isinstance(src, int):
            self.stream = cv2.VideoCapture(src, cv2.CAP_AVFOUNDATION)
        else:
            self.stream = cv2.VideoCapture(src)
        if not self.stream.isOpened():
            print(f"Failed to open {src}.")
            self.stopped = True
            return

        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        # Keep OpenCV's internal buffer as small as possible so we always
        # get the newest frame rather than reading stale buffered frames.
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Initial read — retry a few times with a short delay.
        # Some USB cameras (especially when multiple are opened in quick succession)
        # need a warm-up period before the first frame is available.
        self.ret = False
        self.frame = None
        for _warmup in range(10):
            self.ret, self.frame = self.stream.read()
            if self.ret:
                break
            time.sleep(0.1)

        self.stopped = False
        self.frame_id = 0
        self._lock = threading.Lock()
        self._thread: "threading.Thread | None" = None

    def start(self):
        self._thread = threading.Thread(target=self._update, daemon=True)  # type: ignore
        self._thread.start()  # type: ignore
        return self

    def _update(self):
        """Drain OpenCV's internal network buffer with grab() in a tight loop.
        We only decode (retrieve) when the main thread actually wants a frame.
        This keeps the buffer empty so read() always returns the freshest frame."""
        while not self.stopped:
            # grab() signals the camera / advances the buffer pointer (cheap)
            grabbed = self.stream.grab()
            if not grabbed:
                time.sleep(0.005)
                continue
            # Decode the grabbed frame and expose it to the main thread
            ret, frame = self.stream.retrieve()
            if ret:
                with self._lock:
                    self.ret = ret
                    self.frame = frame
                    self.frame_id += 1
            
            # Throttle if reading from a local video file to simulate real-time stream
            if isinstance(self.src, str) and self.src.lower().endswith(('.mp4', '.avi', '.mov')):
                time.sleep(1/30.0)
                    
        # Safely release the camera inside the thread that owns it to prevent Segfaults
        self.stream.release()

    def read(self):
        with self._lock:
            return self.ret, self.frame, self.frame_id

    def stop(self):
        self.stopped = True
        if hasattr(self, '_thread') and self._thread.is_alive():  # type: ignore
            self._thread.join(timeout=2.0)  # type: ignore


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

    def __init__(self, cam_id: str, face_api_base: str = "http://127.0.0.1:5001"):
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

        self.face_thread = FaceDetectorThread(face_det_model, conf=0.5)
        self.face_thread.start()
        
        self.enhancement_mode = enhancement_mode
        self._enhancer = AdaptiveEnhancer()
        
        self.frame_times = deque(maxlen=30)
        self.display_fps = 15.0
        self.last_processed_frame_id = -1
        
        # Output frame for rendering
        self.viz_frame = None
        self.recording_enabled = True # TODO: Read from config
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
        
        # Use OpenCV VideoWriter (MP4V)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(self.recording_path, fourcc, fps, (w, h))
        self.recording_start_time = time.time()
        self.recording_frames = 0
        
        # Insert metadata
        try:
            _storage.start_recording(self.cam_id, self.cam_label, filename)
        except Exception as e:
            print(f"[CAM] DB Error starting recording: {e}")
            
    def _close_segment(self):
        if self.video_writer:
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
        
        # Adaptive Enhancement (Processing Branch)
        enhanced_frame = self._enhancer.enhance(frame, self.enhancement_mode)
        
        # We always prefer latest frame for inference
        self.face_thread.submit(enhanced_frame)

        # FPS tracking
        now = time.time()
        self.frame_times.append(now)
        if len(self.frame_times) >= 10 and len(self.frame_times) % 10 == 0:
            recent = list(self.frame_times)
            self.display_fps = max(1.0, min(60.0, 9.0 / (recent[-1] - recent[0])))

        faces = self.face_thread.result

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
                    res = requests.post("http://localhost:5001/api/recognize_live", files=files, data=data, headers=headers, timeout=2.0)
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
        
        # Handle Recording
        if self.recording_enabled and _storage:
            w_f, h_f = frame.shape[1], frame.shape[0]
            fps = self.display_fps if self.display_fps > 0 else 15.0
            
            if self.video_writer is None:
                self._start_new_segment(w_f, h_f, fps)
                
            if self.video_writer is not None:
                self.video_writer.write(viz)
                self.recording_frames += 1
                
                # Close segment after ~45 seconds
                if time.time() - self.recording_start_time > 45.0:
                    self._close_segment()

    def stop(self):
        self._close_segment()
        self.face_thread.stop()
        self.stream.stop()

# ── Remote Camera IPC State ────────────────────────────────────────────────────
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
_CAM_RELOAD_INTERVAL = 10.0   # seconds between hot-reload checks
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
    # We use RemoteCameraSource which polls http://localhost:5001/api/camera/frame/{cam_id}
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
        face_det = YOLO(os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "yolo26n-face.pt"))
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
            if cid in processors:
                continue  # already running
            now = time.time()
            if cid in retry_times and now < retry_times[cid]:
                continue  # still in back-off
            proc = _start_processor(cam_cfg, face_det)
            if proc:
                processors[cid] = proc
                retry_times.pop(cid, None)
                retry_counts.pop(cid, None)
            else:
                # Back-off: pick next delay based on how many retries done
                count = retry_counts.get(cid, 0)
                delay = _CAM_RETRY_DELAYS[min(count, len(_CAM_RETRY_DELAYS) - 1)]
                retry_counts[cid] = count + 1
                retry_times[cid] = time.time() + delay
                print(f"  [CAM] Will retry {cid} in {delay}s")


    print("\n--- Multi-Camera System Ready (IDLE) ---")
    print("Awaiting Start signal from Dashboard http://localhost:5001/")

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

            # ── Process frames ───────────────────────────────────────────────
            for proc in processors.values():
                proc.process_frame()
                if proc.viz_frame is not None:
                    _, buffer = cv2.imencode('.jpg', proc.viz_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    frame_bytes = buffer.tobytes()
                    _STREAM_FRAMES[proc.cam_id] = frame_bytes
                    if proc.cam_id not in _STREAM_BUFFERS:
                        _STREAM_BUFFERS[proc.cam_id] = deque(maxlen=900)
                    _STREAM_BUFFERS[proc.cam_id].append((time.time(), frame_bytes))
            
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

