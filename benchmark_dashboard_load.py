import os, sys, time, json, statistics, concurrent.futures, psutil, threading, requests
import cv2

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE, "face_engine"))

os.environ["INFERENCE_MODE"] = "mps_batch"
os.environ["INFERENCE_BATCH_SIZE"] = "1"
os.environ["OPTIMIZE_SCHEDULING"] = "true"
os.environ["LAZY_ENCODING"] = "true"

from ultralytics import YOLO
from live_scorer import CameraStream, CameraProcessor, CentralInferenceWorker, _STREAM_FRAMES

_METRICS = {
    "rtsp_connect_times": [],
    "grab_intervals": {},
    "decode_latency": [],
    "ai_queue_depths": [],
    "inference_latency": [],
    "ai_inference_count": 0,
    "end_to_end_latency": [],
    "frame_drops": {},
    "cpu_timeseries": [],
    "ram_timeseries": [],
    "jpeg_encode_latency": [],
    "jpeg_encode_count": 0,
    "encoded_frame_ids": [],
    "client_fps": [],
    "client_failures": 0,
    "client_disconnects": 0,
    "http_errors": 0
}

# --- Monkeypatching ---

# 1. cv2.imencode (Dashboard JPEG tracking)
original_imencode = cv2.imencode
def patched_imencode(ext, img, params=None):
    start = time.time()
    res = original_imencode(ext, img, params)
    _METRICS["jpeg_encode_latency"].append(time.time() - start)
    _METRICS["jpeg_encode_count"] += 1
    _METRICS["encoded_frame_ids"].append(id(img))
    return res
cv2.imencode = patched_imencode

# 2. CameraStream (Capture FPS, Decode Latency, Capture Timestamp)
original_camerastream_update = CameraStream._update
def patched_camerastream_update(self):
    self._last_grab_time = time.time()
    self._capture_timestamps = {}
    
    while not self.stopped:
        grabbed = self.stream.grab()
        now = time.time()
        if grabbed:
            cid_key = getattr(self, 'cam_id', id(self))
            _METRICS["grab_intervals"].setdefault(cid_key, []).append(now - self._last_grab_time)
            self._last_grab_time = now
            
            d_start = time.time()
            ret, frame = self.stream.retrieve()
            if ret:
                _METRICS["decode_latency"].append(time.time() - d_start)
                with self._lock:
                    self.ret = ret
                    self.frame = frame
                    self.frame_id += 1
                    self._capture_timestamps[self.frame_id] = now
                    
                    # Monkeypatch live_scorer's STREAM_FRAMES directly here so dashboard gets it
                    _STREAM_FRAMES[cid_key] = (self.frame_id, frame)
        else:
            if isinstance(self.src, str):
                self.stream.release()
                time.sleep(0.5)
                self.stream = cv2.VideoCapture(self.src)
            else:
                time.sleep(0.01)
CameraStream._update = patched_camerastream_update

# 3. CameraProcessor (Frame Drops)
original_process_frame = CameraProcessor.process_frame
def patched_process_frame(self):
    ret, frame, fid = self.stream.read()
    if ret:
        last_fid = getattr(self, '_last_fid', fid - 1)
        if fid > last_fid + 1:
            _METRICS["frame_drops"].setdefault(self.cam_id, 0)
            _METRICS["frame_drops"][self.cam_id] += (fid - last_fid - 1)
        self._last_fid = fid
        
        capture_time = getattr(self.stream, '_capture_timestamps', {}).get(fid, time.time())
        _METRICS.setdefault('capture_times_by_cam', {})[self.cam_id] = capture_time
        
    original_process_frame(self)
CameraProcessor.process_frame = patched_process_frame

# 4. CentralInferenceWorker (Queue depth, E2E Latency, Inference Latency)
original_worker_run = CentralInferenceWorker._run
def patched_worker_run(self):
    while not self._stopped:
        batch = []
        cam_ids = []
        capture_times = []
        
        with self.frames_lock:
            q_depth = sum(1 for f in self.latest_frames.values() if f is not None)
            _METRICS["ai_queue_depths"].append(q_depth)
            
            for cid, frame in list(self.latest_frames.items()):
                if frame is not None:
                    batch.append(frame)
                    cam_ids.append(cid)
                    capture_times.append(_METRICS.get('capture_times_by_cam', {}).get(cid, time.time()))
                    self.latest_frames[cid] = None
                if len(batch) >= self.batch_size:
                    break
                    
        if not batch:
            time.sleep(0.01)
            continue
            
        inf_start = time.time()
        try:
            out = self.model(batch, verbose=False, conf=0.5, device=self.device)
            inf_end = time.time()
            _METRICS["inference_latency"].append(inf_end - inf_start)
            _METRICS["ai_inference_count"] += len(batch)
            
            for ct in capture_times:
                _METRICS["end_to_end_latency"].append(inf_end - ct)
                
            for i, res in enumerate(out):
                self.results[cam_ids[i]] = []
        except Exception as e:
            pass
            
CentralInferenceWorker._run = patched_worker_run

# --- Resource Monitor Thread ---
_monitor_stop = False
def resource_monitor():
    process = psutil.Process(os.getpid())
    while not _monitor_stop:
        _METRICS["cpu_timeseries"].append(process.cpu_percent())
        _METRICS["ram_timeseries"].append(process.memory_info().rss / (1024 * 1024))
        time.sleep(1.0)
monitor_thread = threading.Thread(target=resource_monitor, daemon=True)
monitor_thread.start()

# --- Dashboard Client Simulator ---
_clients_stop = False
def mjpeg_client(cam_id):
    try:
        resp = requests.get(f"http://127.0.0.1:5002/video_feed/{cam_id}", stream=True, timeout=5)
        if resp.status_code != 200:
            _METRICS["http_errors"] += 1
            return
            
        frames_received = 0
        start_time = time.time()
        for line in resp.iter_lines():
            if _clients_stop: break
            if b"Content-Type: image/jpeg" in line:
                frames_received += 1
                
        elapsed = time.time() - start_time
        if elapsed > 0:
            _METRICS["client_fps"].append(frames_received / elapsed)
            
        _METRICS["client_disconnects"] += 1
    except requests.exceptions.RequestException:
        _METRICS["client_failures"] += 1

# --- Benchmarking Logic ---
print("Initializing FaceDetection...")
model_path = os.path.join(BASE, "face_engine", "models", "yolo26n-face.pt")
face_det = YOLO(model_path)

# Ensure Uvicorn has time to start (from live_scorer.py)
time.sleep(2)

def reset_metrics():
    for k in ["decode_latency", "ai_queue_depths", "inference_latency", "end_to_end_latency", "cpu_timeseries", "ram_timeseries", "jpeg_encode_latency", "encoded_frame_ids", "client_fps"]:
        _METRICS[k].clear()
    _METRICS["ai_inference_count"] = 0
    _METRICS["jpeg_encode_count"] = 0
    _METRICS["grab_intervals"].clear()
    _METRICS["frame_drops"].clear()
    _METRICS["client_failures"] = 0
    _METRICS["client_disconnects"] = 0
    _METRICS["http_errors"] = 0

def aggregate_stats(duration):
    ai_fps = _METRICS["ai_inference_count"] / duration
    avg_inf_lat = sum(_METRICS["inference_latency"]) / max(1, len(_METRICS["inference_latency"])) * 1000
    avg_e2e = sum(_METRICS["end_to_end_latency"]) / max(1, len(_METRICS["end_to_end_latency"])) * 1000
    depths = _METRICS["ai_queue_depths"]
    avg_queue = sum(depths) / max(1, len(depths))
    total_drops = sum(_METRICS["frame_drops"].values())
    cpus = _METRICS["cpu_timeseries"]
    rams = _METRICS["ram_timeseries"]
    
    from live_scorer import _SHARED_ENCODERS, _ENCODERS_LOCK
    with _ENCODERS_LOCK:
        total_unique = sum(enc.unique_frames for enc in _SHARED_ENCODERS.values())
        total_encodes = sum(enc.jpeg_encodes for enc in _SHARED_ENCODERS.values())
        total_clients = sum(enc.clients_served for enc in _SHARED_ENCODERS.values())
    
    unique_encoded_frames = total_unique
    duplicate_encodes = total_encodes - total_unique
    _METRICS["jpeg_encode_count"] = total_encodes
    
    # Reset internal metrics for the next phase
    with _ENCODERS_LOCK:
        for enc in _SHARED_ENCODERS.values():
            enc.jpeg_encodes = 0
            enc.unique_frames = 0
            enc.clients_served = 0
            enc.stale_skips = 0
    avg_jpeg_lat = sum(_METRICS["jpeg_encode_latency"]) / max(1, len(_METRICS["jpeg_encode_latency"])) * 1000
    avg_client_fps = sum(_METRICS["client_fps"]) / max(1, len(_METRICS["client_fps"]))
    
    return {
        "total_frame_drops": total_drops,
        "ai_inference_fps": round(ai_fps, 2),
        "avg_inference_latency_ms": round(avg_inf_lat, 2),
        "avg_e2e_latency_ms": round(avg_e2e, 2),
        "avg_ai_queue_depth": round(avg_queue, 2),
        "cpu_avg_percent": round(sum(cpus)/max(1, len(cpus)), 1) if cpus else 0,
        "ram_avg_mb": round(sum(rams)/max(1, len(rams)), 1) if rams else 0,
        "jpeg_encode_count": _METRICS["jpeg_encode_count"],
        "unique_frames_encoded": unique_encoded_frames,
        "duplicate_encodes": duplicate_encodes,
        "avg_jpeg_encode_latency_ms": round(avg_jpeg_lat, 2),
        "avg_client_delivered_fps": round(avg_client_fps, 2),
        "client_failures": _METRICS["client_failures"],
        "http_errors": _METRICS["http_errors"]
    }

def benchmark_mode(num_cams, clients_per_cam, duration=20):
    global _clients_stop
    print(f"--- Benchmarking {num_cams} Cams x {clients_per_cam} Clients ---")
    
    processors = {}
    for i in range(num_cams):
        cid = f"cam_{i}"
        src = f"rtsp://localhost:8554/cam{i+1}"
        proc = CameraProcessor(cid, src, face_det)
        proc.stream.cam_id = cid
        processors[cid] = proc
        
    print("Warming up for 5 seconds...")
    warmup_start = time.time()
    while time.time() - warmup_start < 5:
        for proc in processors.values(): proc.process_frame()
        time.sleep(0.01)
        
    # Baseline phase (No clients)
    print("Running baseline (no clients)...")
    reset_metrics()
    start_time = time.time()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=min(num_cams, 6))
    def _process_one(proc): proc.process_frame()
    while time.time() - start_time < duration:
        list(executor.map(_process_one, processors.values()))
    baseline_stats = aggregate_stats(duration)
    
    # Dashboard phase
    print(f"Running loaded ({clients_per_cam} clients/cam)...")
    reset_metrics()
    _clients_stop = False
    
    client_threads = []
    if clients_per_cam > 0:
        for i in range(num_cams):
            for _ in range(clients_per_cam):
                t = threading.Thread(target=mjpeg_client, args=(f"cam_{i}",))
                t.start()
                client_threads.append(t)
    
    time.sleep(2) # Let clients connect
    start_time = time.time()
    reset_metrics() # Reset again after clients connect to measure purely under load
    
    while time.time() - start_time < duration:
        list(executor.map(_process_one, processors.values()))
        
    _clients_stop = True
    for t in client_threads: t.join()
    
    loaded_stats = aggregate_stats(duration)
    
    for proc in processors.values():
        proc.stream.stop()
        
    return {
        "scenario": f"{num_cams}x{clients_per_cam}",
        "num_cameras": num_cams,
        "clients_per_cam": clients_per_cam,
        "baseline_no_clients": baseline_stats,
        "loaded_with_clients": loaded_stats
    }

print("Running Mock Dashboard Load Tests...")
scenarios = [
    (1, 1),
    (1, 5),
    (1, 10),
    (12, 1),
    (12, 3) # Using 3 clients per cam for 12 cameras = 36 concurrent HTTP streams, to avoid saturating loopback socket limits on macOS immediately.
]

results = []
for cams, clients in scenarios:
    res = benchmark_mode(cams, clients)
    print(res)
    results.append(res)
    
_monitor_stop = True

with open("phase7_dashboard_metrics.json", "w") as f:
    json.dump(results, f, indent=2)

cv2.imencode = original_imencode
cv2.VideoCapture = original_videocapture
CameraStream._update = original_camerastream_update
CameraProcessor.process_frame = original_process_frame
CentralInferenceWorker._run = original_worker_run

print("Phase 7C benchmark completed and monkeypatches restored.")
