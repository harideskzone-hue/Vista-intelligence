import os, sys, time, json, statistics, concurrent.futures, psutil, threading
import cv2

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE, "face_engine"))

os.environ["INFERENCE_MODE"] = "mps_batch"
os.environ["INFERENCE_BATCH_SIZE"] = "1"
os.environ["OPTIMIZE_SCHEDULING"] = "true"
os.environ["LAZY_ENCODING"] = "true"

from ultralytics import YOLO
from live_scorer import CameraStream, CameraProcessor, CentralInferenceWorker, _global_central_worker

_METRICS = {
    "rtsp_connect_times": [],
    "reconnects": 0,
    "grab_intervals": {},
    "decode_latency": [],
    "ai_queue_depths": [],
    "inference_latency": [],
    "ai_inference_count": 0,
    "end_to_end_latency": [],
    "frame_drops": {},
    "cpu_timeseries": [],
    "ram_timeseries": []
}

# --- Monkeypatching ---

# 1. RTSP Connect
original_videocapture = cv2.VideoCapture
def patched_videocapture(*args, **kwargs):
    start = time.time()
    cap = original_videocapture(*args, **kwargs)
    _METRICS["rtsp_connect_times"].append(time.time() - start)
    return cap
cv2.VideoCapture = patched_videocapture

# 2. CameraStream (Capture FPS, Jitter, Decode Latency, Capture Timestamp)
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
        else:
            _METRICS["reconnects"] += 1
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
        
        # Attach capture timestamp to the frame object for E2E latency tracking
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
                cid = cam_ids[i]
                boxes = []
                if len(res.boxes) > 0:
                    for box in res.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        boxes.append((x1, y1, x2, y2))
                self.results[cid] = boxes
        except Exception as e:
            print(f"Inference error: {e}")
            
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

# --- Benchmarking Logic ---
print("Initializing FaceDetection...")
model_path = os.path.join(BASE, "face_engine", "models", "yolo26n-face.pt")
face_det = YOLO(model_path)

def benchmark(num_cams, duration=20):
    print(f"--- Benchmarking {num_cams} RTSP Streams ---")
    # Reset metrics
    for k in ["rtsp_connect_times", "decode_latency", "ai_queue_depths", "inference_latency", "end_to_end_latency", "cpu_timeseries", "ram_timeseries"]:
        _METRICS[k].clear()
    _METRICS["reconnects"] = 0
    _METRICS["ai_inference_count"] = 0
    _METRICS["grab_intervals"].clear()
    _METRICS["frame_drops"].clear()
    
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
        
    print("Running measurement...")
    # Clear metrics gathered during warmup
    for k in ["decode_latency", "ai_queue_depths", "inference_latency", "end_to_end_latency"]:
        _METRICS[k].clear()
    _METRICS["ai_inference_count"] = 0
    
    start_time = time.time()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=min(num_cams, 6))
    
    def _process_one(proc):
        proc.process_frame()
        
    while time.time() - start_time < duration:
        list(executor.map(_process_one, processors.values()))
        
    for proc in processors.values():
        proc.stream.stop()
        
    # Aggregate stats
    avg_connect = sum(_METRICS["rtsp_connect_times"]) / max(1, len(_METRICS["rtsp_connect_times"]))
    
    all_intervals = [i for intervals in _METRICS["grab_intervals"].values() for i in intervals]
    avg_capture_fps = (1.0 / (sum(all_intervals) / len(all_intervals))) if all_intervals else 0
    jitter = statistics.stdev(all_intervals) if len(all_intervals) > 1 else 0
    
    avg_decode = sum(_METRICS["decode_latency"]) / max(1, len(_METRICS["decode_latency"])) * 1000
    
    ai_fps = _METRICS["ai_inference_count"] / duration
    avg_inf_lat = sum(_METRICS["inference_latency"]) / max(1, len(_METRICS["inference_latency"])) * 1000
    avg_e2e = sum(_METRICS["end_to_end_latency"]) / max(1, len(_METRICS["end_to_end_latency"])) * 1000
    
    depths = _METRICS["ai_queue_depths"]
    avg_queue = sum(depths) / max(1, len(depths))
    max_queue = max(depths) if depths else 0
    
    total_drops = sum(_METRICS["frame_drops"].values())
    
    cpus = _METRICS["cpu_timeseries"]
    rams = _METRICS["ram_timeseries"]
    
    return {
        "num_cameras": num_cams,
        "avg_connect_sec": round(avg_connect, 3),
        "reconnects": _METRICS["reconnects"],
        "avg_capture_fps": round(avg_capture_fps, 2),
        "inter_frame_jitter_sec": round(jitter, 4),
        "avg_decode_latency_ms": round(avg_decode, 2),
        "total_frame_drops": total_drops,
        "ai_inference_fps": round(ai_fps, 2),
        "avg_inference_latency_ms": round(avg_inf_lat, 2),
        "avg_e2e_latency_ms": round(avg_e2e, 2),
        "avg_ai_queue_depth": round(avg_queue, 2),
        "max_ai_queue_depth": max_queue,
        "cpu_avg_percent": round(sum(cpus)/max(1, len(cpus)), 1),
        "ram_avg_mb": round(sum(rams)/max(1, len(rams)), 1),
        "mps_utilization": "NOT MEASURED",
        "network_bandwidth": "NOT MEASURED",
        "dashboard_clients": "NOT MEASURED"
    }

print("Running Mock RTSP Tests...")
results = []
for cams in [1, 3, 6, 10, 12]:
    res = benchmark(cams)
    print(res)
    results.append(res)
    
_monitor_stop = True

with open("phase7_network_metrics.json", "w") as f:
    json.dump(results, f, indent=2)

# Restore Monkeypatches
cv2.VideoCapture = original_videocapture
CameraStream._update = original_camerastream_update
CameraProcessor.process_frame = original_process_frame
CentralInferenceWorker._run = original_worker_run

print("Phase 7A benchmark completed and monkeypatches restored.")
