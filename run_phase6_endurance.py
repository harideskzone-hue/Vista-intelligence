
import time, os, sys, psutil, json, concurrent.futures
import cv2

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE, "face_engine"))

os.environ["INFERENCE_MODE"] = "mps_batch"
os.environ["INFERENCE_BATCH_SIZE"] = "1"
os.environ["OPTIMIZE_SCHEDULING"] = "true"
os.environ["LAZY_ENCODING"] = "true"

from ultralytics import YOLO
from live_scorer import CameraProcessor, CameraStream, INFERENCE_MODE, INFERENCE_BATCH_SIZE, OPTIMIZE_SCHEDULING, LAZY_ENCODING

# 1. Monkeypatch CameraStream to loop on EOF and count reconnects
original_init = CameraStream.__init__
def patched_init(self, src):
    original_init(self, src)
    self.eof_reconnect_count = 0
CameraStream.__init__ = patched_init

def _patched_update(self):
    while not self.stopped:
        grabbed = self.stream.grab()
        if not grabbed:
            # EOF reached on video file! Re-open it!
            if isinstance(self.src, str):
                self.stream.release()
                self.stream = cv2.VideoCapture(self.src)
                self.eof_reconnect_count += 1
                continue
            time.sleep(0.005)
            continue
        
        ret, frame = self.stream.retrieve()
        if ret:
            with self._lock:
                self.ret = ret
                self.frame = frame
                self.frame_id += 1
        
        if isinstance(self.src, str) and self.src.lower().endswith((".mp4", ".avi", ".mov")):
            time.sleep(1/30.0)
            
    self.stream.release()

CameraStream._update = _patched_update

num_cameras = 12
duration_sec = 35 * 60 # 35 minutes

print(f"[PHASE 6 ENDURANCE] {num_cameras} CAMERAS for {duration_sec/60} minutes")

model_path = os.path.join(BASE, "face_engine", "models", "yolo26n-face.pt")
face_det = YOLO(model_path)

processors = {}
for i in range(num_cameras):
    cid = f"cam_{i}"
    src = os.path.join(BASE, "test_video.mp4")
    proc = CameraProcessor(cid, src, face_det)
    processors[cid] = proc

print("Warming up for 10 seconds...")
warmup_start = time.time()
while time.time() - warmup_start < 10:
    for proc in processors.values(): proc.process_frame()
    time.sleep(0.01)

print("Starting 35-minute endurance run...")
start_time = time.time()
metrics = []
frames_processed = {cid: 0 for cid in processors}

process = psutil.Process(os.getpid())
executor = concurrent.futures.ThreadPoolExecutor(max_workers=6)

def _process_one(proc):
    start = time.time()
    proc.process_frame()
    latency = time.time() - start
    
    encoded = False
    if getattr(proc, "has_new_viz", False) and proc.viz_frame is not None:
        proc.has_new_viz = False
        encoded = True
    return proc.cam_id, encoded, latency

last_log = time.time()
loop_count = 0

latencies = []

while time.time() - start_time < duration_sec:
    res = executor.map(_process_one, processors.values())
    for cid, encoded, lat in res:
        latencies.append(lat)
        if encoded: frames_processed[cid] += 1
            
    time.sleep(0.01)
    loop_count += 1
    
    # Log telemetry every 30 seconds
    now = time.time()
    if now - last_log >= 30:
        cpu = process.cpu_percent(interval=None)
        mem = process.memory_info().rss / (1024 * 1024)
        elapsed = now - start_time
        avg_fps = (sum(frames_processed.values()) / elapsed) / num_cameras
        
        reconnects = sum(proc.stream.eof_reconnect_count for proc in processors.values() if hasattr(proc.stream, "eof_reconnect_count"))
        
        p95_lat = sorted(latencies)[int(len(latencies)*0.95)] if latencies else 0
        latencies.clear()
        
        m = {
            "timestamp": now,
            "elapsed_min": round(elapsed / 60, 2),
            "cpu_percent": round(cpu, 1),
            "ram_mb": round(mem, 1),
            "avg_fps_per_camera": round(avg_fps, 2),
            "p95_latency": round(p95_lat, 3),
            "reconnects": reconnects
        }
        metrics.append(m)
        print(f"[{m['elapsed_min']} min] CPU: {m['cpu_percent']}%, RAM: {m['ram_mb']}MB, FPS: {m['avg_fps_per_camera']}, Reconnects: {m['reconnects']}")
        
        with open("phase6_endurance_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
            
        last_log = now

for proc in processors.values(): proc.stop()
executor.shutdown(wait=True)
print("Endurance test completed successfully.")
