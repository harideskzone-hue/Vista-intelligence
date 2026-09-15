
import time, os, sys, psutil, json, concurrent.futures
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE, "face_engine"))

os.environ["INFERENCE_MODE"] = "mps_batch"
os.environ["INFERENCE_BATCH_SIZE"] = "1"
os.environ["OPTIMIZE_SCHEDULING"] = "true"
os.environ["LAZY_ENCODING"] = "true"

from ultralytics import YOLO
from live_scorer import CameraProcessor, INFERENCE_MODE, INFERENCE_BATCH_SIZE, OPTIMIZE_SCHEDULING, LAZY_ENCODING

num_cameras = 12
duration_sec = 35 * 60 # 35 minutes

print(f"[ENDURANCE TEST] {num_cameras} CAMERAS for {duration_sec/60} minutes")
print(f"Config: Mode={INFERENCE_MODE}, Batch={INFERENCE_BATCH_SIZE}, Sched={OPTIMIZE_SCHEDULING}, Lazy={LAZY_ENCODING}")

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
    proc.process_frame()
    encoded = False
    if getattr(proc, "has_new_viz", False) and proc.viz_frame is not None:
        proc.has_new_viz = False
        encoded = True
    return proc.cam_id, encoded

last_log = time.time()
loop_count = 0

while time.time() - start_time < duration_sec:
    res = executor.map(_process_one, processors.values())
    for cid, encoded in res:
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
        
        m = {
            "timestamp": now,
            "elapsed_min": round(elapsed / 60, 2),
            "cpu_percent": round(cpu, 1),
            "ram_mb": round(mem, 1),
            "avg_fps_per_camera": round(avg_fps, 2)
        }
        metrics.append(m)
        print(f"[{m['elapsed_min']} min] CPU: {m['cpu_percent']}%, RAM: {m['ram_mb']}MB, FPS: {m['avg_fps_per_camera']}")
        
        # Save incrementally in case of crash
        with open("phase5_endurance_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
            
        last_log = now

for proc in processors.values(): proc.stop()
executor.shutdown(wait=True)
print("Endurance test completed successfully.")
