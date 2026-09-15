
import time, os, sys, psutil, json, concurrent.futures, platform
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE, "face_engine"))
from ultralytics import YOLO
from live_scorer import CameraProcessor, INFERENCE_MODE, INFERENCE_BATCH_SIZE

def get_mps_availability():
    try:
        import torch
        return torch.backends.mps.is_available()
    except ImportError: return False

num_cameras = int(sys.argv[1])
duration_sec = 15

print(f"\n[{num_cameras} CAMERAS | {INFERENCE_MODE} batch={INFERENCE_BATCH_SIZE}] Starting...")

model_path = os.path.join(BASE, "face_engine", "models", "yolo26n-face.pt")
face_det = YOLO(model_path)

processors = {}
for i in range(num_cameras):
    cid = f"cam_{i}"
    src = os.path.join(BASE, "test_video.mp4")
    proc = CameraProcessor(cid, src, face_det)
    processors[cid] = proc

# Warmup for 5 seconds (excluded from stats)
print("Warming up...")
warmup_start = time.time()
while time.time() - warmup_start < 5:
    for proc in processors.values(): proc.process_frame()
    time.sleep(0.01)
    
print("Starting benchmark measurements...")
start_time = time.time()
cpu_samples, mem_samples = [], []
frames_processed = {cid: 0 for cid in processors}

process = psutil.Process(os.getpid())
executor = concurrent.futures.ThreadPoolExecutor(max_workers=6)

def _process_one(proc):
    t0 = time.time()
    proc.process_frame()
    encoded = False
    if getattr(proc, 'has_new_viz', False) and proc.viz_frame is not None:
        proc.has_new_viz = False
        import cv2
        _, _ = cv2.imencode('.jpg', proc.viz_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        encoded = True
    return proc.cam_id, encoded

loop_count = 0
while time.time() - start_time < duration_sec:
    if loop_count % 10 == 0:
        cpu_samples.append(process.cpu_percent(interval=None))
        mem_samples.append(process.memory_info().rss / (1024 * 1024))
        
    res = executor.map(_process_one, processors.values())
    for cid, encoded in res:
        if encoded: frames_processed[cid] += 1
            
    time.sleep(0.01)
    loop_count += 1
    
for proc in processors.values(): proc.stop()
executor.shutdown(wait=True)

avg_cpu = sum(cpu_samples) / max(1, len(cpu_samples))
avg_mem = sum(mem_samples) / max(1, len(mem_samples))
total_frames = sum(frames_processed.values())
avg_fps = (total_frames / duration_sec) / max(1, num_cameras)

out = {
    "num_cameras": num_cameras,
    "mode": INFERENCE_MODE,
    "batch": INFERENCE_BATCH_SIZE,
    "avg_cpu_percent": avg_cpu,
    "avg_memory_mb": avg_mem,
    "avg_output_fps_per_camera": avg_fps,
    "total_frames_processed": total_frames
}
print(json.dumps(out))
