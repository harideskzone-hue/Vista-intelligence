import subprocess
import json
import sys
import os
import time

cams = [1, 3, 6, 10, 12]
configs = [
    {"mode": "legacy", "batch": "1"},
    {"mode": "mps_batch", "batch": "1"},
    {"mode": "mps_batch", "batch": "2"},
    {"mode": "mps_batch", "batch": "4"},
]

results = []

script = """
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

print(f"\\n[{num_cameras} CAMERAS | {INFERENCE_MODE} batch={INFERENCE_BATCH_SIZE}] Starting...")

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
"""

with open("benchmark_runner.py", "w") as f:
    f.write(script)

print("Starting Phase 2 benchmark suite...")

for config in configs:
    for count in cams:
        env = os.environ.copy()
        env["INFERENCE_MODE"] = config["mode"]
        env["INFERENCE_BATCH_SIZE"] = config["batch"]
        
        try:
            res = subprocess.run(
                [sys.executable, "benchmark_runner.py", str(count)],
                env=env,
                capture_output=True,
                text=True,
                timeout=45
            )
            
            # Find JSON line in stdout
            for line in res.stdout.splitlines():
                if line.startswith("{"):
                    data = json.loads(line)
                    results.append(data)
                    print(f"Done: {count} cams | {config['mode']} b={config['batch']} | FPS: {data['avg_output_fps_per_camera']:.1f} | CPU: {data['avg_cpu_percent']:.1f}%")
        except subprocess.TimeoutExpired:
            print(f"Timeout: {count} cams | {config['mode']} b={config['batch']}")
            
        time.sleep(2) # Cool down
        
with open("phase2_metrics.json", "w") as f:
    json.dump(results, f, indent=2)
    
print("Phase 2 benchmark completed. Results saved to phase2_metrics.json.")
