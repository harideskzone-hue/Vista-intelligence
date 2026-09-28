import cv2
import time
import os
import psutil
from ultralytics import YOLO

video_path = "test_video.mp4"
model_path = "models/yolo26n-face.pt"

print("Loading YOLO model...")
model = YOLO(model_path)

resolutions = [
    ("1920x1080 (Native/Current)", (1920, 1080)),
    ("1280x720", (1280, 720)),
    ("640x480 (Proposed)", (640, 480)),
    ("640x360", (640, 360))
]

results = {}

for name, (w, h) in resolutions:
    print(f"\n======================================")
    print(f"Benchmarking Resolution: {name}")
    print(f"======================================")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Failed to open video")
        break
        
    total_frames = 0
    total_faces = 0
    start_time = time.time()
    
    # Process max 200 frames for speed
    MAX_FRAMES = 200
    
    while total_frames < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret: break
        
        # Resize to target resolution
        frame = cv2.resize(frame, (w, h))
        
        # Inference
        out = model(frame, verbose=False, conf=0.5, device='cpu')
        
        if len(out) > 0 and len(out[0].boxes) > 0:
            total_faces += len(out[0].boxes)
            
        total_frames += 1
        
    end_time = time.time()
    elapsed = end_time - start_time
    fps = total_frames / elapsed if elapsed > 0 else 0
    latency_ms = (elapsed / total_frames) * 1000 if total_frames > 0 else 0
    
    # Process Memory
    process = psutil.Process(os.getpid())
    ram_mb = process.memory_info().rss / 1024 / 1024
    
    print(f"Total Frames: {total_frames}")
    print(f"Total Faces Detected: {total_faces}")
    print(f"Inference FPS: {fps:.1f}")
    print(f"Avg Latency: {latency_ms:.1f} ms")
    print(f"RAM Usage: {ram_mb:.1f} MB")
    
    results[name] = {
        "faces": total_faces,
        "fps": fps,
        "latency_ms": latency_ms,
        "ram_mb": ram_mb
    }
    
    cap.release()

print("\n\n--- Benchmark Summary ---")
for name, res in results.items():
    print(f"{name}: {res['faces']} faces, {res['fps']:.1f} FPS, {res['latency_ms']:.1f}ms latency, {res['ram_mb']:.1f} MB RAM")
