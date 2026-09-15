#!/usr/bin/env python3
import time
import os
import sys
import psutil
import json
import concurrent.futures
from collections import deque
import platform

# Add engine path
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE, "face_engine"))

from ultralytics import YOLO
from live_scorer import CameraProcessor

def get_mps_availability():
    try:
        import torch
        return torch.backends.mps.is_available()
    except ImportError:
        return False

def run_benchmark(num_cameras, duration_sec=30):
    print(f"\n[{num_cameras} CAMERAS] Starting benchmark for {duration_sec} seconds...")
    
    model_path = os.path.join(BASE, "face_engine", "models", "yolo26n-face.pt")
    face_det = YOLO(model_path)
    
    print(f"Initializing {num_cameras} camera processors...")
    processors = {}
    for i in range(num_cameras):
        cam_id = f"cam_{i}"
        src = os.path.join(BASE, "test_video.mp4")
        proc = CameraProcessor(cam_id, src, face_det)
        processors[cam_id] = proc
    
    time.sleep(3) # Wait for cameras to connect and warm up
    
    start_time = time.time()
    cpu_samples = []
    mem_samples = []
    frames_processed = {cam_id: 0 for cam_id in processors}
    
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
            
        results = executor.map(_process_one, processors.values())
        for cam_id, encoded in results:
            if encoded:
                frames_processed[cam_id] += 1
                
        time.sleep(0.01)
        loop_count += 1
        
    for proc in processors.values():
        proc.stop()
    executor.shutdown(wait=True)
    
    avg_cpu = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0
    avg_mem = sum(mem_samples) / len(mem_samples) if mem_samples else 0
    total_frames = sum(frames_processed.values())
    avg_fps_per_cam = (total_frames / duration_sec) / num_cameras if num_cameras > 0 else 0
    
    print(f"[{num_cameras} CAMERAS] Avg CPU: {avg_cpu:.1f}%, Mem: {avg_mem:.1f} MB, Avg FPS/Cam: {avg_fps_per_cam:.1f}")
    
    return {
        "num_cameras": num_cameras,
        "avg_cpu_percent": avg_cpu,
        "avg_memory_mb": avg_mem,
        "avg_output_fps_per_camera": avg_fps_per_cam,
        "total_frames_processed": total_frames
    }

def main():
    print("Gathering Environment Info...")
    env_info = {
        "os": platform.platform(),
        "python": sys.version.split(' ')[0],
        "processor": platform.processor(),
        "mps_available": get_mps_availability()
    }
    
    results = []
    camera_counts = [1, 3, 6, 10, 12]
    
    for count in camera_counts:
        res = run_benchmark(count, duration_sec=15)
        results.append(res)
        time.sleep(2)
        
    out_data = {
        "environment": env_info,
        "benchmarks": results
    }
    
    with open("baseline_metrics.json", "w") as f:
        json.dump(out_data, f, indent=2)
        
    print("\nBenchmark saved to baseline_metrics.json")
    
if __name__ == "__main__":
    main()
