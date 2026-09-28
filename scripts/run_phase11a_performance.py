import os
import sys
import glob
import time
import json
import hashlib
import subprocess
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import psutil
import threading
import queue

def get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    except Exception:
        return "unknown"

def hash_file(filepath):
    if not os.path.exists(filepath):
        return "missing"
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()

class ResourceProfiler:
    def __init__(self, interval=0.1):
        self.interval = interval
        self.running = False
        self.cpu_history = []
        self.ram_history = []
        
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._monitor)
        self.thread.start()
        
    def _monitor(self):
        process = psutil.Process(os.getpid())
        while self.running:
            self.cpu_history.append(process.cpu_percent(interval=None))
            self.ram_history.append(process.memory_info().rss / (1024 * 1024))
            time.sleep(self.interval)
            
    def stop(self):
        self.running = False
        self.thread.join()
        
    def get_stats(self):
        return {
            "cpu_avg": np.mean(self.cpu_history) if self.cpu_history else 0,
            "cpu_max": np.max(self.cpu_history) if self.cpu_history else 0,
            "ram_avg_mb": np.mean(self.ram_history) if self.ram_history else 0,
            "ram_max_mb": np.max(self.ram_history) if self.ram_history else 0
        }

def run_performance_experiment(exp_id, batch_size, provider, model_path, dataset_dir, expected_sequences, target_fps=30):
    print(f"\n--- Running Experiment: {exp_id} ---")
    print(f"Config: batch_size={batch_size}, provider={provider}")
    
    actual_hash = hash_file(model_path)
    model = YOLO(model_path)
    if provider == "mps" and not __import__('torch').backends.mps.is_available():
        print("Warning: MPS requested but not available. Falling back to CPU.")
        provider = "cpu"
    
    profiler = ResourceProfiler()
    profiler.start()
    
    inference_latencies = []
    frame_ages = []
    total_dropped = 0
    total_processed = 0
    total_produced = 0
    
    # We will simulate a live stream of frames at target_fps (30fps).
    # To keep the test fast, we will only take a subset of frames from the sequences.
    # E.g. max 300 frames per sequence.
    MAX_FRAMES_PER_SEQ = 100
    
    frame_queue = queue.Queue(maxsize=30) # 1 second buffer
    producer_done = threading.Event()
    
    def producer():
        nonlocal total_produced, total_dropped
        for seq in expected_sequences:
            seq_path = os.path.join(dataset_dir, seq)
            if not os.path.exists(seq_path):
                continue
            
            img_dir = os.path.join(seq_path, 'img1')
            images = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))[:MAX_FRAMES_PER_SEQ]
            
            for img_path in images:
                frame = cv2.imread(img_path)
                timestamp = time.time()
                try:
                    frame_queue.put_nowait((frame, timestamp))
                    total_produced += 1
                except queue.Full:
                    total_dropped += 1
                time.sleep(1.0 / target_fps)
        producer_done.set()

    def consumer():
        nonlocal total_processed
        while not producer_done.is_set() or not frame_queue.empty():
            batch_frames = []
            batch_timestamps = []
            
            try:
                # Wait for at least one frame
                f, t = frame_queue.get(timeout=0.1)
                batch_frames.append(f)
                batch_timestamps.append(t)
            except queue.Empty:
                continue
                
            # Greedily pull up to batch_size
            while len(batch_frames) < batch_size:
                try:
                    f, t = frame_queue.get_nowait()
                    batch_frames.append(f)
                    batch_timestamps.append(t)
                except queue.Empty:
                    break
            
            # Predict
            t0 = time.time()
            _ = model.predict(batch_frames, imgsz=640, conf=0.40, classes=[0], device=provider, verbose=False)
            t1 = time.time()
            
            inf_time_ms = (t1 - t0) * 1000
            inference_latencies.append(inf_time_ms)
            
            for bt in batch_timestamps:
                age_ms = (t1 - bt) * 1000
                frame_ages.append(age_ms)
                total_processed += 1
                
    start_time = time.time()
    t_prod = threading.Thread(target=producer)
    t_cons = threading.Thread(target=consumer)
    
    t_prod.start()
    t_cons.start()
    
    t_prod.join()
    t_cons.join()
    
    profiler.stop()
    total_time = time.time() - start_time
    
    throughput_fps = total_processed / total_time if total_time > 0 else 0
    p50_latency = np.percentile(inference_latencies, 50) if inference_latencies else 0
    p95_latency = np.percentile(inference_latencies, 95) if inference_latencies else 0
    p50_age = np.percentile(frame_ages, 50) if frame_ages else 0
    p95_age = np.percentile(frame_ages, 95) if frame_ages else 0
    max_age = np.max(frame_ages) if frame_ages else 0
    
    resource_stats = profiler.get_stats()
    
    manifest = {
        "experiment_id": exp_id,
        "git_commit": get_git_commit(),
        "model_hash": actual_hash,
        "hardware": {
            "device": provider,
            "os": os.uname().sysname,
            "compute_provider": "Apple MPS" if provider == "mps" else "CPU"
        },
        "dataset": {
            "name": "MOT17 Subset",
            "sampling_protocol": f"Live stream simulation @ {target_fps} fps (max {MAX_FRAMES_PER_SEQ} frames/seq)",
            "dataset_hash": hash_file(os.path.join(dataset_dir, expected_sequences[0], 'gt', 'gt.txt'))
        },
        "evaluation": {
            "protocol_version": "11A-Live-1.0",
            "thresholds": {
                "conf": 0.40,
                "imgsz": 640
            }
        },
        "metrics": {
            "performance": {
                "input_fps": target_fps,
                "inference_fps": throughput_fps,
                "p50_latency_ms": p50_latency,
                "p95_latency_ms": p95_latency,
                "p50_frame_age_ms": p50_age,
                "p95_frame_age_ms": p95_age,
                "max_frame_age_ms": max_age,
                "dropped_frames": total_dropped,
                "processed_frames": total_processed
            }
        },
        "resource_usage": resource_stats
    }
    
    os.makedirs("/Users/hariharans/Documents/SIH26187/reports/phase11", exist_ok=True)
    with open(f"/Users/hariharans/Documents/SIH26187/reports/phase11/{exp_id}.json", "w") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"Results saved to reports/phase11/{exp_id}.json")
    print(f"FPS: {throughput_fps:.2f}, Latency P50: {p50_latency:.2f}ms, P95: {p95_latency:.2f}ms")
    print(f"Frame Age P50: {p50_age:.2f}ms, P95: {p95_age:.2f}ms, Max: {max_age:.2f}ms, Dropped: {total_dropped}")
    
def main():
    model_path = "/Users/hariharans/Documents/SIH26187/models/yolo11n.pt"
    dataset_dir = "/Users/hariharans/Documents/SIH26187/data/external/mot17/MOT17/train"
    
    expected_sequences = [
        "MOT17-02-DPM", "MOT17-02-FRCNN", "MOT17-02-SDP",
        "MOT17-04-DPM", "MOT17-04-FRCNN", "MOT17-04-SDP",
        "MOT17-05-DPM", "MOT17-05-FRCNN", "MOT17-05-SDP",
        "MOT17-09-DPM", "MOT17-09-FRCNN", "MOT17-09-SDP",
        "MOT17-10-DPM", "MOT17-10-FRCNN", "MOT17-10-SDP",
        "MOT17-11-DPM", "MOT17-11-FRCNN", "MOT17-11-SDP",
        "MOT17-13-DPM", "MOT17-13-FRCNN", "MOT17-13-SDP"
    ]
    
    # 11A-4: Batch Performance (using MPS, imgsz=640, conf=0.40)
    for b in [1, 2, 4]:
        run_performance_experiment(f"11A_4_batch_{b}", batch_size=b, provider="mps", model_path=model_path, dataset_dir=dataset_dir, expected_sequences=expected_sequences)
        
    # 11A-5: Provider Performance (using batch=1, imgsz=640, conf=0.40)
    # MPS is already run above (batch 1), we run CPU here
    run_performance_experiment("11A_5_provider_cpu", batch_size=1, provider="cpu", model_path=model_path, dataset_dir=dataset_dir, expected_sequences=expected_sequences)

if __name__ == "__main__":
    main()
