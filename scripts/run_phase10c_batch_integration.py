#!/usr/bin/env python3
import os
import sys
import time
import json
import threading
import queue
import cv2
import psutil
import configparser
import numpy as np
import pandas as pd
import motmetrics as mm
from ultralytics import YOLO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from edge.boundary.adapter import BoundaryTrackingAdapter

# --- MOT Metrics Utils ---
def load_mot_gt(gt_file):
    gt = pd.read_csv(gt_file, header=None)
    gt.columns = ["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "conf", "class", "visibility"]
    gt = gt[gt["class"] == 1]
    return gt

def compute_iou_matrix(gt_boxes, pred_boxes, max_iou=0.5):
    if len(gt_boxes) == 0 or len(pred_boxes) == 0: return np.empty((len(gt_boxes), len(pred_boxes)))
    gt_boxes, pred_boxes = np.asarray(gt_boxes, dtype=float), np.asarray(pred_boxes, dtype=float)
    gt_x1, gt_y1, gt_w, gt_h = gt_boxes[:,0], gt_boxes[:,1], gt_boxes[:,2], gt_boxes[:,3]
    gt_x2, gt_y2 = gt_x1 + gt_w, gt_y1 + gt_h
    pr_x1, pr_y1, pr_w, pr_h = pred_boxes[:,0], pred_boxes[:,1], pred_boxes[:,2], pred_boxes[:,3]
    pr_x2, pr_y2 = pr_x1 + pr_w, pr_y1 + pr_h
    inter_x1, inter_y1 = np.maximum(gt_x1[:, None], pr_x1[None, :]), np.maximum(gt_y1[:, None], pr_y1[None, :])
    inter_x2, inter_y2 = np.minimum(gt_x2[:, None], pr_x2[None, :]), np.minimum(gt_y2[:, None], pr_y2[None, :])
    inter_w, inter_h = np.maximum(0, inter_x2 - inter_x1), np.maximum(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    union_area = (gt_w * gt_h)[:, None] + (pr_w * pr_h)[None, :] - inter_area
    iou = inter_area / np.maximum(union_area, 1e-12)
    dist = 1.0 - iou
    dist[dist > max_iou] = np.nan
    return dist

def process_summary(mh, accs, names):
    metrics = ["mota", "idf1", "num_false_positives", "num_misses", "num_switches", "num_objects", "num_predictions", "mostly_tracked", "mostly_lost"]
    summary = mh.compute_many(accs, metrics=mm.metrics.motchallenge_metrics + ["num_objects", "num_predictions"], names=names, generate_overall=True)
    agg = summary.loc["OVERALL", metrics].to_dict()
    for k, v in agg.items(): agg[k] = float(v) if not pd.isna(v) else None
    gt_objs, preds, fn, idsw = agg["num_objects"], agg["num_predictions"], agg["num_misses"], agg["num_switches"]
    tp = gt_objs - fn
    agg["det_recall"] = tp / gt_objs if gt_objs > 0 else 0
    agg["det_precision"] = tp / preds if preds > 0 else 0
    agg["idsw_per_1k_gt"] = (idsw / gt_objs * 1000) if gt_objs > 0 else 0
    return agg


# --- Part A: Live Throughput Experiment ---

class BoundedDropQueue:
    """Thread-safe queue that drops oldest frame on full."""
    def __init__(self, maxsize):
        self.q = queue.Queue(maxsize=maxsize)
        self.dropped_count = 0
        self.lock = threading.Lock()

    def put_drop_oldest(self, item):
        with self.lock:
            while self.q.full():
                try:
                    self.q.get_nowait()
                    self.dropped_count += 1
                except queue.Empty:
                    break
            self.q.put(item)
            
    def get_batch(self, batch_size, max_wait_s=0.05):
        batch = []
        start = time.time()
        while len(batch) < batch_size:
            rem = max_wait_s - (time.time() - start)
            if rem <= 0 and len(batch) > 0:
                break
            try:
                item = self.q.get(timeout=max(rem, 0.001))
                batch.append(item)
            except queue.Empty:
                if len(batch) > 0:
                    break
        return batch
        
    def qsize(self):
        return self.q.qsize()

def simulate_live_throughput(seq_path, batch_size, maxsize, max_wait_s=0.05, target_ingest_fps=14.5):
    img_dir = os.path.join(seq_path, "img1")
    frames = sorted(os.listdir(img_dir))[:300] 
    
    # PRE-LOAD FRAMES TO ELIMINATE DISK I/O LATENCY IN INGESTION THREAD
    preloaded_images = []
    for f in frames:
        preloaded_images.append(cv2.imread(os.path.join(img_dir, f)))
        
    q = BoundedDropQueue(maxsize)
    metrics = {
        "ingested": 0, "processed": 0, "capture_latency_sum": 0, "batch_latency_sum": 0,
        "queue_depth_sum": 0, "batch_waits": [], "frame_ages": []
    }
    stop_event = threading.Event()
    
    def ingestion_thread():
        interval = 1.0 / target_ingest_fps
        next_time = time.time()
        for img in preloaded_images:
            if stop_event.is_set(): break
            # Wait precisely until next_time
            now = time.time()
            if now < next_time:
                time.sleep(next_time - now)
            
            q.put_drop_oldest((time.time(), img))
            metrics["ingested"] += 1
            next_time += interval
            
    model = YOLO("yolo11n.pt")
    model.predict(np.zeros((640,640,3), dtype=np.uint8), imgsz=640, device="mps", verbose=False)
    
    def worker_thread():
        while metrics["processed"] + q.dropped_count < len(preloaded_images):
            if stop_event.is_set(): break
            q_len = q.qsize()
            metrics["queue_depth_sum"] += q_len
            
            t0 = time.time()
            batch = q.get_batch(batch_size, max_wait_s)
            wait_time = time.time() - t0
            metrics["batch_waits"].append(wait_time)
            
            if not batch:
                time.sleep(0.01)
                continue
                
            imgs = [b[1] for b in batch]
            capture_times = [b[0] for b in batch]
            
            t_inf_start = time.time()
            # Calculate frame age before inference begins
            for ct in capture_times:
                metrics["frame_ages"].append((t_inf_start - ct) * 1000)
                
            model.predict(imgs, imgsz=640, device="mps", verbose=False)
            t_inf_end = time.time()
            
            metrics["batch_latency_sum"] += (t_inf_end - t_inf_start)
            for ct in capture_times:
                metrics["capture_latency_sum"] += (t_inf_end - ct)
            metrics["processed"] += len(batch)
            
    ing_t = threading.Thread(target=ingestion_thread)
    work_t = threading.Thread(target=worker_thread)
    
    start_time = time.time()
    ing_t.start()
    work_t.start()
    
    ing_t.join()
    # allow worker 2 seconds to drain queue
    work_t.join(timeout=2.0)
    stop_event.set()
    
    total_time = time.time() - start_time
    processed = max(metrics["processed"], 1)
    num_batches = len(metrics["batch_waits"])
    
    process = psutil.Process()
    ram_mb = process.memory_info().rss / (1024*1024)
    
    frame_ages = metrics["frame_ages"] if metrics["frame_ages"] else [0.0]
    
    return {
        "configured_ingestion_fps": target_ingest_fps,
        "measured_ingestion_fps": metrics["ingested"] / total_time,
        "inference_fps": processed / total_time,
        "wall_clock_duration_s": total_time,
        "input_frame_count": metrics["ingested"],
        "processed_frame_count": processed,
        "capture_to_inference_latency_ms": (metrics["capture_latency_sum"] / processed) * 1000,
        "per_frame_inference_latency_ms": (metrics["batch_latency_sum"] / processed) * 1000,
        "avg_frame_age_ms": sum(frame_ages) / len(frame_ages),
        "p50_frame_age_ms": np.percentile(frame_ages, 50),
        "p95_frame_age_ms": np.percentile(frame_ages, 95),
        "max_frame_age_ms": np.max(frame_ages),
        "avg_queue_depth": metrics["queue_depth_sum"] / max(num_batches, 1),
        "dropped_frames": q.dropped_count,
        "avg_batch_wait_ms": (sum(metrics["batch_waits"]) / max(num_batches, 1)) * 1000,
        "cpu_percent": psutil.cpu_percent(),
        "ram_mb": ram_mb
    }

# --- Part B: Controlled Tracking Comparison ---

def evaluate_tracking_batch(seq_path, batch_size):
    seqinfo = configparser.ConfigParser()
    seqinfo.read(os.path.join(seq_path, "seqinfo.ini"))
    source_fps = float(seqinfo["Sequence"]["frameRate"])
    seq_length = int(seqinfo["Sequence"]["seqLength"])
    gt_df = load_mot_gt(os.path.join(seq_path, "gt", "gt.txt"))
    img_dir = os.path.join(seq_path, "img1")
    
    acc_trk = mm.MOTAccumulator(auto_id=True)
    adapter = BoundaryTrackingAdapter()
    adapter._profile_locked = True
    
    sampled_indices = []
    next_target_time = 0.0
    sampling_interval = 1.0 / 1.2
    
    for frame_idx in range(1, seq_length + 1):
        frame_time = (frame_idx - 1) / source_fps
        if frame_time >= next_target_time - 1e-5:
            sampled_indices.append(frame_idx)
            next_target_time += sampling_interval
            
    frames_to_process = []
    for idx in sampled_indices:
        img_path = os.path.join(img_dir, f"{idx:06d}.jpg")
        if os.path.exists(img_path):
            frames_to_process.append((idx, cv2.imread(img_path)))
            
    model = YOLO("yolo11n.pt")
    
    for i in range(0, len(frames_to_process), batch_size):
        batch = frames_to_process[i:i+batch_size]
        imgs = [b[1] for b in batch]
        idxs = [b[0] for b in batch]
        
        results = model.predict(imgs, conf=0.40, imgsz=640, device="mps", verbose=False)
        
        for j, res in enumerate(results):
            f_idx = idxs[j]
            observations = adapter.detect_and_track(batch[j][1], timestamp=(f_idx-1)/source_fps)
            if adapter.model.predictor and len(adapter.model.predictor.trackers) > 0:
                adapter.model.predictor.trackers[0].args.match_thresh = 0.8
                
            trk_boxes, trk_ids = [], []
            for obs in observations:
                if obs.class_name == "person":
                    xmin, ymin, xmax, ymax = obs.bbox
                    trk_boxes.append([xmin, ymin, xmax - xmin, ymax - ymin])
                    trk_ids.append(obs.track_id)
                    
            frame_gt = gt_df[gt_df["frame"] == f_idx]
            gt_boxes = frame_gt[["bb_left", "bb_top", "bb_width", "bb_height"]].values.tolist()
            gt_ids = frame_gt["id"].values.tolist()
            
            dist_trk = compute_iou_matrix(gt_boxes, trk_boxes, max_iou=0.5)
            acc_trk.update(gt_ids, trk_ids, dist_trk)
            
    return {"acc_trk": acc_trk}

def run_phase10c():
    out_dir = os.path.join(os.path.dirname(__file__), "..", "reports", "phase10")
    os.makedirs(out_dir, exist_ok=True)
    mot17_train_dir = os.path.join(os.path.dirname(__file__), "..", "data", "external", "mot17", "MOT17", "train")
    if not os.path.exists(mot17_train_dir):
        print("MOT17 dataset missing.")
        sys.exit(1)
        
    sequences = sorted([os.path.join(mot17_train_dir, d) for d in os.listdir(mot17_train_dir) if os.path.isdir(os.path.join(mot17_train_dir, d))])[:1] # Use 1 sequence for speed
    seq = sequences[0]
    
    results = {"A_Live_Throughput": {}, "B_Tracking_Comparison": {}}
    
    # Part A
    for q_size in [10, 30]:
        for b_size in [1, 2, 4]:
            print(f"Running Part A: Q={q_size}, Batch={b_size}")
            metrics = simulate_live_throughput(seq, b_size, q_size)
            results["A_Live_Throughput"][f"Q{q_size}_B{b_size}"] = metrics
            
    # Part B
    mh = mm.metrics.create()
    for b_size in [1, 2, 4]:
        print(f"Running Part B: Batch={b_size}")
        acc = evaluate_tracking_batch(seq, b_size)
        agg = process_summary(mh, [acc["acc_trk"]], ["seq"])
        results["B_Tracking_Comparison"][f"B{b_size}"] = agg
        
    out_path = os.path.join(out_dir, "phase10c_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Phase 10C results written to {out_path}")

if __name__ == "__main__":
    run_phase10c()
