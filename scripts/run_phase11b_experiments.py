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
import motmetrics as mm
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

def compute_iou_matrix(gt_boxes, pred_boxes, max_iou=0.5):
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return np.empty((len(gt_boxes), len(pred_boxes)))
    
    gt_boxes = np.asarray(gt_boxes, dtype=float)
    pred_boxes = np.asarray(pred_boxes, dtype=float)
    
    gt_x1, gt_y1, gt_w, gt_h = gt_boxes[:,0], gt_boxes[:,1], gt_boxes[:,2], gt_boxes[:,3]
    gt_x2, gt_y2 = gt_x1 + gt_w, gt_y1 + gt_h
    
    pr_x1, pr_y1, pr_w, pr_h = pred_boxes[:,0], pred_boxes[:,1], pred_boxes[:,2], pred_boxes[:,3]
    pr_x2, pr_y2 = pr_x1 + pr_w, pr_y1 + pr_h
    
    inter_x1 = np.maximum(gt_x1[:, None], pr_x1[None, :])
    inter_y1 = np.maximum(gt_y1[:, None], pr_y1[None, :])
    inter_x2 = np.minimum(gt_x2[:, None], pr_x2[None, :])
    inter_y2 = np.minimum(gt_y2[:, None], pr_y2[None, :])
    
    inter_w = np.maximum(0, inter_x2 - inter_x1)
    inter_h = np.maximum(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    
    gt_area = gt_w * gt_h
    pr_area = pr_w * pr_h
    union_area = gt_area[:, None] + pr_area[None, :] - inter_area
    
    iou = inter_area / np.maximum(union_area, 1e-12)
    dist = 1.0 - iou
    dist[dist > max_iou] = np.nan
    return dist

def greedy_match(dist_matrix):
    matched_gt = set()
    matched_pred = set()
    matches = []
    
    if dist_matrix.size > 0:
        r, c = np.where(~np.isnan(dist_matrix))
        items = [(dist_matrix[i, j], i, j) for i, j in zip(r, c)]
        items.sort(key=lambda x: x[0]) 
        
        for d, i, j in items:
            if i not in matched_gt and j not in matched_pred:
                matched_gt.add(i)
                matched_pred.add(j)
                matches.append((i, j))
                
    return matches, matched_gt, matched_pred

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

def run_experiment(exp_id, imgsz, conf, batch_size, target_fps, model_path, dataset_dir, expected_sequences):
    print(f"\n--- Running Experiment: {exp_id} ---")
    print(f"Config: conf={conf}, batch={batch_size}, fps={target_fps}")
    
    actual_hash = hash_file(model_path)
    
    # Live stream queue simulation to measure latency and frame age correctly
    profiler = ResourceProfiler()
    profiler.start()
    
    # Detection accumulators
    det_tp = 0
    det_fp = 0
    det_fn = 0
    
    # Tracking accumulators
    acc = mm.MOTAccumulator(auto_id=True)
    
    inference_latencies = []
    frame_ages = []
    total_dropped = 0
    total_processed = 0
    
    producer_fps = 30.0 # simulated camera
    frame_queue = queue.Queue(maxsize=30)
    producer_done = threading.Event()
    
    def producer():
        nonlocal total_dropped
        for seq in expected_sequences:
            seq_path = os.path.join(dataset_dir, seq)
            if not os.path.exists(seq_path):
                continue
                
            import configparser
            seqinfo = configparser.ConfigParser()
            seqinfo.read(os.path.join(seq_path, 'seqinfo.ini'))
            seq_fps = int(seqinfo['Sequence']['frameRate'])
            seq_length = int(seqinfo['Sequence']['seqLength'])
            
            # Sampling logic: 
            # If target_fps = 1.0, stride = seq_fps (e.g. 30 for 30fps video)
            # If target_fps = 0.5, stride = seq_fps * 2
            # For exact "1.2 FPS" average from Phase 8B where we used stride=seq_fps
            # Note: The sequences have different FPS (mostly 30, some 14).
            # If we strictly want exact 11B-0 frames, target_fps="baseline" means stride = seq_fps.
            if target_fps == "baseline":
                stride = seq_fps
            else:
                stride = max(1, int(seq_fps / target_fps))
                
            img_dir = os.path.join(seq_path, 'img1')
            sampled_indices = list(range(1, seq_length + 1, stride))
            
            # Put sequence marker to tell consumer to reset model state
            frame_queue.put(("RESET_SEQ", seq))
            
            for idx in sampled_indices:
                img_path = os.path.join(img_dir, f"{idx:06d}.jpg")
                if not os.path.exists(img_path):
                    continue
                frame = cv2.imread(img_path)
                timestamp = time.time()
                try:
                    frame_queue.put_nowait(("FRAME", (frame, timestamp, seq, idx)))
                except queue.Full:
                    total_dropped += 1
                time.sleep(1.0 / producer_fps)
                
        producer_done.set()

    def consumer():
        nonlocal total_processed, det_tp, det_fp, det_fn
        # Model inside consumer thread
        model = YOLO(model_path)
        
        while not producer_done.is_set() or not frame_queue.empty():
            try:
                msg = frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue
                
            if msg[0] == "RESET_SEQ":
                model = YOLO(model_path) # Fresh model to clear tracker state
                continue
                
            batch_data = [msg[1]]
            # Greedily pull up to batch_size
            while len(batch_data) < batch_size:
                try:
                    msg2 = frame_queue.get_nowait()
                    if msg2[0] == "RESET_SEQ":
                        # Put it back and break
                        frame_queue.put(msg2)
                        break
                    batch_data.append(msg2[1])
                except queue.Empty:
                    break
                    
            batch_frames = [b[0] for b in batch_data]
            batch_ts = [b[1] for b in batch_data]
            batch_seqs = [b[2] for b in batch_data]
            batch_idxs = [b[3] for b in batch_data]
            
            # RAW DETECTION EVAL (Not timed for latency)
            det_results = model.predict(batch_frames, imgsz=imgsz, conf=conf, classes=[0], device="mps", verbose=False)
            
            # TRACKING EVAL (Timed)
            t0 = time.time()
            if len(batch_frames) == 1:
                trk_results = model.track(batch_frames[0], persist=True, tracker="bytetrack.yaml", imgsz=imgsz, conf=conf, classes=[0], device="mps", verbose=False)
                trk_results = [trk_results[0]]
            else:
                # Passing a list to model.track assigns them to parallel tracker streams (tracker[0], tracker[1]...).
                # This breaks sequential temporal association if they are from the same sequence.
                # We will run it as requested to observe the metric impact.
                trk_results = model.track(batch_frames, persist=True, tracker="bytetrack.yaml", imgsz=imgsz, conf=conf, classes=[0], device="mps", verbose=False)
            t1 = time.time()
            
            inf_time_ms = (t1 - t0) * 1000
            inference_latencies.append(inf_time_ms)
            
            # Update metrics
            for i in range(len(batch_frames)):
                seq_name = batch_seqs[i]
                frame_idx = batch_idxs[i]
                age_ms = (t1 - batch_ts[i]) * 1000
                frame_ages.append(age_ms)
                total_processed += 1
                
                # Load GT for this frame
                gt_file = os.path.join(dataset_dir, seq_name, 'gt', 'gt.txt')
                gt_df = pd.read_csv(gt_file, header=None)
                gt_df.columns = ['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'class', 'visibility']
                frame_gt = gt_df[(gt_df['class'] == 1) & (gt_df['frame'] == frame_idx)]
                gt_boxes = frame_gt[['bb_left', 'bb_top', 'bb_width', 'bb_height']].values.tolist()
                gt_ids = frame_gt['id'].values.tolist()
                
                # Raw detection metric
                det_pred_boxes = []
                for box in det_results[i].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    det_pred_boxes.append([x1, y1, x2 - x1, y2 - y1])
                dist_matrix_det = compute_iou_matrix(gt_boxes, det_pred_boxes, max_iou=0.5)
                matches_det, _, _ = greedy_match(dist_matrix_det)
                det_tp += len(matches_det)
                det_fp += len(det_pred_boxes) - len(matches_det)
                det_fn += len(gt_boxes) - len(matches_det)
                
                # Tracking metric
                trk_pred_boxes = []
                trk_pred_ids = []
                if trk_results[i].boxes.id is not None:
                    boxes = trk_results[i].boxes.xyxy.cpu().numpy()
                    ids = trk_results[i].boxes.id.cpu().numpy()
                    for box, tid in zip(boxes, ids):
                        x1, y1, x2, y2 = box
                        trk_pred_boxes.append([x1, y1, x2 - x1, y2 - y1])
                        trk_pred_ids.append(int(tid))
                
                # Using a sequence-specific identifier for the MOTAccumulator to prevent cross-sequence ID conflicts?
                # MOTAccumulator usually runs per sequence, but here we aggregate all updates directly.
                # To prevent ID collisions across sequences, we shift GT and Pred IDs by hashing the sequence name.
                seq_offset = int(hashlib.md5(seq_name.encode()).hexdigest(), 16) % 100000000
                shifted_gt_ids = [seq_offset + int(gid) for gid in gt_ids]
                shifted_pred_ids = [seq_offset + int(pid) for pid in trk_pred_ids]
                
                dist_matrix_trk = compute_iou_matrix(gt_boxes, trk_pred_boxes, max_iou=0.5)
                acc.update(shifted_gt_ids, shifted_pred_ids, dist_matrix_trk)

    start_time = time.time()
    t_prod = threading.Thread(target=producer)
    t_cons = threading.Thread(target=consumer)
    t_prod.start()
    t_cons.start()
    t_prod.join()
    t_cons.join()
    profiler.stop()
    
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=['num_frames', 'mota', 'idf1', 'mostly_tracked', 'mostly_lost', 
                                       'num_false_positives', 'num_misses', 'num_switches', 'num_fragmentations', 
                                       'num_predictions', 'num_objects', 'num_matches'], name='aggregate')
    
    # Calculate Precision and Recall from MOT matching
    tp = summary.loc['aggregate', 'num_matches']
    fp = summary.loc['aggregate', 'num_false_positives']
    fn = summary.loc['aggregate', 'num_misses']
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    mota = summary.loc['aggregate', 'mota']
    idf1 = summary.loc['aggregate', 'idf1']
    mostly_tracked = summary.loc['aggregate', 'mostly_tracked']
    mostly_lost = summary.loc['aggregate', 'mostly_lost']
    num_switches = summary.loc['aggregate', 'num_switches']
    num_fragmentations = summary.loc['aggregate', 'num_fragmentations']
    num_objects = summary.loc['aggregate', 'num_objects']
    
    idsw_per_1k = (num_switches / num_objects) * 1000 if num_objects > 0 else 0
    
    det_precision = det_tp / (det_tp + det_fp) if (det_tp + det_fp) > 0 else 0
    det_recall = det_tp / (det_tp + det_fn) if (det_tp + det_fn) > 0 else 0
    
    manifest = {
        "experiment_id": exp_id,
        "git_commit": get_git_commit(),
        "model_hash": actual_hash,
        "config": {
            "conf": conf,
            "imgsz": imgsz,
            "batch": batch_size,
            "fps": target_fps,
            "tracker": "bytetrack"
        },
        "metrics": {
            "raw_detection": {
                "TP": det_tp,
                "FP": det_fp,
                "FN": det_fn,
                "precision": float(det_precision),
                "recall": float(det_recall)
            },
            "tracking": {
                "TP": int(tp),
                "FP": int(fp),
                "FN": int(fn),
                "precision": float(precision),
                "recall": float(recall),
                "mota": float(mota),
                "idf1": float(idf1),
                "mostly_tracked": int(mostly_tracked),
                "mostly_lost": int(mostly_lost),
                "num_switches": int(num_switches),
                "num_fragmentations": int(num_fragmentations),
                "idsw_per_1k": float(idsw_per_1k)
            },
            "performance": {
                "latency_p50_ms": float(np.percentile(inference_latencies, 50)) if inference_latencies else 0,
                "frame_age_max_ms": float(np.max(frame_ages)) if frame_ages else 0,
                "dropped_frames": int(total_dropped)
            },
            "resource_usage": profiler.get_stats()
        }
    }
    
    os.makedirs("/Users/hariharans/Documents/SIH26187/reports/phase11", exist_ok=True)
    with open(f"/Users/hariharans/Documents/SIH26187/reports/phase11/{exp_id}.json", "w") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"[{exp_id}] DET Recall: {det_recall:.4f} | TRK Recall: {recall:.4f} | MOTA: {mota:.4f} | IDF1: {idf1:.4f} | IDSW: {num_switches}")

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
    
    # 11B-1 Confidence
    # run_experiment("11B_1_conf_40_baseline", 640, 0.40, 1, "baseline", model_path, dataset_dir, expected_sequences)
    run_experiment("11B_1_conf_20_candidate", 640, 0.20, 1, "baseline", model_path, dataset_dir, expected_sequences)
    
    # 11B-2 Batching
    run_experiment("11B_2_batch_2", 640, 0.40, 2, "baseline", model_path, dataset_dir, expected_sequences)
    run_experiment("11B_2_batch_4", 640, 0.40, 4, "baseline", model_path, dataset_dir, expected_sequences)
    
    # 11B-3 Combined
    run_experiment("11B_3_combined_20_b4", 640, 0.20, 4, "baseline", model_path, dataset_dir, expected_sequences)
    
    # 11B-4 Sampling Sensitivity (using conf=0.40, batch=1 for isolated sampling effect test)
    run_experiment("11B_4_fps_0_5", 640, 0.40, 1, 0.5, model_path, dataset_dir, expected_sequences)
    run_experiment("11B_4_fps_1_0", 640, 0.40, 1, 1.0, model_path, dataset_dir, expected_sequences)
    run_experiment("11B_4_fps_2_0", 640, 0.40, 1, 2.0, model_path, dataset_dir, expected_sequences)

if __name__ == "__main__":
    main()
