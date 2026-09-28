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

def run_11b_0_baseline():
    print("Starting 11B-0 Baseline Reproduction")
    
    model_path = "/Users/hariharans/Documents/SIH26187/models/yolo11n.pt"
    expected_hash = "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1"
    
    actual_hash = hash_file(model_path)
    if actual_hash != expected_hash:
        print(f"ABORT: Model hash mismatch. Expected {expected_hash}, got {actual_hash}")
        sys.exit(1)
        
    dataset_dir = "/Users/hariharans/Documents/SIH26187/data/external/mot17/MOT17/train"
    if not os.path.exists(dataset_dir):
        print(f"ABORT: Dataset directory not found: {dataset_dir}")
        sys.exit(1)
            
    expected_sequences = [
        "MOT17-02-DPM", "MOT17-02-FRCNN", "MOT17-02-SDP",
        "MOT17-04-DPM", "MOT17-04-FRCNN", "MOT17-04-SDP",
        "MOT17-05-DPM", "MOT17-05-FRCNN", "MOT17-05-SDP",
        "MOT17-09-DPM", "MOT17-09-FRCNN", "MOT17-09-SDP",
        "MOT17-10-DPM", "MOT17-10-FRCNN", "MOT17-10-SDP",
        "MOT17-11-DPM", "MOT17-11-FRCNN", "MOT17-11-SDP",
        "MOT17-13-DPM", "MOT17-13-FRCNN", "MOT17-13-SDP"
    ]
    
    model = YOLO(model_path)
    # Ensure tracker state is clean
    if hasattr(model, "predictor") and model.predictor is not None:
        if hasattr(model.predictor, "trackers"):
            model.predictor.trackers = []
    
    acc = mm.MOTAccumulator(auto_id=True)
    profiler = ResourceProfiler()
    profiler.start()
    
    inference_latencies = []
    total_frames = 0
    start_time = time.time()
    
    print("Evaluating sequences for 11B-0...")
    for seq in expected_sequences:
        seq_path = os.path.join(dataset_dir, seq)
        if not os.path.exists(seq_path):
            continue
            
        # Re-instantiate model to ensure clean tracker state for each sequence
        model = YOLO(model_path)
                
        import configparser
        seqinfo = configparser.ConfigParser()
        seqinfo.read(os.path.join(seq_path, 'seqinfo.ini'))
        fps = int(seqinfo['Sequence']['frameRate'])
        seq_length = int(seqinfo['Sequence']['seqLength'])
        
        gt_file = os.path.join(seq_path, 'gt', 'gt.txt')
        gt_df = pd.read_csv(gt_file, header=None)
        gt_df.columns = ['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'class', 'visibility']
        gt_df = gt_df[gt_df['class'] == 1]
        
        frame_stride = fps # Phase 8B used stride = fps for 1FPS mode
        img_dir = os.path.join(seq_path, 'img1')
        
        for frame_idx in range(1, seq_length + 1, frame_stride):
            img_path = os.path.join(img_dir, f"{frame_idx:06d}.jpg")
            if not os.path.exists(img_path):
                continue
                
            frame = cv2.imread(img_path)
            total_frames += 1
            
            t0 = time.time()
            results = model.track(
                frame, 
                persist=True, 
                tracker="bytetrack.yaml", 
                classes=[0], 
                conf=0.40, 
                imgsz=640, 
                device="mps", 
                verbose=False
            )
            t1 = time.time()
            inference_latencies.append((t1 - t0) * 1000)
            
            pred_boxes = []
            pred_ids = []
            
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids = results[0].boxes.id.cpu().numpy()
                for box, tid in zip(boxes, ids):
                    x1, y1, x2, y2 = box
                    pred_boxes.append([x1, y1, x2 - x1, y2 - y1])
                    pred_ids.append(int(tid))
                    
            frame_gt = gt_df[gt_df['frame'] == frame_idx]
            gt_boxes = frame_gt[['bb_left', 'bb_top', 'bb_width', 'bb_height']].values.tolist()
            gt_ids = frame_gt['id'].values.tolist()
            
            dist_matrix = compute_iou_matrix(gt_boxes, pred_boxes, max_iou=0.5)
            acc.update(gt_ids, pred_ids, dist_matrix)

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
    
    # Assert against Phase 8B baseline
    # The expected values are from 8B_tracking_results.json 1_fps mode
    # Note: In Phase 8B, the image size used was 480 and conf was 0.35!
    # Wait, the 8B_tracking_results.json says: 
    #   "confidence_threshold": 0.35, "image_size": 480
    # But the user directive explicitly requested:
    #   "imgsz: 640", "conf: 0.40"
    # Because 10A standardized on 640/0.40. I must run the script and assert it manually.
    
    manifest = {
        "experiment_id": "11B_0_BASELINE_REPRODUCTION",
        "git_commit": get_git_commit(),
        "model_hash": actual_hash,
        "config_hash": "frozen_baseline",
        "dataset": {
            "name": "MOT17",
            "sampling_protocol": "stride=fps (~1.2 FPS average)",
            "dataset_hash": hash_file(os.path.join(dataset_dir, expected_sequences[0], 'gt', 'gt.txt'))
        },
        "evaluation": {
            "protocol_version": "11B-1.0",
            "thresholds": {
                "conf": 0.40,
                "imgsz": 640,
                "iou": 0.50
            }
        },
        "metrics": {
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
            }
        }
    }
    
    os.makedirs("/Users/hariharans/Documents/SIH26187/reports/phase11", exist_ok=True)
    with open("/Users/hariharans/Documents/SIH26187/reports/phase11/11B_0_baseline.json", "w") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"MOTA: {mota:.4f}, IDF1: {idf1:.4f}")
    print(f"TP: {tp}, FP: {fp}, FN: {fn}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}")
    print(f"ID Switches: {num_switches} ({idsw_per_1k:.2f}/1k)")
    print(f"MT: {mostly_tracked}, ML: {mostly_lost}, Frag: {num_fragmentations}")

if __name__ == "__main__":
    run_11b_0_baseline()
