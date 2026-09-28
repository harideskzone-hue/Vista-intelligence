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

def run_experiment(exp_id, imgsz, conf, model_path, dataset_dir, expected_sequences):
    print(f"\n--- Running Experiment: {exp_id} ---")
    print(f"Config: imgsz={imgsz}, conf={conf}")
    
    actual_hash = hash_file(model_path)
    model = YOLO(model_path)
    
    profiler = ResourceProfiler()
    profiler.start()
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    inference_latencies = []
    total_frames = 0
    start_time = time.time()
    
    for seq in expected_sequences:
        seq_path = os.path.join(dataset_dir, seq)
        if not os.path.exists(seq_path):
            continue
            
        import configparser
        seqinfo = configparser.ConfigParser()
        seqinfo.read(os.path.join(seq_path, 'seqinfo.ini'))
        fps = int(seqinfo['Sequence']['frameRate'])
        seq_length = int(seqinfo['Sequence']['seqLength'])
        
        gt_file = os.path.join(seq_path, 'gt', 'gt.txt')
        gt_df = pd.read_csv(gt_file, header=None)
        gt_df.columns = ['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'class', 'visibility']
        gt_df = gt_df[gt_df['class'] == 1]
        
        frame_stride = fps
        img_dir = os.path.join(seq_path, 'img1')
        
        for frame_idx in range(1, seq_length + 1, frame_stride):
            img_path = os.path.join(img_dir, f"{frame_idx:06d}.jpg")
            if not os.path.exists(img_path):
                continue
                
            frame = cv2.imread(img_path)
            total_frames += 1
            
            t0 = time.time()
            results = model.predict(frame, imgsz=imgsz, conf=conf, classes=[0], device="mps", verbose=False)
            t1 = time.time()
            inference_latencies.append((t1 - t0) * 1000) # in ms
            
            pred_boxes = []
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                pred_boxes.append([x1, y1, x2 - x1, y2 - y1])
                
            frame_gt = gt_df[gt_df['frame'] == frame_idx]
            gt_boxes = frame_gt[['bb_left', 'bb_top', 'bb_width', 'bb_height']].values.tolist()
            
            dist_matrix = compute_iou_matrix(gt_boxes, pred_boxes, max_iou=0.5)
            matches, matched_gt, matched_pred = greedy_match(dist_matrix)
            
            total_tp += len(matches)
            total_fp += len(pred_boxes) - len(matches)
            total_fn += len(gt_boxes) - len(matches)

    profiler.stop()
    total_time = time.time() - start_time
    throughput_fps = total_frames / total_time if total_time > 0 else 0
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    p50_latency = np.percentile(inference_latencies, 50) if inference_latencies else 0
    p95_latency = np.percentile(inference_latencies, 95) if inference_latencies else 0
    
    resource_stats = profiler.get_stats()
    
    manifest = {
        "experiment_id": exp_id,
        "git_commit": get_git_commit(),
        "model_hash": actual_hash,
        "hardware": {
            "device": "mps",
            "os": os.uname().sysname,
            "compute_provider": "Apple MPS"
        },
        "dataset": {
            "name": "MOT17",
            "source": "MOTChallenge",
            "sequences": expected_sequences,
            "annotation_source": "gt.txt",
            "target_class": "pedestrian",
            "visibility_policy": "same as Phase 8B",
            "sampling_protocol": "same as Phase 8B (stride=fps)",
            "dataset_hash": hash_file(os.path.join(dataset_dir, expected_sequences[0], 'gt', 'gt.txt'))
        },
        "evaluation": {
            "protocol_version": "11A-1.0",
            "sampling_fps": 1.0,
            "thresholds": {
                "conf": conf,
                "imgsz": imgsz,
                "iou": 0.50
            }
        },
        "metrics": {
            "accuracy": {
                "TP": total_tp,
                "FP": total_fp,
                "FN": total_fn,
                "precision": precision,
                "recall": recall,
                "f1": f1
            },
            "latency": {
                "p50_ms": p50_latency,
                "p95_ms": p95_latency
            },
            "throughput": {
                "fps": throughput_fps
            }
        },
        "resource_usage": resource_stats
    }
    
    os.makedirs("/Users/hariharans/Documents/SIH26187/reports/phase11", exist_ok=True)
    with open(f"/Users/hariharans/Documents/SIH26187/reports/phase11/{exp_id}.json", "w") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"Results saved to reports/phase11/{exp_id}.json")
    print(f"TP: {total_tp}, FP: {total_fp}, FN: {total_fn}, Prec: {precision:.4f}, Rec: {recall:.4f}, F1: {f1:.4f}")
    print(f"P50: {p50_latency:.2f}ms, P95: {p95_latency:.2f}ms, FPS: {throughput_fps:.2f}")
    
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
    
    # 11A-1: Image Size
    for imgsz in [576, 512, 480]:
        run_experiment(f"11A_1_imgsz_{imgsz}", imgsz=imgsz, conf=0.40, model_path=model_path, dataset_dir=dataset_dir, expected_sequences=expected_sequences)
        
    # 11A-2: Confidence
    for conf in [0.20, 0.30, 0.50]:
        run_experiment(f"11A_2_conf_{int(conf*100)}", imgsz=640, conf=conf, model_path=model_path, dataset_dir=dataset_dir, expected_sequences=expected_sequences)

if __name__ == "__main__":
    main()
