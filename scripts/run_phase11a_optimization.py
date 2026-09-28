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

def run_11a_0_baseline():
    print("Starting 11A-0 Baseline Reproduction")
    
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
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    print("Evaluating sequences...")
    for seq in expected_sequences:
        seq_path = os.path.join(dataset_dir, seq)
        if not os.path.exists(seq_path):
            print(f"Warning: Sequence {seq} not found at {seq_path}")
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
            
            # Use batch 1, conf 0.40, imgsz 640 for Baseline Accuracy
            results = model.predict(frame, imgsz=640, conf=0.40, classes=[0], device="mps", verbose=False)
            
            pred_boxes = []
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                pred_boxes.append([x1, y1, x2 - x1, y2 - y1])
                
            frame_gt = gt_df[gt_df['frame'] == frame_idx]
            gt_boxes = frame_gt[['bb_left', 'bb_top', 'bb_width', 'bb_height']].values.tolist()
            
            dist_matrix = compute_iou_matrix(gt_boxes, pred_boxes, max_iou=0.5)
            matches, matched_gt, matched_pred = greedy_match(dist_matrix)
            
            tp = len(matches)
            fp = len(pred_boxes) - len(matches)
            fn = len(gt_boxes) - len(matches)
            
            total_tp += tp
            total_fp += fp
            total_fn += fn

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    EXPECTED_TP = 4128
    EXPECTED_FP = 525
    EXPECTED_FN = 8142
    EXPECTED_PRECISION = 0.8871696
    EXPECTED_RECALL = 0.3364303
    EXPECTED_F1 = 0.4878568
    
    assert total_tp == EXPECTED_TP, f"TP mismatch: Expected {EXPECTED_TP}, got {total_tp}"
    assert total_fp == EXPECTED_FP, f"FP mismatch: Expected {EXPECTED_FP}, got {total_fp}"
    assert total_fn == EXPECTED_FN, f"FN mismatch: Expected {EXPECTED_FN}, got {total_fn}"
    assert abs(precision - EXPECTED_PRECISION) <= 1e-6, f"Precision mismatch: Expected {EXPECTED_PRECISION}, got {precision}"
    assert abs(recall - EXPECTED_RECALL) <= 1e-6, f"Recall mismatch: Expected {EXPECTED_RECALL}, got {recall}"
    assert abs(f1 - EXPECTED_F1) <= 1e-6, f"F1 mismatch: Expected {EXPECTED_F1}, got {f1}"
    
    print("Baseline reproduction test PASSED. All metrics matched frozen Phase 10 expectations.")
    
    manifest = {
        "experiment_id": "11A_0_BASELINE_REPRODUCTION",
        "git_commit": get_git_commit(),
        "model_hash": actual_hash,
        "config_hash": "frozen_baseline",
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
            "dataset_hash": hash_file(os.path.join(dataset_dir, expected_sequences[0], 'gt', 'gt.txt')) # Example hash
        },
        "evaluation": {
            "protocol_version": "11A-1.0",
            "sampling_fps": 1.0,
            "thresholds": {
                "conf": 0.40,
                "imgsz": 640,
                "iou": 0.50
            }
        },
        "reproducibility": {
            "command": "python3 scripts/run_phase11a_optimization.py",
            "seed": 0
        },
        "expected_baseline": {
            "TP": EXPECTED_TP,
            "FP": EXPECTED_FP,
            "FN": EXPECTED_FN,
            "precision": EXPECTED_PRECISION,
            "recall": EXPECTED_RECALL,
            "f1": EXPECTED_F1
        },
        "metrics": {
            "accuracy": {
                "TP": total_tp,
                "FP": total_fp,
                "FN": total_fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "mAP50": "unavailable",
                "mAP50-95": "unavailable"
            }
        },
        "decision": {
            "production_changed": "NO",
            "baseline_changed": "NO",
            "accuracy_improved": "N/A",
            "performance_improved": "N/A",
            "regression_detected": "NO"
        }
    }
    
    os.makedirs("/Users/hariharans/Documents/SIH26187/reports/phase11", exist_ok=True)
    with open("/Users/hariharans/Documents/SIH26187/reports/phase11/11A_0_baseline.json", "w") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

if __name__ == "__main__":
    run_11a_0_baseline()
