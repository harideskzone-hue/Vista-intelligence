import os
import sys
import glob
import time
import json
import argparse
import configparser
import subprocess
import cv2
import numpy as np
import motmetrics as mm
import pandas as pd

# Ensure we can import edge modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from edge.boundary.adapter import BoundaryTrackingAdapter

def get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    except Exception:
        return "unknown"

def hash_file(filepath):
    if not os.path.exists(filepath):
        return "missing"
    import hashlib
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()

def get_production_config(adapter):
    model_path = adapter._model_path
    return {
        "git_commit": get_git_commit(),
        "model_weights": model_path,
        "model_sha256": hash_file(model_path) if model_path else None,
        "tracker": adapter.tracker_type,
        "confidence_threshold": adapter.conf_threshold,
        "iou_threshold": 0.7, # Default YOLO track threshold
        "image_size": adapter.imgsz,
        "device": str(adapter.model.device) if adapter.model else "unknown",
        "python_version": sys.version.split(' ')[0],
        "ultralytics_version": __import__('ultralytics').__version__ if adapter.is_model_loaded else "N/A",
        "pytorch_version": __import__('torch').__version__ if adapter.is_model_loaded else "N/A"
    }

def load_mot_gt(gt_file):
    gt = pd.read_csv(gt_file, header=None)
    gt.columns = ['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'class', 'visibility']
    # Filter for pedestrians (class 1)
    gt = gt[gt['class'] == 1]
    return gt

def compute_iou_matrix(gt_boxes, pred_boxes, max_iou=0.5):
    """Safe IoU matrix function avoiding numpy 2.0 deprecated asfarray."""
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return np.empty((len(gt_boxes), len(pred_boxes)))
    
    gt_boxes = np.asarray(gt_boxes, dtype=float)
    pred_boxes = np.asarray(pred_boxes, dtype=float)
    
    gt_x1, gt_y1, gt_w, gt_h = gt_boxes[:,0], gt_boxes[:,1], gt_boxes[:,2], gt_boxes[:,3]
    gt_x2, gt_y2 = gt_x1 + gt_w, gt_y1 + gt_h
    
    pr_x1, pr_y1, pr_w, pr_h = pred_boxes[:,0], pred_boxes[:,1], pred_boxes[:,2], pred_boxes[:,3]
    pr_x2, pr_y2 = pr_x1 + pr_w, pr_y1 + pr_h
    
    # Calculate intersections
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

def evaluate_sequence(seq_path, adapter, mode="full"):
    seq_name = os.path.basename(seq_path)
    seqinfo = configparser.ConfigParser()
    seqinfo.read(os.path.join(seq_path, 'seqinfo.ini'))
    
    fps = int(seqinfo['Sequence']['frameRate'])
    seq_length = int(seqinfo['Sequence']['seqLength'])
    
    gt_file = os.path.join(seq_path, 'gt', 'gt.txt')
    gt_df = load_mot_gt(gt_file)
    
    img_dir = os.path.join(seq_path, 'img1')
    
    acc = mm.MOTAccumulator(auto_id=True)
    
    frame_stride = 1 if mode == "full" else fps  # For 1 FPS, stride = fps
    processed_frames = 0
    start_time = time.time()
    
    adapter = BoundaryTrackingAdapter()
    if adapter.model is not None and __import__('torch').backends.mps.is_available():
        adapter.model.to('mps')
    
    for frame_idx in range(1, seq_length + 1, frame_stride):
        img_path = os.path.join(img_dir, f"{frame_idx:06d}.jpg")
        if not os.path.exists(img_path):
            continue
            
        frame = cv2.imread(img_path)
        timestamp = (frame_idx - 1) / float(fps)
        
        observations = adapter.detect_and_track(frame, timestamp=timestamp)
        processed_frames += 1
        
        pred_boxes = []
        pred_ids = []
        for obs in observations:
            if obs.class_name == "person":
                xmin, ymin, xmax, ymax = obs.bbox
                pred_boxes.append([xmin, ymin, xmax - xmin, ymax - ymin])
                pred_ids.append(obs.track_id)
                
        frame_gt = gt_df[gt_df['frame'] == frame_idx]
        gt_boxes = frame_gt[['bb_left', 'bb_top', 'bb_width', 'bb_height']].values.tolist()
        gt_ids = frame_gt['id'].values.tolist()
        
        distances = compute_iou_matrix(gt_boxes, pred_boxes, max_iou=0.5)
        acc.update(gt_ids, pred_ids, distances)
        
    duration = time.time() - start_time
    
    return {
        "acc": acc,
        "stats": {
            "source_fps": fps,
            "sampled_fps": fps / frame_stride,
            "source_frame_count": seq_length,
            "processed_frame_count": processed_frames,
            "frame_skip_ratio": (seq_length - processed_frames) / float(seq_length),
            "gt_object_count": len(gt_df),
            "predicted_track_count": len(set([o.track_id for o in adapter._cached_observations])) if processed_frames > 0 else 0,
            "runtime": duration
        }
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    
    print("Loading production tracker...")
    adapter = BoundaryTrackingAdapter()
    if adapter.model is not None and __import__('torch').backends.mps.is_available():
        adapter.model.to('mps')
    if not adapter.is_model_loaded:
        print("[!] FATAL: Production tracker model failed to load. Interface mismatch or missing model.")
        sys.exit(1)
        
    config = get_production_config(adapter)
    print(f"Configuration captured: {json.dumps(config, indent=2)}")
    
    mot17_train_dir = "data/external/mot17/MOT17/train"
    if not os.path.exists(mot17_train_dir):
        print(f"[!] FATAL: {mot17_train_dir} does not exist.")
        sys.exit(1)
        
    sequences = sorted([os.path.join(mot17_train_dir, d) for d in os.listdir(mot17_train_dir) if os.path.isdir(os.path.join(mot17_train_dir, d))])
    
    if args.smoke_test:
        sequences = sequences[:1]
        print(f"Running smoke test on single sequence: {sequences[0]}")
        
    results_data = {
        "configuration": config,
        "modes": {
            "full_frame": {},
            "1_fps": {}
        }
    }
    
    mh = mm.metrics.create()
    
    for mode in ["full", "1_fps"]:
        print(f"\n--- Running evaluation mode: {mode} ---")
        accs = []
        names = []
        mode_stats = {}
        
        for seq_path in sequences:
            seq_name = os.path.basename(seq_path)
            print(f"Evaluating {seq_name}...")
            res = evaluate_sequence(seq_path, adapter, mode=mode)
            accs.append(res["acc"])
            names.append(seq_name)
            mode_stats[seq_name] = res["stats"]
            
        summary = mh.compute_many(
            accs, 
            metrics=mm.metrics.motchallenge_metrics, 
            names=names,
            generate_overall=True
        )
        
        metrics_of_interest = ['mota', 'idf1', 'mostly_tracked', 'mostly_lost', 'num_false_positives', 'num_misses', 'num_switches', 'num_fragmentations']
        
        print(summary[metrics_of_interest])
        
        def safe_float(v):
            if pd.isna(v): return None
            return float(v)
            
        agg_metrics = summary.loc["OVERALL", metrics_of_interest].to_dict()
        for k, v in agg_metrics.items(): agg_metrics[k] = safe_float(v)
        
        seq_metrics_dict = summary.loc[names, metrics_of_interest].to_dict(orient="index")
        for seq, seq_metrics in seq_metrics_dict.items():
            for k, v in seq_metrics.items(): seq_metrics[k] = safe_float(v)
            
        results_data["modes"][mode] = {
            "aggregate_metrics": agg_metrics,
            "sequence_metrics": seq_metrics_dict,
            "sequence_stats": mode_stats,
            "HOTA": "NOT MEASURED"
        }
        
    os.makedirs("reports/phase8", exist_ok=True)
    with open("reports/phase8/8B_tracking_results.json", "w") as f:
        json.dump(results_data, f, indent=2)
        
    with open("reports/phase8/8B_tracking_report.md", "w") as f:
        f.write("# Phase 8B: MOT17 Tracking Evaluation Report\n\n")
        f.write("## 1. Production Configuration\n")
        for k, v in config.items():
            f.write(f"- **{k}**: `{v}`\n")
        
        for mode in ["full", "1_fps"]:
            f.write(f"\n## 2. Evaluation Mode: {mode.upper()}\n")
            f.write("### Aggregate Metrics\n")
            f.write("| Metric | Value |\n|---|---|\n")
            for k, v in results_data["modes"][mode]["aggregate_metrics"].items():
                f.write(f"| {k} | {v:.4f} |\n" if v is not None else f"| {k} | N/A |\n")
            f.write(f"| HOTA | NOT MEASURED |\n")
            
    print("\n[✓] Evaluation complete. Results saved to reports/phase8/8B_tracking_results.json and 8B_tracking_report.md")

if __name__ == "__main__":
    main()
