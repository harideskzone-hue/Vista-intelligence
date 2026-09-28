#!/usr/bin/env python3
import os
import sys
import time
import json
import psutil
import configparser
import cv2
import numpy as np
import motmetrics as mm
import pandas as pd
from ultralytics import YOLO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from edge.boundary.adapter import BoundaryTrackingAdapter

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

def evaluate_sequence(seq_path, match_thresh, track_buffer, imgsz, target_fps=1.2):
    seqinfo = configparser.ConfigParser()
    seqinfo.read(os.path.join(seq_path, "seqinfo.ini"))
    source_fps = float(seqinfo["Sequence"]["frameRate"])
    seq_length = int(seqinfo["Sequence"]["seqLength"])
    gt_df = load_mot_gt(os.path.join(seq_path, "gt", "gt.txt"))
    img_dir = os.path.join(seq_path, "img1")
    
    acc_trk = mm.MOTAccumulator(auto_id=True)
    
    adapter = BoundaryTrackingAdapter()
    
    # We must patch the tracker args at runtime inside the adapter's model
    # To do this safely, we wait until the first prediction builds the tracker, then modify it.
    
    processed_frames = 0
    sampled_indices = []
    next_target_time = 0.0
    sampling_interval = 1.0 / target_fps
    
    for frame_idx in range(1, seq_length + 1):
        frame_time = (frame_idx - 1) / source_fps
        if frame_time >= next_target_time - 1e-5:
            sampled_indices.append(frame_idx)
            next_target_time += sampling_interval
            
    latency_sum = 0
                
    for frame_idx in sampled_indices:
        img_path = os.path.join(img_dir, f"{frame_idx:06d}.jpg")
        if not os.path.exists(img_path): continue
        frame = cv2.imread(img_path)
        timestamp = (frame_idx - 1) / source_fps
        
        # Override adapter parameters right before detection if the tracker exists, or force the config
        # Actually, adapter uses model.track(). Let's force it.
        adapter._profile_locked = True
        
        # We need to pass imgsz to adapter model
        # adapter.detect_and_track handles this by calling model.track()
        # We can directly invoke model.track with custom args to get exact override, or patch tracker.
        
        t0 = time.time()
        results = adapter.model.track(frame, conf=0.40, imgsz=imgsz, classes=[0], persist=True, verbose=False)
        
        if adapter.model.predictor and hasattr(adapter.model.predictor, "trackers") and len(adapter.model.predictor.trackers) > 0:
            t = adapter.model.predictor.trackers[0]
            t.args.match_thresh = match_thresh
            t.args.track_buffer = track_buffer
            t.max_frames_lost = track_buffer
            
        latency_sum += (time.time() - t0)
            
        trk_boxes, trk_ids = [], []
        if results and len(results) > 0 and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy()
            for b, tid in zip(boxes, ids):
                trk_boxes.append([b[0], b[1], b[2]-b[0], b[3]-b[1]])
                trk_ids.append(int(tid))
                
        frame_gt = gt_df[gt_df["frame"] == frame_idx]
        gt_boxes = frame_gt[["bb_left", "bb_top", "bb_width", "bb_height"]].values.tolist()
        gt_ids = frame_gt["id"].values.tolist()
        
        dist_trk = compute_iou_matrix(gt_boxes, trk_boxes, max_iou=0.5)
        acc_trk.update(gt_ids, trk_ids, dist_trk)
        processed_frames += 1
        
    return {"acc_trk": acc_trk, "latency_sum": latency_sum, "frames": processed_frames}

def process_summary(mh, accs, names):
    metrics_of_interest = ["mota", "idf1", "num_false_positives", "num_misses", "num_switches", "num_objects", "num_predictions", "mostly_tracked", "mostly_lost"]
    summary = mh.compute_many(accs, metrics=mm.metrics.motchallenge_metrics + ["num_objects", "num_predictions"], names=names, generate_overall=True)
    agg = summary.loc["OVERALL", metrics_of_interest].to_dict()
    for k, v in agg.items(): agg[k] = float(v) if not pd.isna(v) else None
    gt_objs = agg["num_objects"]
    preds = agg["num_predictions"]
    fn = agg["num_misses"]
    idsw = agg["num_switches"]
    tp = gt_objs - fn
    agg["det_recall"] = tp / gt_objs if gt_objs > 0 else 0
    agg["det_precision"] = tp / preds if preds > 0 else 0
    agg["idsw_per_1k_gt"] = (idsw / gt_objs * 1000) if gt_objs > 0 else 0
    return agg

def get_sys_metrics():
    proc = psutil.Process()
    return {
        "cpu_percent": psutil.cpu_percent(),
        "ram_mb": proc.memory_info().rss / (1024*1024)
    }

def run_experiments():
    print("Starting Phase 10B Experiments...")
    out_dir = os.path.join(os.path.dirname(__file__), "..", "reports", "phase10")
    os.makedirs(out_dir, exist_ok=True)
    
    mot17_train_dir = os.path.join(os.path.dirname(__file__), "..", "data", "external", "mot17", "MOT17", "train")
    if not os.path.exists(mot17_train_dir):
        print("MOT17 dataset not found. Blocked.")
        sys.exit(1)
        
    # Limit to 3 sequences for time efficiency in the script, as in Phase 8 validation
    sequences = sorted([os.path.join(mot17_train_dir, d) for d in os.listdir(mot17_train_dir) if os.path.isdir(os.path.join(mot17_train_dir, d))])[:3]
    mh = mm.metrics.create()
    
    results = {"B1": {}, "B2": {}, "B3": {}}
    
    # ------------------ B1 ------------------
    print("Running B1 - ByteTrack Sensitivity")
    b1_configs = [
        (0.80, 30), (0.70, 30), (0.60, 30),
        (0.80, 60), (0.70, 60), (0.60, 60)
    ]
    for mt, tb in b1_configs:
        print(f"B1: match_thresh={mt}, track_buffer={tb}")
        accs = []
        for seq in sequences:
            res = evaluate_sequence(seq, match_thresh=mt, track_buffer=tb, imgsz=640)
            accs.append(res["acc_trk"])
        agg = process_summary(mh, accs, [os.path.basename(s) for s in sequences])
        results["B1"][f"mt{mt}_tb{tb}"] = agg

    # ------------------ B2 ------------------
    print("Running B2 - YOLO Resolution")
    b2_configs = [640, 480, 320]
    for sz in b2_configs:
        print(f"B2: imgsz={sz}")
        accs = []
        lat_sum = 0
        frames = 0
        sys_metrics = get_sys_metrics()
        for seq in sequences:
            res = evaluate_sequence(seq, match_thresh=0.80, track_buffer=30, imgsz=sz)
            accs.append(res["acc_trk"])
            lat_sum += res["latency_sum"]
            frames += res["frames"]
        agg = process_summary(mh, accs, [os.path.basename(s) for s in sequences])
        agg["latency_ms"] = (lat_sum / frames) * 1000 if frames > 0 else 0
        agg["fps"] = frames / lat_sum if lat_sum > 0 else 0
        agg["cpu_percent"] = sys_metrics["cpu_percent"]
        agg["ram_mb"] = sys_metrics["ram_mb"]
        results["B2"][f"imgsz_{sz}"] = agg

    # ------------------ B3 ------------------
    print("Running B3 - Batch Inference on MPS")
    model = YOLO("yolo11n.pt")
    device = "mps" # Forced MPS target as required
    try:
        # Load a dummy image
        seq = sequences[0]
        img_path = os.path.join(seq, "img1", "000001.jpg")
        frame = cv2.imread(img_path)
        
        batch_sizes = [1, 2, 4]
        for bs in batch_sizes:
            print(f"B3: batch_size={bs}")
            batch = [frame.copy() for _ in range(bs)]
            
            # warmup
            for _ in range(3):
                model.predict(batch, imgsz=640, device=device, verbose=False)
                
            t0 = time.time()
            iters = 10
            for _ in range(iters):
                model.predict(batch, imgsz=640, device=device, verbose=False)
            t1 = time.time()
            
            sys_metrics = get_sys_metrics()
            
            total_frames = bs * iters
            total_time = t1 - t0
            results["B3"][f"batch_{bs}"] = {
                "images_per_sec": total_frames / total_time,
                "batch_latency_ms": (total_time / iters) * 1000,
                "per_frame_latency_ms": (total_time / total_frames) * 1000,
                "cpu_percent": sys_metrics["cpu_percent"],
                "ram_mb": sys_metrics["ram_mb"]
            }
    except Exception as e:
        print(f"B3 MPS Test Failed: {e}")
        results["B3"]["error"] = str(e)
        
    out_json = os.path.join(out_dir, "phase10b_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {out_json}")

if __name__ == "__main__":
    run_experiments()
