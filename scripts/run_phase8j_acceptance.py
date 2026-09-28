import os
import sys
import time
import json
import psutil
import threading
import configparser
import cv2
import numpy as np
import motmetrics as mm
import pandas as pd
import uuid

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from edge.boundary.adapter import BoundaryTrackingAdapter
from ultralytics import YOLO

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

def evaluate_sequence(seq_path, adapter, raw_model, target_fps=None, force_static=False):
    seqinfo = configparser.ConfigParser()
    seqinfo.read(os.path.join(seq_path, "seqinfo.ini"))
    source_fps = float(seqinfo["Sequence"]["frameRate"])
    seq_length = int(seqinfo["Sequence"]["seqLength"])
    gt_df = load_mot_gt(os.path.join(seq_path, "gt", "gt.txt"))
    img_dir = os.path.join(seq_path, "img1")
    
    acc_raw = mm.MOTAccumulator(auto_id=True)
    acc_trk = mm.MOTAccumulator(auto_id=True)
    
    processed_frames = 0
    sampled_indices = []
    next_target_time = 0.0
    sampling_interval = (1.0 / target_fps) if target_fps else 0.0
    
    for frame_idx in range(1, seq_length + 1):
        frame_time = (frame_idx - 1) / source_fps
        if target_fps is None or frame_time >= next_target_time - 1e-5:
            sampled_indices.append(frame_idx)
            if target_fps is not None: next_target_time += sampling_interval
                
    for frame_idx in sampled_indices:
        img_path = os.path.join(img_dir, f"{frame_idx:06d}.jpg")
        if not os.path.exists(img_path): continue
        frame = cv2.imread(img_path)
        timestamp = (frame_idx - 1) / source_fps
        
        # RAW YOLO
        res_raw = raw_model.predict(frame, conf=0.35, imgsz=480, classes=[0], verbose=False)
        raw_boxes = []
        if res_raw and len(res_raw) > 0 and res_raw[0].boxes is not None:
            for b in res_raw[0].boxes:
                xyxy = b.xyxy[0].cpu().numpy().tolist()
                raw_boxes.append([xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]])
                
        global_raw_id = getattr(evaluate_sequence, "global_raw_id", 0)
        raw_ids = []
        for _ in raw_boxes:
            global_raw_id += 1
            raw_ids.append(global_raw_id)
        evaluate_sequence.global_raw_id = global_raw_id
        
        # FORCE STATIC IF NEEDED
        if force_static:
            adapter._profile_locked = True
            if adapter.model is not None and hasattr(adapter.model, "predictor") and adapter.model.predictor is not None:
                if hasattr(adapter.model.predictor, "trackers") and len(adapter.model.predictor.trackers) > 0:
                    t = adapter.model.predictor.trackers[0]
                    t.max_frames_lost = 30
                    t.args.match_thresh = 0.8
        
        # TRACKED
        observations = adapter.detect_and_track(frame, timestamp=timestamp)
        trk_boxes, trk_ids = [], []
        for obs in observations:
            if obs.class_name == "person":
                xmin, ymin, xmax, ymax = obs.bbox
                trk_boxes.append([xmin, ymin, xmax - xmin, ymax - ymin])
                trk_ids.append(obs.track_id)
                
        # GT Matching
        frame_gt = gt_df[gt_df["frame"] == frame_idx]
        gt_boxes = frame_gt[["bb_left", "bb_top", "bb_width", "bb_height"]].values.tolist()
        gt_ids = frame_gt["id"].values.tolist()
        
        dist_raw = compute_iou_matrix(gt_boxes, raw_boxes, max_iou=0.5)
        dist_trk = compute_iou_matrix(gt_boxes, trk_boxes, max_iou=0.5)
        
        acc_raw.update(gt_ids, raw_ids, dist_raw)
        acc_trk.update(gt_ids, trk_ids, dist_trk)
        processed_frames += 1
        
    return {"acc_raw": acc_raw, "acc_trk": acc_trk}

def process_summary(mh, accs, names):
    metrics_of_interest = ["mota", "idf1", "num_false_positives", "num_misses", "num_switches", "num_objects", "num_predictions", "mostly_tracked", "mostly_lost"]
    summary = mh.compute_many(accs, metrics=mm.metrics.motchallenge_metrics + ["num_objects", "num_predictions"], names=names, generate_overall=True)
    agg = summary.loc["OVERALL", metrics_of_interest].to_dict()
    for k, v in agg.items(): agg[k] = float(v) if not pd.isna(v) else None
    gt_objs = agg["num_objects"]
    preds = agg["num_predictions"]
    fp = agg["num_false_positives"]
    fn = agg["num_misses"]
    idsw = agg["num_switches"]
    
    tp = gt_objs - fn
    agg["tp"] = tp
    agg["det_recall"] = tp / gt_objs if gt_objs > 0 else 0
    agg["det_precision"] = tp / preds if preds > 0 else 0
    agg["idsw_per_1k_gt"] = (idsw / gt_objs * 1000) if gt_objs > 0 else 0
    return agg

def run_experiment():
    mot17_train_dir = "data/external/mot17/MOT17/train"
    # ALL 21 VARIANTS
    sequences = sorted([os.path.join(mot17_train_dir, d) for d in os.listdir(mot17_train_dir) if os.path.isdir(os.path.join(mot17_train_dir, d))])
    
    mh = mm.metrics.create()
    results = {}
    
    configs = [
        ("Native", None, False),
        ("Static_1.2", 1.2, True),
        ("Dynamic_1.2", 1.2, False)
    ]
    
    for name, fps, is_static in configs:
        print(f"Evaluating {name}...")
        accs_raw, accs_trk, seq_names = [], [], []
        for seq_path in sequences:
            seq_name = os.path.basename(seq_path)
            adapter = BoundaryTrackingAdapter()
            if adapter.model is not None and __import__("torch").backends.mps.is_available(): adapter.model.to("mps")
            
            raw_model = YOLO(adapter._model_path)
            if __import__("torch").backends.mps.is_available(): raw_model.to("mps")
            
            res = evaluate_sequence(seq_path, adapter, raw_model, target_fps=fps, force_static=is_static)
            accs_raw.append(res["acc_raw"])
            accs_trk.append(res["acc_trk"])
            seq_names.append(seq_name)
            
        results[name] = {
            "raw": process_summary(mh, accs_raw, seq_names),
            "trk": process_summary(mh, accs_trk, seq_names)
        }
    
    return results

def main():
    print("Starting Phase 8J Acceptance Evaluation (21 Variants)...")
    res = run_experiment()
    with open("reports/phase8/8J_acceptance_results.json", "w") as f:
        json.dump(res, f, indent=2)
    print("Saved 8J_acceptance_results.json")

if __name__ == "__main__":
    main()
