import os
import sys
import glob
import time
import json
import psutil
import argparse
import configparser
import subprocess
import cv2
import numpy as np
import motmetrics as mm
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from edge.boundary.adapter import BoundaryTrackingAdapter

def load_mot_gt(gt_file):
    gt = pd.read_csv(gt_file, header=None)
    gt.columns = ['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'class', 'visibility']
    gt = gt[gt['class'] == 1]
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

def evaluate_sequence(seq_path, adapter, target_fps=None):
    seqinfo = configparser.ConfigParser()
    seqinfo.read(os.path.join(seq_path, 'seqinfo.ini'))
    source_fps = float(seqinfo['Sequence']['frameRate'])
    seq_length = int(seqinfo['Sequence']['seqLength'])
    gt_df = load_mot_gt(os.path.join(seq_path, 'gt', 'gt.txt'))
    img_dir = os.path.join(seq_path, 'img1')
    
    acc = mm.MOTAccumulator(auto_id=True)
    processed_frames = 0
    sampled_indices = []
    
    next_target_time = 0.0
    sampling_interval = (1.0 / target_fps) if target_fps else 0.0
    
    adapter = BoundaryTrackingAdapter()
    if adapter.model is not None and __import__('torch').backends.mps.is_available(): adapter.model.to('mps')
    
    process = psutil.Process()
    mem_start = process.memory_info().rss
    start_time = time.time()
    
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
        observations = adapter.detect_and_track(frame, timestamp=timestamp)
        processed_frames += 1
        
        pred_boxes, pred_ids = [], []
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
    mem_end = process.memory_info().rss
    return {
        "acc": acc,
        "stats": {
            "target_fps": target_fps if target_fps else "Native",
            "source_fps": source_fps,
            "effective_fps": processed_frames / (seq_length / source_fps) if seq_length > 0 else 0,
            "sampled_frames": processed_frames,
            "source_frames": seq_length,
            "sampling_ratio": processed_frames / seq_length if seq_length > 0 else 0,
            "sampling_interval_sec": sampling_interval,
            "runtime": duration,
            "processing_fps": processed_frames / duration if duration > 0 else 0,
            "memory_mb": max(0, mem_end - mem_start) / (1024 * 1024),
            "cpu_util": process.cpu_percent()
        }
    }

def main():
    adapter = BoundaryTrackingAdapter()
    if adapter.model is not None and __import__('torch').backends.mps.is_available(): adapter.model.to('mps')
    if not adapter.is_model_loaded: sys.exit(1)
        
    mot17_train_dir = "data/external/mot17/MOT17/train"
    sequences = sorted([os.path.join(mot17_train_dir, d) for d in os.listdir(mot17_train_dir) if os.path.isdir(os.path.join(mot17_train_dir, d)) and d.endswith("-FRCNN")])
    
    target_rates = [None, 15, 10, 5, 2, 1.2, 1]
    results_data = {"rates": {}}
    mh = mm.metrics.create()
    
    for rate in target_rates:
        rate_key = "Native" if rate is None else str(rate)
        accs, names, mode_stats = [], [], {}
        
        for seq_path in sequences:
            seq_name = os.path.basename(seq_path)
            res = evaluate_sequence(seq_path, adapter, target_fps=rate)
            accs.append(res["acc"])
            names.append(seq_name)
            mode_stats[seq_name] = res["stats"]
            
        summary = mh.compute_many(accs, metrics=mm.metrics.motchallenge_metrics + ['num_objects', 'num_predictions'], names=names, generate_overall=True)
        metrics_of_interest = ['mota', 'idf1', 'mostly_tracked', 'mostly_lost', 'num_false_positives', 'num_misses', 'num_switches', 'num_fragmentations', 'num_objects', 'num_predictions']
        
        def safe_float(v): return float(v) if not pd.isna(v) else None
            
        agg = summary.loc["OVERALL", metrics_of_interest].to_dict()
        for k, v in agg.items(): agg[k] = safe_float(v)
        
        gt_objs, preds, fp, fn, idsw = agg['num_objects'], agg['num_predictions'], agg['num_false_positives'], agg['num_misses'], agg['num_switches']
        tp = gt_objs - fn
        
        agg['tp'] = tp
        agg['det_recall'] = tp / gt_objs if gt_objs > 0 else 0
        agg['det_precision'] = tp / preds if preds > 0 else 0
        agg['fp_rate'] = fp / gt_objs if gt_objs > 0 else 0
        agg['fn_rate'] = fn / gt_objs if gt_objs > 0 else 0
        agg['idsw_per_1k_gt'] = (idsw / gt_objs * 1000) if gt_objs > 0 else 0
        
        total_runtime = sum([s['runtime'] for s in mode_stats.values()])
        total_frames = sum([s['sampled_frames'] for s in mode_stats.values()])
        agg['total_runtime'] = total_runtime
        agg['processing_fps'] = total_frames / total_runtime if total_runtime > 0 else 0
        
        results_data["rates"][rate_key] = {"metrics": agg, "stats": mode_stats}
        
    os.makedirs("reports/phase8", exist_ok=True)
    with open("reports/phase8/8G_degradation_results.json", "w") as f: json.dump(results_data, f, indent=2)
        
    with open("reports/phase8/8G_degradation_report.md", "w") as f:
        f.write("# Phase 8G: Sampling Degradation Study\n\n")
        f.write("## 1. Methodology\n")
        f.write("- **Dataset**: 7 unique MOT17 underlying sequences. (Sanity check verified that DPM/FRCNN/SDP subsets are fully redundant as our pipeline ignores MOTChallenge `det.txt` boxes).\n")
        f.write("- **Tracker**: `yolo11n.pt` + `bytetrack.yaml` (Production Configuration Frozen)\n")
        f.write("- **Temporal Sparsity**: Tracker state is NOT reset between sampled frames. Interstitial frames are entirely skipped. Time-based fractional sampling is used (e.g. 1.2 target FPS).\n")
        f.write("- **Evaluation Focus**: Measuring operational degradation caused strictly by temporal sparsity to identify where continuity breaks down.\n\n")
        
        f.write("## 2. Overall Degradation Results\n\n")
        f.write("### Tracking Continuity (Relative to Native)\n")
        f.write("| Sampling | MOTA | Δ MOTA | IDF1 | Δ IDF1 | IDSW / 1K GT | Δ IDSW / 1K GT | Mostly Tracked | Mostly Lost |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        
        native_mota = results_data["rates"]["Native"]["metrics"]["mota"]
        native_idf1 = results_data["rates"]["Native"]["metrics"]["idf1"]
        native_idsw = results_data["rates"]["Native"]["metrics"]["idsw_per_1k_gt"]
        
        for rate in ["Native", "15", "10", "5", "2", "1.2", "1"]:
            m = results_data["rates"][rate]["metrics"]
            dmota = m["mota"] - native_mota if rate != "Native" else 0
            didf1 = m["idf1"] - native_idf1 if rate != "Native" else 0
            didsw = m["idsw_per_1k_gt"] - native_idsw if rate != "Native" else 0
            f.write(f"| {rate} | {m['mota']:.4f} | {dmota:+.4f} | {m['idf1']:.4f} | {didf1:+.4f} | {m['idsw_per_1k_gt']:.2f} | {didsw:+.2f} | {int(m['mostly_tracked'])} | {int(m['mostly_lost'])} |\n")
        
        f.write("\n### Normalized Detection Metrics\n")
        f.write("| Sampling | GT Evaluated | TP | FP | FN | Precision | Recall | FP Rate | FN Rate |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for rate in ["Native", "15", "10", "5", "2", "1.2", "1"]:
            m = results_data["rates"][rate]["metrics"]
            f.write(f"| {rate} | {int(m['num_objects'])} | {int(m['tp'])} | {int(m['num_false_positives'])} | {int(m['num_misses'])} | {m['det_precision']:.4f} | {m['det_recall']:.4f} | {m['fp_rate']:.4f} | {m['fn_rate']:.4f} |\n")
            
        f.write("\n### Sampling Performance & Runtime\n")
        f.write("| Target FPS | Total Evaluated Frames | Effective Processing FPS | Wall-Clock Runtime (s) |\n")
        f.write("|---|---|---|---|\n")
        for rate in ["Native", "15", "10", "5", "2", "1.2", "1"]:
            m = results_data["rates"][rate]["metrics"]
            total_frames = sum([s['sampled_frames'] for s in results_data["rates"][rate]["stats"].values()])
            f.write(f"| {rate} | {int(total_frames)} | {m['processing_fps']:.1f} | {m['total_runtime']:.1f} |\n")

if __name__ == '__main__':
    main()
