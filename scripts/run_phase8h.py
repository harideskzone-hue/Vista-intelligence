import os
import sys
import time
import json
import psutil
import threading
import argparse
import configparser
import subprocess
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

def evaluate_sequence(seq_path, adapter, raw_model, target_fps=None):
    seqinfo = configparser.ConfigParser()
    seqinfo.read(os.path.join(seq_path, 'seqinfo.ini'))
    source_fps = float(seqinfo['Sequence']['frameRate'])
    seq_length = int(seqinfo['Sequence']['seqLength'])
    gt_df = load_mot_gt(os.path.join(seq_path, 'gt', 'gt.txt'))
    img_dir = os.path.join(seq_path, 'img1')
    
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
                
    
    adapter = BoundaryTrackingAdapter()
    if adapter.model is not None and __import__("torch").backends.mps.is_available(): adapter.model.to("mps")
    from ultralytics import YOLO
    raw_model = YOLO(adapter._model_path)
    if __import__("torch").backends.mps.is_available(): raw_model.to("mps")
    
    process = psutil.Process()
    process.cpu_percent(interval=None) # Prime the cpu_percent
    
    cpu_samples = []
    stop_sampling = False
    def cpu_sampler():
        while not stop_sampling:
            cpu_samples.append(process.cpu_percent(interval=0.1))
    
    sampler_thread = threading.Thread(target=cpu_sampler)
    sampler_thread.start()
    
    peak_rss = process.memory_info().rss
    start_time = time.time()
    
    per_frame_diagnostics = []
    
    for frame_idx in sampled_indices:
        img_path = os.path.join(img_dir, f"{frame_idx:06d}.jpg")
        if not os.path.exists(img_path): continue
        frame = cv2.imread(img_path)
        timestamp = (frame_idx - 1) / source_fps
        
        # --- MODE A: RAW YOLO ---
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

        
        # --- MODE B: TRACKED ---
        observations = adapter.detect_and_track(frame, timestamp=timestamp)
        trk_boxes, trk_ids = [], []
        for obs in observations:
            if obs.class_name == "person":
                xmin, ymin, xmax, ymax = obs.bbox
                trk_boxes.append([xmin, ymin, xmax - xmin, ymax - ymin])
                trk_ids.append(obs.track_id)
                
        # GT Matching
        frame_gt = gt_df[gt_df['frame'] == frame_idx]
        gt_boxes = frame_gt[['bb_left', 'bb_top', 'bb_width', 'bb_height']].values.tolist()
        gt_ids = frame_gt['id'].values.tolist()
        
        dist_raw = compute_iou_matrix(gt_boxes, raw_boxes, max_iou=0.5)
        dist_trk = compute_iou_matrix(gt_boxes, trk_boxes, max_iou=0.5)
        
        raw_ev = acc_raw.update(gt_ids, raw_ids, dist_raw)
        trk_ev = acc_trk.update(gt_ids, trk_ids, dist_trk)
        
        from scipy.optimize import linear_sum_assignment
        def fast_tp(dist):
            if dist.size == 0: return 0
            dist = dist.copy()
            dist[np.isnan(dist)] = 1e5
            r, c = linear_sum_assignment(dist)
            valid = 0
            for i, j in zip(r, c):
                if dist[i, j] < 1e5: valid += 1
            return valid
            
        tp_raw = fast_tp(dist_raw)
        tp_trk = fast_tp(dist_trk)
        
        per_frame_diagnostics.append({
            "frame": int(frame_idx),
            "gt_persons": len(gt_boxes),
            "raw_detections": len(raw_boxes),
            "trk_outputs": len(trk_boxes),
            "raw_tp": int(tp_raw),
            "raw_fp": int(len(raw_boxes) - tp_raw),
            "raw_fn": int(len(gt_boxes) - tp_raw),
            "trk_tp": int(tp_trk),
            "trk_fp": int(len(trk_boxes) - tp_trk),
            "trk_fn": int(len(gt_boxes) - tp_trk)
        })
        
        processed_frames += 1
        peak_rss = max(peak_rss, process.memory_info().rss)
        
    duration = time.time() - start_time
    stop_sampling = True
    sampler_thread.join()
    
    if not cpu_samples: cpu_samples = [process.cpu_percent()]
    avg_cpu = sum(cpu_samples) / len(cpu_samples)
    peak_cpu = max(cpu_samples)
    
    return {
        "acc_raw": acc_raw,
        "acc_trk": acc_trk,
        "per_frame": per_frame_diagnostics,
        "stats": {
            "target_fps": target_fps if target_fps else "Native",
            "sampled_frames": processed_frames,
            "runtime": duration,
            "processing_fps": processed_frames / duration if duration > 0 else 0,
            "peak_rss_mb": peak_rss / (1024 * 1024),
            "avg_cpu": avg_cpu,
            "peak_cpu": peak_cpu
        }
    }

def main():
    adapter = BoundaryTrackingAdapter()
    if adapter.model is not None and __import__('torch').backends.mps.is_available(): adapter.model.to('mps')
    if not adapter.is_model_loaded: sys.exit(1)
    
    # Initialize a completely separate YOLO model for raw predict calls to avoid polluting the adapter.model tracker state
    raw_model = YOLO(adapter._model_path)
    if __import__('torch').backends.mps.is_available(): raw_model.to('mps')
        
    mot17_train_dir = "data/external/mot17/MOT17/train"
    sequences = sorted([os.path.join(mot17_train_dir, d) for d in os.listdir(mot17_train_dir) if os.path.isdir(os.path.join(mot17_train_dir, d)) and d.endswith("-FRCNN")])
    
    target_rates = [None, 5, 2, 1.2, 1]
    results_data = {"rates": {}}
    mh = mm.metrics.create()
    
    metrics_of_interest = ['mota', 'idf1', 'num_false_positives', 'num_misses', 'num_switches', 'num_objects', 'num_predictions']
    
    for rate in target_rates:
        rate_key = "Native" if rate is None else str(rate)
        print(f"Evaluating {rate_key} FPS...")
        accs_raw, accs_trk, names, mode_stats = [], [], [], {}
        pf_all = []
        
        for seq_path in sequences:
            seq_name = os.path.basename(seq_path)
            res = evaluate_sequence(seq_path, adapter, raw_model, target_fps=rate)
            accs_raw.append(res["acc_raw"])
            accs_trk.append(res["acc_trk"])
            names.append(seq_name)
            mode_stats[seq_name] = res["stats"]
            pf_all.extend(res["per_frame"])
            
        def process_summary(accs):
            summary = mh.compute_many(accs, metrics=mm.metrics.motchallenge_metrics + ['num_objects', 'num_predictions'], names=names, generate_overall=True)
            agg = summary.loc["OVERALL", metrics_of_interest].to_dict()
            for k, v in agg.items(): agg[k] = float(v) if not pd.isna(v) else None
            
            gt_objs, preds = agg['num_objects'], agg['num_predictions']
            fp, fn, idsw = agg['num_false_positives'], agg['num_misses'], agg['num_switches']
            tp = gt_objs - fn
            
            agg['tp'] = tp
            agg['det_recall'] = tp / gt_objs if gt_objs > 0 else 0
            agg['det_precision'] = tp / preds if preds > 0 else 0
            agg['fp_rate'] = fp / gt_objs if gt_objs > 0 else 0
            agg['fn_rate'] = fn / gt_objs if gt_objs > 0 else 0
            agg['idsw_per_1k_gt'] = (idsw / gt_objs * 1000) if gt_objs > 0 else 0
            return agg
            
        agg_raw = process_summary(accs_raw)
        agg_trk = process_summary(accs_trk)
        
        total_runtime = sum([s['runtime'] for s in mode_stats.values()])
        total_frames = sum([s['sampled_frames'] for s in mode_stats.values()])
        avg_cpu = sum([s['avg_cpu'] for s in mode_stats.values()]) / len(mode_stats)
        peak_cpu = max([s['peak_cpu'] for s in mode_stats.values()])
        peak_rss = max([s['peak_rss_mb'] for s in mode_stats.values()])
        
        results_data["rates"][rate_key] = {
            "raw": agg_raw,
            "trk": agg_trk,
            "stats": {
                "total_frames": total_frames,
                "total_runtime": total_runtime,
                "processing_fps": total_frames / total_runtime if total_runtime > 0 else 0,
                "avg_cpu": avg_cpu,
                "peak_cpu": peak_cpu,
                "peak_rss_mb": peak_rss
            },
            "per_frame_diagnostics": pf_all
        }
        
    os.makedirs("reports/phase8", exist_ok=True)
    with open("reports/phase8/8H_isolation_results.json", "w") as f: json.dump(results_data, f, indent=2)
        
    with open("reports/phase8/8H_isolation_report.md", "w") as f:
        f.write("# Phase 8H: Detection vs Association Isolation\n\n")
        f.write("## 1. Tracker Suppression Gap (Recall & Precision)\n")
        f.write("| Sampling | GT Objects | Raw Recall | Tracked Recall | **Suppression Gap** | Raw Precision | Tracked Precision | **Precision Gap** |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        
        for rate in ["Native", "5", "2", "1.2", "1"]:
            r = results_data["rates"][rate]["raw"]
            t = results_data["rates"][rate]["trk"]
            gap = r['det_recall'] - t['det_recall']
            pgap = r['det_precision'] - t['det_precision']
            f.write(f"| {rate} | {int(r['num_objects'])} | {r['det_recall']:.4f} | {t['det_recall']:.4f} | **{gap:+.4f}** | {r['det_precision']:.4f} | {t['det_precision']:.4f} | **{pgap:+.4f}** |\n")
            
        f.write("\n## 2. Tracking Continuity Degradation\n")
        f.write("| Sampling | MOTA | IDF1 | IDSW / 1K GT |\n")
        f.write("|---|---|---|---|\n")
        for rate in ["Native", "5", "2", "1.2", "1"]:
            t = results_data["rates"][rate]["trk"]
            f.write(f"| {rate} | {t['mota']:.4f} | {t['idf1']:.4f} | {t['idsw_per_1k_gt']:.2f} |\n")
            
        f.write("\n## 3. Resource Utilization\n")
        f.write("| Sampling | Evaluated Frames | Processing FPS | Wall-Clock Runtime (s) | Average CPU (%) | Peak CPU (%) | Peak RSS (MB) |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for rate in ["Native", "5", "2", "1.2", "1"]:
            s = results_data["rates"][rate]["stats"]
            f.write(f"| {rate} | {s['total_frames']} | {s['processing_fps']:.1f} | {s['total_runtime']:.1f} | {s['avg_cpu']:.1f}% | {s['peak_cpu']:.1f}% | {s['peak_rss_mb']:.1f} |\n")

if __name__ == '__main__':
    main()
