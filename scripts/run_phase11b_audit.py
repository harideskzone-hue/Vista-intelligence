import os
import sys
import glob
import time
import json
import hashlib
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import motmetrics as mm
from ultralytics.trackers.byte_tracker import BYTETracker

class SimpleNamespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    def __getattr__(self, item):
        return False

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
    return matches

def init_byte_tracker():
    args = SimpleNamespace(
        track_high_thresh=0.5,
        track_low_thresh=0.1,
        new_track_thresh=0.6,
        track_buffer=30,
        match_thresh=0.8,
        gmc_method="sparseOptFlow" # Default bytetrack yaml config
    )
    return BYTETracker(args=args)

def run_experiment(exp_id, imgsz, conf, target_fps, batch_size, model_path, dataset_dir, expected_sequences, use_custom_tracker=False):
    print(f"\n--- Running Experiment: {exp_id} ---")
    
    model = YOLO(model_path)
    acc = mm.MOTAccumulator(auto_id=True)
    
    det_tp = 0
    det_fp = 0
    det_fn = 0
    total_gt_observations = 0
    
    for seq in expected_sequences:
        seq_path = os.path.join(dataset_dir, seq)
        if not os.path.exists(seq_path):
            continue
            
        import configparser
        seqinfo = configparser.ConfigParser()
        seqinfo.read(os.path.join(seq_path, 'seqinfo.ini'))
        seq_fps = int(seqinfo['Sequence']['frameRate'])
        seq_length = int(seqinfo['Sequence']['seqLength'])
        
        stride = seq_fps if target_fps == "baseline" else max(1, int(seq_fps / target_fps))
        sampled_indices = list(range(1, seq_length + 1, stride))
        img_dir = os.path.join(seq_path, 'img1')
        
        gt_file = os.path.join(seq_path, 'gt', 'gt.txt')
        gt_df = pd.read_csv(gt_file, header=None)
        gt_df.columns = ['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'class', 'visibility']
        gt_df = gt_df[gt_df['class'] == 1]
        
        # Fresh model/tracker for sequence
        if not use_custom_tracker:
            model = YOLO(model_path)
        else:
            tracker = init_byte_tracker()
            
        seq_offset = int(hashlib.md5(seq.encode()).hexdigest(), 16) % 100000000
        
        # Batch iterator over sampled frames
        for i in range(0, len(sampled_indices), batch_size):
            batch_idxs = sampled_indices[i:i+batch_size]
            batch_frames = []
            for idx in batch_idxs:
                img_path = os.path.join(img_dir, f"{idx:06d}.jpg")
                batch_frames.append(cv2.imread(img_path))
                
            if len(batch_frames) == 0:
                continue
                
            # Detection eval
            det_results = model.predict(batch_frames, imgsz=imgsz, conf=conf, classes=[0], device="mps", verbose=False)
            
            # Tracking eval
            if use_custom_tracker:
                # Custom decoupled batched-YOLO to sequential ByteTrack
                trk_pred_list = []
                for j in range(len(det_results)):
                    det = det_results[j].boxes.cpu().numpy()
                    tr = tracker.update(det, img=det_results[j].orig_img)
                    trk_pred_list.append(tr)
            else:
                # Standard ultralytics track (which implicitly couples or independent parallels depending on batch)
                if len(batch_frames) == 1:
                    tr = model.track(batch_frames[0], persist=True, tracker="bytetrack.yaml", imgsz=imgsz, conf=conf, classes=[0], device="mps", verbose=False)[0]
                    if tr.boxes.id is not None:
                        # [x1, y1, x2, y2, track_id, conf, cls] format manually built
                        res = []
                        for box, tid in zip(tr.boxes.xyxy.cpu().numpy(), tr.boxes.id.cpu().numpy()):
                            res.append([box[0], box[1], box[2], box[3], tid, 1.0, 0, 0]) # dummy trailing args to match _format_output
                        trk_pred_list = [np.array(res)]
                    else:
                        trk_pred_list = [np.array([])]
                else:
                    tr_batch = model.track(batch_frames, persist=True, tracker="bytetrack.yaml", imgsz=imgsz, conf=conf, classes=[0], device="mps", verbose=False)
                    trk_pred_list = []
                    for tr in tr_batch:
                        if tr.boxes.id is not None:
                            res = []
                            for box, tid in zip(tr.boxes.xyxy.cpu().numpy(), tr.boxes.id.cpu().numpy()):
                                res.append([box[0], box[1], box[2], box[3], tid, 1.0, 0, 0])
                            trk_pred_list.append(np.array(res))
                        else:
                            trk_pred_list.append(np.array([]))

            # Metric updates per frame in the batch
            for j in range(len(batch_idxs)):
                frame_idx = batch_idxs[j]
                frame_gt = gt_df[gt_df['frame'] == frame_idx]
                gt_boxes = frame_gt[['bb_left', 'bb_top', 'bb_width', 'bb_height']].values.tolist()
                gt_ids = frame_gt['id'].values.tolist()
                total_gt_observations += len(gt_boxes)
                
                det_pred_boxes = []
                for box in det_results[j].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    det_pred_boxes.append([x1, y1, x2 - x1, y2 - y1])
                dist_matrix_det = compute_iou_matrix(gt_boxes, det_pred_boxes, max_iou=0.5)
                matches_det = greedy_match(dist_matrix_det)
                det_tp += len(matches_det)
                det_fp += len(det_pred_boxes) - len(matches_det)
                det_fn += len(gt_boxes) - len(matches_det)
                
                trk_pred_boxes = []
                trk_pred_ids = []
                tr_arr = trk_pred_list[j]
                if len(tr_arr) > 0:
                    for obj in tr_arr:
                        x1, y1, x2, y2, tid = obj[0], obj[1], obj[2], obj[3], obj[4]
                        trk_pred_boxes.append([x1, y1, x2 - x1, y2 - y1])
                        trk_pred_ids.append(int(tid))
                        
                shifted_gt_ids = [seq_offset + int(gid) for gid in gt_ids]
                shifted_pred_ids = [seq_offset + int(pid) for pid in trk_pred_ids]
                
                dist_matrix_trk = compute_iou_matrix(gt_boxes, trk_pred_boxes, max_iou=0.5)
                acc.update(shifted_gt_ids, shifted_pred_ids, dist_matrix_trk)

    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=['num_frames', 'mota', 'idf1', 'mostly_tracked', 'mostly_lost', 
                                       'num_false_positives', 'num_misses', 'num_switches', 'num_fragmentations', 
                                       'num_predictions', 'num_objects', 'num_matches'], name='aggregate')
    
    tp = summary.loc['aggregate', 'num_matches']
    fp = summary.loc['aggregate', 'num_false_positives']
    fn = summary.loc['aggregate', 'num_misses']
    trk_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    trk_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    det_precision = det_tp / (det_tp + det_fp) if (det_tp + det_fp) > 0 else 0
    det_recall = det_tp / (det_tp + det_fn) if (det_tp + det_fn) > 0 else 0
    
    num_switches = summary.loc['aggregate', 'num_switches']
    idsw_per_1k_gt = (num_switches / total_gt_observations) * 1000 if total_gt_observations > 0 else 0
    
    manifest = {
        "experiment": exp_id,
        "det_TP": int(det_tp),
        "det_FP": int(det_fp),
        "det_FN": int(det_fn),
        "det_Recall": float(det_recall),
        "trk_TP": int(tp),
        "trk_FP": int(fp),
        "trk_FN": int(fn),
        "trk_Recall": float(trk_recall),
        "mota": float(summary.loc['aggregate', 'mota']),
        "idf1": float(summary.loc['aggregate', 'idf1']),
        "idsw": int(num_switches),
        "idsw_per_1k_gt_obs": float(idsw_per_1k_gt)
    }
    
    print(f"[{exp_id}] DET Rec: {det_recall:.4f} | TRK Rec: {trk_recall:.4f} | MOTA: {manifest['mota']:.4f} | IDF1: {manifest['idf1']:.4f} | IDSW: {num_switches} ({idsw_per_1k_gt:.2f}/1k)")
    
    with open(f"/Users/hariharans/Documents/SIH26187/reports/phase11/{exp_id}.json", "w") as f:
        json.dump(manifest, f, indent=2)

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
    
    # 1. Baseline Audits
    run_experiment("11B_0_audit", 640, 0.40, "baseline", 1, model_path, dataset_dir, expected_sequences)
    run_experiment("11B_1_audit_conf20", 640, 0.20, "baseline", 1, model_path, dataset_dir, expected_sequences)
    run_experiment("11B_4_audit_fps_0_5", 640, 0.40, 0.5, 1, model_path, dataset_dir, expected_sequences)
    run_experiment("11B_4_audit_fps_2_0", 640, 0.40, 2.0, 1, model_path, dataset_dir, expected_sequences)
    
    # 2. 11B-5 Temporal Batching Integrity
    # Batched detection + sequential tracker (use_custom_tracker=True)
    run_experiment("11B_5_batch_1", 640, 0.40, "baseline", 1, model_path, dataset_dir, expected_sequences, use_custom_tracker=True)
    run_experiment("11B_5_batch_2", 640, 0.40, "baseline", 2, model_path, dataset_dir, expected_sequences, use_custom_tracker=True)
    run_experiment("11B_5_batch_4", 640, 0.40, "baseline", 4, model_path, dataset_dir, expected_sequences, use_custom_tracker=True)

if __name__ == "__main__":
    main()
