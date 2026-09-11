import os
import cv2
import json
import time
import sys
import numpy as np
from pathlib import Path

from ultralytics import YOLO
from fast_plate_ocr import LicensePlateRecognizer

from run_a11 import get_video_metadata, GenericTemporalVoter

def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

def laplacian_variance(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

# Modes
MODES = {
    "A": {"name": "A10_frozen", "stride": 6, "min_area": 10000, "conf": 0.25, "clahe": False, "adaptive_stride": False},
    "B": {"name": "Dense_stride1", "stride": 1, "min_area": 10000, "conf": 0.25, "clahe": False, "adaptive_stride": False},
    "C": {"name": "Sharp_frame_6", "stride": 6, "min_area": 10000, "conf": 0.25, "clahe": False, "adaptive_stride": True},
    "D": {"name": "Conf_0.10", "stride": 6, "min_area": 10000, "conf": 0.10, "clahe": False, "adaptive_stride": False},
    "E": {"name": "CLAHE", "stride": 6, "min_area": 10000, "conf": 0.25, "clahe": True, "adaptive_stride": False},
    "F": {"name": "Size_5000", "stride": 6, "min_area": 5000, "conf": 0.25, "clahe": False, "adaptive_stride": False}
}

def process_video_ablation(video_path: Path):
    video_meta = get_video_metadata(str(video_path))
    
    vehicle_model = YOLO("yolo11n.pt")
    plate_model = YOLO("anpr/models/best.pt")
    ocr_model = LicensePlateRecognizer(
        onnx_model_path="data/fast_plate_ocr/models/fine_tuned/2026-09-09_11-29-31/best.onnx",
        plate_config_path="data/fast_plate_ocr/models/cct_xs_v2_global_plate_config.yaml"
    )
    
    cap = cv2.VideoCapture(str(video_path))
    
    mode_stats = {m: {
        "calls": 0, "plates": 0, "widths": [], "heights": [], "areas": [], "confs": [],
        "lpr_calls": 0, "voters": {}, "track_history": {}
    } for m in MODES}
    
    track_history = {} # Global track history for buffering
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1
        
        results = vehicle_model.track(frame, classes=[2, 3, 5, 7], persist=True, tracker="bytetrack.yaml", verbose=False)
        
        if results and results[0].boxes and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            
            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box)
                veh_area = (x2 - x1) * (y2 - y1)
                
                if track_id not in track_history:
                    track_history[track_id] = {"frames_seen": 0, "buffer": []}
                    
                track_history[track_id]["frames_seen"] += 1
                fs = track_history[track_id]["frames_seen"]
                
                veh_crop = frame[max(0, y1):min(video_meta.get("height", y2), y2), 
                                 max(0, x1):min(video_meta.get("width", x2), x2)]
                
                if veh_crop.size > 0:
                    var = laplacian_variance(veh_crop)
                    track_history[track_id]["buffer"].append({
                        "crop": veh_crop, "area": veh_area, "var": var, "frame": frame_idx, "x1": x1, "y1": y1
                    })
                
                # Evaluate Modes
                for m_id, m_cfg in MODES.items():
                    if fs % m_cfg["stride"] != 0:
                        continue # Not the right stride frame
                        
                    buf = track_history[track_id]["buffer"]
                    if not buf: continue
                    
                    if m_cfg["adaptive_stride"]:
                        # Select best var in last 'stride' frames
                        window = buf[-m_cfg["stride"]:]
                        best_entry = max(window, key=lambda x: x["var"])
                    else:
                        best_entry = buf[-1] # take latest
                        
                    if best_entry["area"] < m_cfg["min_area"]:
                        continue # size gate
                        
                    crop = best_entry["crop"]
                    if m_cfg["clahe"]:
                        crop = apply_clahe(crop)
                        
                    # Plate detection
                    mode_stats[m_id]["calls"] += 1
                    lp_results = plate_model.predict(crop, conf=m_cfg["conf"], verbose=False)
                    
                    if lp_results and len(lp_results[0].boxes) > 0:
                        best_lp_idx = np.argmax(lp_results[0].boxes.conf.cpu().numpy())
                        lp_box = lp_results[0].boxes.xyxy[best_lp_idx].cpu().numpy()
                        lp_conf = float(lp_results[0].boxes.conf[best_lp_idx].cpu().numpy())
                        
                        mode_stats[m_id]["plates"] += 1
                        mode_stats[m_id]["confs"].append(lp_conf)
                        
                        lx1, ly1, lx2, ly2 = map(int, lp_box)
                        w, h = lx2 - lx1, ly2 - ly1
                        mode_stats[m_id]["widths"].append(w)
                        mode_stats[m_id]["heights"].append(h)
                        mode_stats[m_id]["areas"].append(w * h)
                        
                        plate_crop = crop[max(0, ly1):min(crop.shape[0], ly2), max(0, lx1):min(crop.shape[1], lx2)]
                        
                        # OCR & Voter
                        if plate_crop.size > 0:
                            if track_id not in mode_stats[m_id]["voters"]:
                                mode_stats[m_id]["voters"][track_id] = GenericTemporalVoter(min_evidence=0.4)
                                
                            rgb_crop = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB)
                            try:
                                mode_stats[m_id]["lpr_calls"] += 1
                                preds = ocr_model.run(rgb_crop)
                                if preds:
                                    ocr_text = getattr(preds[0], 'plate', "").replace("_", "")
                                    probs = getattr(preds[0], 'char_probs', [])
                                    if probs is None: probs = []
                                    ocr_conf = float(np.mean(probs)) if len(probs) > 0 else 1.0
                                    
                                    quality_score = min(1.0, (h / 40.0)) * lp_conf
                                    if ocr_text:
                                        mode_stats[m_id]["voters"][track_id].add(
                                            ocr_text, ocr_conf, lp_conf, quality_score, best_entry["frame"], probs
                                        )
                            except:
                                pass

                # Clean buffer to save memory (keep max 10 frames)
                if len(track_history[track_id]["buffer"]) > 10:
                    track_history[track_id]["buffer"] = track_history[track_id]["buffer"][-10:]

        if frame_idx % 50 == 0:
            sys.stdout.write(f"Processed {frame_idx} frames...\r")
            sys.stdout.flush()
            
    sys.stdout.write("\n")
    cap.release()
    
    # Finalize Voter Decisions
    out_stats = {}
    for m_id, m_cfg in MODES.items():
        voters = mode_stats[m_id]["voters"]
        confirmed, probable, unknown = 0, 0, 0
        for tid, voter in voters.items():
            decision = voter.decide()
            if decision["status"] == "CONFIRMED": confirmed += 1
            elif decision["status"] == "PROBABLE": probable += 1
            elif decision["observations"] > 0: unknown += 1
            
        def safe_percentile(arr, p):
            return float(np.percentile(arr, p)) if arr else 0.0
            
        out_stats[m_cfg["name"]] = {
            "calls": mode_stats[m_id]["calls"],
            "plates": mode_stats[m_id]["plates"],
            "yield_pct": round(mode_stats[m_id]["plates"] / max(1, mode_stats[m_id]["calls"]) * 100, 2),
            "plate_widths": {
                "min": safe_percentile(mode_stats[m_id]["widths"], 0),
                "median": safe_percentile(mode_stats[m_id]["widths"], 50),
                "max": safe_percentile(mode_stats[m_id]["widths"], 100)
            },
            "plate_heights": {
                "min": safe_percentile(mode_stats[m_id]["heights"], 0),
                "median": safe_percentile(mode_stats[m_id]["heights"], 50),
                "max": safe_percentile(mode_stats[m_id]["heights"], 100)
            },
            "plate_areas": {
                "min": safe_percentile(mode_stats[m_id]["areas"], 0),
                "median": safe_percentile(mode_stats[m_id]["areas"], 50),
                "max": safe_percentile(mode_stats[m_id]["areas"], 100)
            },
            "conf_median": safe_percentile(mode_stats[m_id]["confs"], 50),
            "lpr_calls": mode_stats[m_id]["lpr_calls"],
            "recognition": {
                "confirmed": confirmed,
                "probable": probable,
                "unknown": unknown
            }
        }
    
    return out_stats

def main():
    videos = [
        "input/a11_benchmark/pexels-george-morina-6719160 (2160p).mp4",
        "input/a11_benchmark/night_driving.mp4",
        "input/a11_benchmark/highway_fast.mp4",
        "input/a11_benchmark/far_vehicles.mp4"
    ]
    
    all_results = {}
    for v in videos:
        p = Path(v)
        if not p.exists(): continue
        print(f"Processing {p.name}...")
        res = process_video_ablation(p)
        all_results[p.stem] = res
        
    with open("output/a12_ablation_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
        
    print("A.12 Ablation complete. Saved to output/a12_ablation_results.json")

if __name__ == "__main__":
    main()
