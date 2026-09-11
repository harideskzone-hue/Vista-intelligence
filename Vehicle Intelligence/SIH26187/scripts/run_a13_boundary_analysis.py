import os
import cv2
import json
import time
import sys
import re
import numpy as np
from collections import defaultdict
from pathlib import Path

from ultralytics import YOLO
from fast_plate_ocr import LicensePlateRecognizer

from run_a11 import get_video_metadata

def laplacian_variance(image):
    if image.size == 0: return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def is_valid_indian_format(text: str) -> bool:
    if not text: return False
    cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
    if len(cleaned) >= 6 and re.match(r'^[A-Z]{2}\d{1,2}[A-Z]{0,3}\d{3,4}$', cleaned):
        return True
    return False

def process_boundary_analysis(video_path: Path):
    video_meta = get_video_metadata(str(video_path))
    
    vehicle_model = YOLO("yolo11n.pt")
    plate_model = YOLO("anpr/models/best.pt")
    ocr_model = LicensePlateRecognizer(
        onnx_model_path="data/fast_plate_ocr/models/fine_tuned/2026-09-09_11-29-31/best.onnx",
        plate_config_path="data/fast_plate_ocr/models/cct_xs_v2_global_plate_config.yaml"
    )
    
    cap = cv2.VideoCapture(str(video_path))
    
    records = []
    
    # We will use stride=6 to keep it fast but conf=0.10 and min_area=2000 to capture weak plates.
    stride = 6
    min_area = 2000
    
    frame_idx = 0
    track_history = {}
    
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
                    track_history[track_id] = {"frames_seen": 0}
                    
                track_history[track_id]["frames_seen"] += 1
                
                if (track_history[track_id]["frames_seen"] - 1) % stride != 0:
                    continue
                    
                if veh_area < min_area:
                    continue
                    
                veh_crop = frame[max(0, y1):min(video_meta.get("height", y2), y2), 
                                 max(0, x1):min(video_meta.get("width", x2), x2)]
                
                if veh_crop.size == 0: continue
                
                # We use conf=0.10 to ensure we capture the boundary of "impossible" vs "uncertain"
                lp_results = plate_model.predict(veh_crop, conf=0.10, verbose=False)
                
                if lp_results and len(lp_results[0].boxes) > 0:
                    # Collect ALL plate detections to map the distribution
                    for idx in range(len(lp_results[0].boxes)):
                        lp_box = lp_results[0].boxes.xyxy[idx].cpu().numpy()
                        lp_conf = float(lp_results[0].boxes.conf[idx].cpu().numpy())
                        
                        lx1, ly1, lx2, ly2 = map(int, lp_box)
                        w, h = lx2 - lx1, ly2 - ly1
                        plate_area = w * h
                        
                        plate_crop = veh_crop[max(0, ly1):min(veh_crop.shape[0], ly2), max(0, lx1):min(veh_crop.shape[1], lx2)]
                        
                        if plate_crop.size > 0:
                            sharpness = laplacian_variance(plate_crop)
                            rgb_crop = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB)
                            
                            ocr_text = ""
                            ocr_conf = 0.0
                            try:
                                preds = ocr_model.run(rgb_crop)
                                if preds:
                                    ocr_text = getattr(preds[0], 'plate', "").replace("_", "")
                                    probs = getattr(preds[0], 'char_probs', [])
                                    ocr_conf = float(np.mean(probs)) if probs else 1.0
                            except:
                                pass
                                
                            is_valid = is_valid_indian_format(ocr_text)
                            
                            records.append({
                                "video": video_path.stem,
                                "frame": frame_idx,
                                "track_id": track_id,
                                "width": w,
                                "height": h,
                                "area": plate_area,
                                "sharpness": sharpness,
                                "det_conf": lp_conf,
                                "ocr_text": ocr_text,
                                "ocr_conf": ocr_conf,
                                "is_valid": is_valid
                            })

        if frame_idx % 50 == 0:
            sys.stdout.write(f"Processed {frame_idx} frames...\r")
            sys.stdout.flush()
            
    sys.stdout.write("\n")
    cap.release()
    return records

def main():
    videos = [
        "input/a11_benchmark/pexels-george-morina-6719160 (2160p).mp4",
        "input/a11_benchmark/night_driving.mp4",
        "input/a11_benchmark/highway_fast.mp4",
        "input/a11_benchmark/far_vehicles.mp4"
    ]
    
    all_records = []
    for v in videos:
        p = Path(v)
        if not p.exists(): continue
        print(f"Processing {p.name}...")
        records = process_boundary_analysis(p)
        all_records.extend(records)
        
    out_path = Path("output/a13_boundary_records.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_records, f, indent=2)
        
    print(f"A.13 Boundary Analysis complete. Saved {len(all_records)} records to {out_path}")

if __name__ == "__main__":
    main()
