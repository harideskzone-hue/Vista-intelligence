import os
import cv2
import json
import re
import time
import sys
import argparse
import numpy as np
import subprocess
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass

from ultralytics import YOLO
from fast_plate_ocr import LicensePlateRecognizer

# ------------------------------------------------------------------------------
# Frozen Configuration (A.10)
# ------------------------------------------------------------------------------
FROZEN_CONFIG = {
    "plate_stride": 6,
    "min_vehicle_area": 10000,
    "verification_stride": 60
}

# ------------------------------------------------------------------------------
# Generic Temporal Voter (Fuzzy Matching + Indian Regex)
# ------------------------------------------------------------------------------
def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def normalized_levenshtein(s1: str, s2: str) -> float:
    if not s1 and not s2: return 1.0
    dist = levenshtein_distance(s1, s2)
    return 1.0 - (dist / max(len(s1), len(s2)))

@dataclass
class Observation:
    text: str
    ocr_confidence: float
    detector_confidence: float
    quality_score: float
    frame_index: int
    char_probs: list
    
    @property
    def evidence(self) -> float:
        return self.ocr_confidence * self.detector_confidence * self.quality_score

class GenericTemporalVoter:
    def __init__(self, min_observations: int = 3, min_evidence: float = 0.1):
        self.observations = []
        self.min_observations = min_observations
        self.min_evidence = min_evidence

    def add(self, text: str, ocr_conf: float, det_conf: float, qual_score: float, frame: int, char_probs: list):
        if not text: return
        self.observations.append(Observation(text, ocr_conf, det_conf, qual_score, frame, char_probs))

    def _cluster_observations(self):
        clusters = []
        for obs in self.observations:
            added = False
            for cluster in clusters:
                centroid = cluster[0]
                sim = normalized_levenshtein(obs.text, centroid.text)
                len_diff = abs(len(obs.text) - len(centroid.text))
                if sim >= 0.6 and len_diff <= 2:
                    cluster.append(obs)
                    cluster.sort(key=lambda o: o.evidence, reverse=True)
                    added = True
                    break
            if not added:
                clusters.append([obs])
        return sorted(clusters, key=lambda c: sum(o.evidence for o in c), reverse=True)

    def _align_and_vote(self, cluster):
        if not cluster:
            return "", 0.0
        lengths = defaultdict(float)
        for obs in cluster:
            lengths[len(obs.text)] += obs.evidence
        target_len = max(lengths.keys(), key=lambda k: lengths[k])
        aligned = [o for o in cluster if len(o.text) == target_len]
        if not aligned:
            return cluster[0].text, sum(o.evidence for o in cluster)
        
        total_cluster_evidence = sum(o.evidence for o in aligned)
        final_chars = []
        for i in range(target_len):
            char_votes = defaultdict(float)
            for obs in aligned:
                char_votes[obs.text[i]] += obs.evidence
            best_char = max(char_votes.keys(), key=lambda k: char_votes[k])
            final_chars.append(best_char)
            
        return "".join(final_chars), total_cluster_evidence
        
    def decide(self) -> dict:
        if not self.observations:
            return {"text": None, "score": 0.0, "status": "UNKNOWN", "observations": 0}
            
        clusters = self._cluster_observations()
        dominant_cluster = clusters[0]
        
        if len(clusters) > 1:
            dom_ev = sum(o.evidence for o in dominant_cluster)
            sec_ev = sum(o.evidence for o in clusters[1])
            if dom_ev > 0 and (sec_ev / dom_ev) >= 0.7:
                return {"text": None, "score": 0.0, "status": "UNKNOWN", "observations": len(dominant_cluster)}
                
        reconstructed_text, cluster_evidence = self._align_and_vote(dominant_cluster)
        avg_evidence = cluster_evidence / len(dominant_cluster)
        obs_count = len(dominant_cluster)
        
        # Indian Plate format regex
        is_valid_format = False
        if reconstructed_text:
            cleaned = re.sub(r'[^A-Z0-9]', '', reconstructed_text.upper())
            if len(cleaned) >= 6 and re.match(r'^[A-Z]{2}\d{1,2}[A-Z]{0,3}\d{3,4}$', cleaned):
                is_valid_format = True
        
        if obs_count >= self.min_observations and avg_evidence >= self.min_evidence and is_valid_format:
            status = "CONFIRMED"
        elif obs_count >= 2 and avg_evidence >= 0.05:
            status = "PROBABLE"
        else:
            status = "UNKNOWN"
            
        return {"text": reconstructed_text if status != "UNKNOWN" else None, 
                "score": avg_evidence, "status": status, "observations": obs_count}

# ------------------------------------------------------------------------------
# Metadata extraction via ffprobe
# ------------------------------------------------------------------------------
def get_video_metadata(video_path: str) -> dict:
    cmd = [
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration:stream=width,height,r_frame_rate,nb_frames,codec_name",
        "-of", "json", video_path
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        data = json.loads(result.stdout)
        
        video_stream = next((s for s in data.get('streams', []) if 'width' in s), None)
        format_info = data.get('format', {})
        
        if not video_stream:
            raise ValueError("No video stream found")
            
        fps_str = video_stream.get('r_frame_rate', '0/0')
        if '/' in fps_str:
            num, den = map(float, fps_str.split('/'))
            fps = num / den if den != 0 else 0.0
        else:
            fps = float(fps_str)
            
        return {
            "filename": os.path.basename(video_path),
            "width": int(video_stream.get('width', 0)),
            "height": int(video_stream.get('height', 0)),
            "fps": round(fps, 2),
            "frames": int(video_stream.get('nb_frames', 0)),
            "duration_sec": float(format_info.get('duration', 0.0))
        }
    except Exception as e:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {}
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frames / fps if fps > 0 else 0.0
        meta = {
            "filename": os.path.basename(video_path),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": round(fps, 2),
            "frames": frames,
            "duration_sec": round(duration, 3)
        }
        cap.release()
        return meta

# ------------------------------------------------------------------------------
# Models (Global Initialization to avoid reloading per video)
# ------------------------------------------------------------------------------
vehicle_model = None
plate_model = None
ocr_model = None

def init_models():
    global vehicle_model, plate_model, ocr_model
    if vehicle_model is None:
        vehicle_model = YOLO("yolo11n.pt")
        plate_model = YOLO("anpr/models/best.pt")
        ocr_model = LicensePlateRecognizer(
            onnx_model_path="data/fast_plate_ocr/models/fine_tuned/2026-09-09_11-29-31/best.onnx",
            plate_config_path="data/fast_plate_ocr/models/cct_xs_v2_global_plate_config.yaml"
        )

# ------------------------------------------------------------------------------
# Video Processor
# ------------------------------------------------------------------------------
def process_video(video_path: Path):
    video_name = video_path.stem
    output_dir = Path(f"output/a11/{video_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    
    # Automatic metadata extraction
    video_meta = get_video_metadata(str(video_path))
    
    init_models()
    
    cap = cv2.VideoCapture(str(video_path))
    out_vid = cv2.VideoWriter(
        str(output_dir / "annotated_video.mp4"),
        cv2.VideoWriter_fourcc(*'mp4v'),
        int(video_meta.get("fps", 30)),
        (video_meta.get("width", 1920), video_meta.get("height", 1080))
    )
    
    voters = {}
    track_history = {}
    detections_log = []
    
    frame_idx = 0
    t0_all = time.time()
    
    c_total_vehicle_bounding_boxes = 0
    c_skipped_by_stride = 0
    c_skipped_by_size = 0
    c_plate_yolo_calls_made = 0
    c_successful_plate_detections = 0
    c_lpr_calls = 0
    
    stride = FROZEN_CONFIG["plate_stride"]
    min_area = FROZEN_CONFIG["min_vehicle_area"]
    verification_stride = FROZEN_CONFIG["verification_stride"]
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_idx += 1
        annotated_frame = frame.copy()
        
        # Vehicle Tracking
        results = vehicle_model.track(frame, classes=[2, 3, 5, 7], persist=True, tracker="bytetrack.yaml", verbose=False)
        
        if results and results[0].boxes and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            confidences = results[0].boxes.conf.cpu().numpy()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            
            for box, track_id, conf, cls_id in zip(boxes, track_ids, confidences, class_ids):
                x1, y1, x2, y2 = map(int, box)
                cls_name = vehicle_model.names[cls_id]
                
                if track_id not in track_history:
                    track_history[track_id] = {
                        "first_seen": frame_idx, 
                        "class": cls_name, 
                        "frames_seen": 0,
                        "status": "UNKNOWN",
                        "first_confirmed_frame": None,
                        "stable_confirmed_frame": None
                    }
                track_history[track_id]["last_seen"] = frame_idx
                track_history[track_id]["frames_seen"] += 1
                
                c_total_vehicle_bounding_boxes += 1
                
                current_status = track_history[track_id]["status"]
                effective_stride = verification_stride if current_status == "CONFIRMED" else stride
                
                if (track_history[track_id]["frames_seen"] - 1) % effective_stride != 0:
                    c_skipped_by_stride += 1
                    continue
                
                if track_id not in voters:
                    voters[track_id] = GenericTemporalVoter(min_evidence=0.4)
                
                veh_area = (x2 - x1) * (y2 - y1)
                if veh_area < min_area:
                    c_skipped_by_size += 1
                    continue
                    
                veh_crop = frame[max(0, y1):min(video_meta.get("height", y2), y2), 
                                 max(0, x1):min(video_meta.get("width", x2), x2)]
                if veh_crop.size == 0: continue
                
                detections_log.append({
                    "frame": frame_idx, "track_id": track_id, "type": "vehicle",
                    "bbox": [x1, y1, x2, y2], "confidence": float(conf), "class": cls_name
                })
                
                # Plate Detection
                c_plate_yolo_calls_made += 1
                lp_results = plate_model.predict(veh_crop, verbose=False)
                
                if lp_results and len(lp_results[0].boxes) > 0:
                    best_lp_idx = np.argmax(lp_results[0].boxes.conf.cpu().numpy())
                    lp_box = lp_results[0].boxes.xyxy[best_lp_idx].cpu().numpy()
                    lp_conf = float(lp_results[0].boxes.conf[best_lp_idx].cpu().numpy())
                    
                    lx1, ly1, lx2, ly2 = map(int, lp_box)
                    gx1, gy1, gx2, gy2 = x1 + lx1, y1 + ly1, x1 + lx2, y1 + ly2
                    
                    plate_crop = veh_crop[max(0, ly1):min(veh_crop.shape[0], ly2), max(0, lx1):min(veh_crop.shape[1], lx2)]
                    if plate_crop.size > 0:
                        c_successful_plate_detections += 1
                        
                        try:
                            rgb_crop = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB)
                            c_lpr_calls += 1
                            preds = ocr_model.run(rgb_crop)
                            
                            if preds:
                                ocr_text = getattr(preds[0], 'plate', "").replace("_", "")
                                probs = getattr(preds[0], 'char_probs', [])
                                if probs is None: probs = []
                                ocr_conf = float(np.mean(probs)) if len(probs) > 0 else 1.0
                                
                                lp_height = gy2 - gy1
                                quality_score = min(1.0, (lp_height / 40.0)) * lp_conf
                                
                                if ocr_text:
                                    voters[track_id].add(
                                        ocr_text, ocr_conf, lp_conf, quality_score, frame_idx, probs
                                    )
                                    
                                    new_decision = voters[track_id].decide()
                                    if new_decision:
                                        new_status = new_decision["status"]
                                        if new_status == "CONFIRMED":
                                            if track_history[track_id]["first_confirmed_frame"] is None:
                                                track_history[track_id]["first_confirmed_frame"] = frame_idx
                                            elif frame_idx > track_history[track_id]["first_confirmed_frame"]:
                                                track_history[track_id]["stable_confirmed_frame"] = frame_idx
                                        else:
                                            track_history[track_id]["stable_confirmed_frame"] = None
                                            
                                        track_history[track_id]["status"] = new_status
                                        
                                    cv2.rectangle(annotated_frame, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)
                                    cv2.putText(annotated_frame, f"{ocr_text} ({ocr_conf:.2f})", (gx1, gy1 - 5),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        except Exception as e:
                            pass
                            
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                current_decision = voters[track_id].decide() if track_id in voters else {"text": None}
                disp_text = current_decision['text'] if current_decision['text'] else "???"
                label = f"ID:{track_id} {cls_name} [{disp_text}]"
                
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                ty = max(20, y1 - 10)
                cv2.rectangle(annotated_frame, (x1, ty - th - 5), (x1 + tw, ty + 5), (0, 0, 0), -1)
                cv2.putText(annotated_frame, label, (x1, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                            
        out_vid.write(annotated_frame)
        if frame_idx % 50 == 0:
            sys.stdout.write(f"Processed {frame_idx} frames...\r")
            sys.stdout.flush()
            
    sys.stdout.write("\n")
    cap.release()
    out_vid.release()
    t1_all = time.time()
    total_processing_time = round(t1_all - t0_all, 2)
    effective_fps = round(frame_idx / max(total_processing_time, 1), 2)
    
    # Compile Results
    tracks_out = []
    stats = {"confirmed": 0, "probable": 0, "unknown": 0}
    
    for t_id, hist in track_history.items():
        voter = voters.get(t_id)
        decision = voter.decide() if voter else {"text": None, "score": 0.0, "status": "UNKNOWN", "observations": 0}
        
        status = decision['status']
        if status == 'CONFIRMED': stats['confirmed'] += 1
        elif status == 'PROBABLE': stats['probable'] += 1
        else: stats['unknown'] += 1
        
        tracks_out.append({
            "track_id": t_id,
            "plate_decision": status,
            "plate": decision['text'],
            "confidence": decision['score'],
            "observations": decision['observations'],
            "first_confirmed_frame": hist.get("first_confirmed_frame"),
            "stable_confirmed_frame": hist.get("stable_confirmed_frame")
        })
        
    report = {
        "video": video_meta,
        "processing": {
            "frames_read": int(video_meta.get('frames', frame_idx)),
            "frames_processed": frame_idx,
            "effective_fps": effective_fps,
            "processing_time_sec": total_processing_time
        },
        "vehicles": {
            "total_detections": c_total_vehicle_bounding_boxes,
            "unique_track_ids": len(track_history)
        },
        "plate_pipeline": {
            "total_vehicle_bounding_boxes": c_total_vehicle_bounding_boxes,
            "skipped_by_stride": c_skipped_by_stride,
            "skipped_by_size": c_skipped_by_size,
            "plate_yolo_calls_made": c_plate_yolo_calls_made,
            "successful_plate_detections": c_successful_plate_detections,
            "lpr_calls": c_lpr_calls
        },
        "recognition": stats,
        "tracks": tracks_out,
        "configuration": FROZEN_CONFIG
    }
    
    with open(report_path, "w") as f: json.dump(report, f, indent=2)
    with open(output_dir / "detections.json", "w") as f: json.dump(detections_log, f, indent=2)
    with open(output_dir / "tracks.json", "w") as f: json.dump(tracks_out, f, indent=2)
    
    return "PASS"

# ------------------------------------------------------------------------------
# Batch mode Main Entrypoint
# ------------------------------------------------------------------------------
def main():
    input_dir = Path("input/a11_benchmark")
    if not input_dir.exists():
        input_dir.mkdir(parents=True)
        print(f"No new videos found in {input_dir}")
        return

    video_files = []
    for ext in ["*.mp4", "*.avi", "*.mkv", "*.mov"]:
        video_files.extend(list(input_dir.glob(ext)))
        
    if not video_files:
        print(f"No new videos found in {input_dir}")
        return

    print("=" * 50)
    print("A.11 BATCH PIPELINE START")
    print("=" * 50)
    
    total_videos = len(video_files)
    processed = 0
    skipped = 0
    failures = 0
    results = {}

    for video_file in sorted(video_files):
        video_name = video_file.stem
        report_path = Path(f"output/a11/{video_name}/report.json")
        
        print(f"[{video_name}]")
        if report_path.exists():
            print("  SKIP: already processed\n")
            skipped += 1
            results[video_name] = "SKIP"
            continue
            
        try:
            status = process_video(video_file)
            processed += 1
            results[video_name] = status
            print(f"  {status}\n")
        except Exception as e:
            sys.stderr.write(f"  FAILED: {e}\n")
            failures += 1
            results[video_name] = "FAIL"

    print("=" * 50)
    print("A.11 BATCH PIPELINE COMPLETE")
    print("=" * 50)
    print(f"\nVideos discovered : {total_videos}")
    print(f"Videos processed  : {processed}")
    print(f"Videos skipped    : {skipped}")
    print(f"Failures          : {failures}")
    
    print("\nResults:")
    for v_name, status in results.items():
        print(f"  {v_name:<30} {status}")
        
    print("\nReports:")
    print("  output/a11/")
    print("=" * 50)

if __name__ == "__main__":
    main()
