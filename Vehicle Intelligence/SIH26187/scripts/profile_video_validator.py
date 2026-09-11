import os
import cv2
import json
import re
import time
import uuid
import sys
import argparse
import numpy as np
import urllib.request
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import List, Optional

from ultralytics import YOLO
from fast_plate_ocr import LicensePlateRecognizer

# ------------------------------------------------------------------------------
# Simple temporal voter bypassing Indian format validation (for UK video test)
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
        self.observations: List[Observation] = []
        self.min_observations = min_observations
        self.min_evidence = min_evidence

    def add(self, text: str, ocr_conf: float, det_conf: float, qual_score: float, frame: int, char_probs: list):
        if not text: return
        self.observations.append(Observation(text, ocr_conf, det_conf, qual_score, frame, char_probs))

    def _cluster_observations(self) -> List[List[Observation]]:
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

    def _align_and_vote(self, cluster: List[Observation]) -> tuple[str, float]:
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
        
        # Indian Plate format regex: State (2 letters) + District (1-2 digits) + Optional Letters (0-3) + Numbers (4 digits)
        # e.g., TN05BT3452, KA511234, MH01AB1234
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
# Main Execution
# ------------------------------------------------------------------------------
def run(video_input, stride=1, min_area=0, verification_stride=0):
    if verification_stride == 0: verification_stride = stride
    run_id = f"s{stride}_a{min_area}_v{verification_stride}_" + str(uuid.uuid4())[:8]
    output_dir = Path(f"output/video_tests/{run_id}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    video_path = video_input
    if video_input.startswith("http://") or video_input.startswith("https://"):
        sys.stderr.write(f"Downloading video from {video_input}...\n")
        video_path = str(output_dir / "downloaded_video.mp4")
        urllib.request.urlretrieve(video_input, video_path)
    
    if not os.path.exists(video_path):
        sys.stderr.write(f"Error: {video_path} not found.\n")
        return
        
    # Models
    vehicle_model_path = "yolo11n.pt"
    plate_model_path = "anpr/models/best.pt"
    ocr_model_path = "data/fast_plate_ocr/models/fine_tuned/2026-09-09_11-29-31/best.onnx"
    ocr_config_path = "data/fast_plate_ocr/models/cct_xs_v2_global_plate_config.yaml"
    
    vehicle_model = YOLO(vehicle_model_path)
    plate_model = YOLO(plate_model_path)
    ocr_model = LicensePlateRecognizer(onnx_model_path=ocr_model_path, plate_config_path=ocr_config_path)
    
    # Video setup
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    out_vid = cv2.VideoWriter(
        str(output_dir / "annotated_video.mp4"),
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps, (width, height)
    )
    
    # State tracking
    voters = {} # track_id -> GenericTemporalVoter
    track_history = {} # track_id -> {first_seen, last_seen, class_name}
    
    # JSON evidence collectors
    detections_log = []
    recognition_events_log = []
    
    frame_idx = 0
    t0_all = time.time()
    
    total_plate_detections = 0
    total_ocr_observations = 0
    
    # Profiling variables
    t_decode_total = 0.0
    t_vehicle_total = 0.0
    t_plate_total = 0.0
    t_lpr_total = 0.0
    t_voter_total = 0.0
    
    c_total_vehicle_bounding_boxes = 0
    c_skipped_by_stride = 0
    c_skipped_by_size = 0
    c_plate_yolo_calls_made = 0
    c_successful_plate_detections = 0
    c_lpr_calls = 0
    c_voter_calls = 0
    
    sys.stderr.write(f"Starting execution on {video_path} ({total_frames} frames)\n")
    
    while True:
        t_decode_start = time.time()
        ret, frame = cap.read()
        t_decode_total += (time.time() - t_decode_start)
        
        if not ret:
            break
            
        frame_idx += 1
        annotated_frame = frame.copy()
        
        # 1. Vehicle Detection & Tracking
        t_veh_start = time.time()
        results = vehicle_model.track(frame, classes=[2, 3, 5, 7], persist=True, tracker="bytetrack.yaml", verbose=False)
        t_vehicle_total += (time.time() - t_veh_start)
        
        if results and results[0].boxes and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            confidences = results[0].boxes.conf.cpu().numpy()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            
            for box, track_id, conf, cls_id in zip(boxes, track_ids, confidences, class_ids):
                x1, y1, x2, y2 = map(int, box)
                cls_name = vehicle_model.names[cls_id]
                
                # Track history
                if track_id not in track_history:
                    track_history[track_id] = {
                        "first_seen": frame_idx, 
                        "class": cls_name, 
                        "frames_seen": 0,
                        "status": "UNKNOWN",
                        "ever_confirmed": False,
                        "reopened": False,
                        "first_confirmed_frame": None,
                        "stable_confirmed_frame": None
                    }
                track_history[track_id]["last_seen"] = frame_idx
                track_history[track_id]["frames_seen"] += 1
                
                c_total_vehicle_bounding_boxes += 1
                
                # A.10-C Evidence-aware Scheduling
                current_status = track_history[track_id]["status"]
                effective_stride = verification_stride if current_status == "CONFIRMED" else stride
                
                # A.10-A Temporal Stride (using effective stride)
                if (track_history[track_id]["frames_seen"] - 1) % effective_stride != 0:
                    c_skipped_by_stride += 1
                    continue
                
                # Ensure voter exists
                if track_id not in voters:
                    voters[track_id] = GenericTemporalVoter(min_evidence=0.4)
                
                # Crop vehicle
                veh_crop = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
                if veh_crop.size == 0: continue
                
                detections_log.append({
                    "frame": frame_idx, "track_id": track_id, "type": "vehicle",
                    "bbox": [x1, y1, x2, y2], "confidence": float(conf), "class": cls_name
                })
                
                # A.10-B Size Gating
                veh_area = (x2 - x1) * (y2 - y1)
                if veh_area < min_area:
                    c_skipped_by_size += 1
                    continue
                
                # 2. Plate Detection within vehicle
                c_plate_yolo_calls_made += 1
                t_plate_start = time.time()
                lp_results = plate_model.predict(veh_crop, verbose=False)
                t_plate_total += (time.time() - t_plate_start)
                
                if lp_results and len(lp_results[0].boxes) > 0:
                    best_lp_idx = np.argmax(lp_results[0].boxes.conf.cpu().numpy())
                    lp_box = lp_results[0].boxes.xyxy[best_lp_idx].cpu().numpy()
                    lp_conf = float(lp_results[0].boxes.conf[best_lp_idx].cpu().numpy())
                    
                    lx1, ly1, lx2, ly2 = map(int, lp_box)
                    # Global coords
                    gx1, gy1, gx2, gy2 = x1 + lx1, y1 + ly1, x1 + lx2, y1 + ly2
                    
                    plate_crop = veh_crop[max(0, ly1):min(veh_crop.shape[0], ly2), max(0, lx1):min(veh_crop.shape[1], lx2)]
                    if plate_crop.size > 0:
                        c_successful_plate_detections += 1
                        # OCR
                        try:
                            # fast-plate-ocr expects RGB
                            rgb_crop = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB)
                            t_lpr_start = time.time()
                            c_lpr_calls += 1
                            preds = ocr_model.run(rgb_crop)
                            t_lpr_total += (time.time() - t_lpr_start)
                            
                            if preds:
                                ocr_text = getattr(preds[0], 'plate', "").replace("_", "")
                                probs = getattr(preds[0], 'char_probs', [])
                                if probs is None: probs = []
                                ocr_conf = float(np.mean(probs)) if len(probs) > 0 else 1.0
                                
                                # Quality score heuristic
                                lp_height = gy2 - gy1
                                quality_score = min(1.0, (lp_height / 40.0)) * lp_conf
                                
                                if ocr_text:
                                    total_ocr_observations += 1
                                    c_voter_calls += 1
                                    t_voter_start = time.time()
                                    voters[track_id].add(
                                        ocr_text, ocr_conf, lp_conf, quality_score, frame_idx, probs
                                    )
                                    t_voter_total += (time.time() - t_voter_start)
                                    
                                    # Update track status for scheduling
                                    new_decision = voters[track_id].decide()
                                    if new_decision:
                                        new_status = new_decision["status"]
                                        prev_status = track_history[track_id]["status"]
                                        
                                        if prev_status == "CONFIRMED" and new_status != "CONFIRMED":
                                            track_history[track_id]["reopened"] = True
                                            
                                        if new_status == "CONFIRMED":
                                            track_history[track_id]["ever_confirmed"] = True
                                            if track_history[track_id]["first_confirmed_frame"] is None:
                                                track_history[track_id]["first_confirmed_frame"] = frame_idx
                                            if track_history[track_id].get("stable_confirmed_frame") is None:
                                                track_history[track_id]["stable_confirmed_frame"] = frame_idx
                                        else:
                                            track_history[track_id]["stable_confirmed_frame"] = None
                                            
                                        track_history[track_id]["status"] = new_status
                                    
                                    recognition_events_log.append({
                                        "frame": frame_idx, "track_id": track_id, "plate_text": ocr_text,
                                        "recognition_confidence": ocr_conf, "plate_detection_confidence": lp_conf,
                                        "quality_score": quality_score, "char_probs": probs
                                    })
                                    
                                    # Annotate plate box
                                    cv2.rectangle(annotated_frame, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)
                                    cv2.putText(annotated_frame, f"{ocr_text} ({ocr_conf:.2f})", (gx1, gy1 - 5),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        except Exception as e:
                            sys.stderr.write(f"OCR Error on frame {frame_idx}: {e}\n")
                
                # Annotate vehicle box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                # Render current best temporal vote
                current_decision = voters[track_id].decide()
                disp_text = current_decision['text'] if current_decision['text'] else "???"
                label = f"ID:{track_id} {cls_name} [{disp_text}]"
                
                # Draw background rectangle for text visibility
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                ty = max(20, y1 - 10)
                cv2.rectangle(annotated_frame, (x1, ty - th - 5), (x1 + tw, ty + 5), (0, 0, 0), -1)
                cv2.putText(annotated_frame, label, (x1, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                            
        out_vid.write(annotated_frame)
        if frame_idx % 50 == 0:
            sys.stderr.write(f"Processed {frame_idx}/{total_frames} frames...\n")
            
    cap.release()
    out_vid.release()
    t1_all = time.time()
    
    # End of video processing
    vehicle_tracks_log = []
    anpr_results_log = []
    
    stats = {
        "plates_recognized": 0,
        "confirmed_plates": 0,
        "probable_plates": 0,
        "unknown_plates": 0
    }
    
    trajectory_counter = 1000 # Dummy association id base
    
    for t_id, voter in voters.items():
        decision = voter.decide()
        traj_id = trajectory_counter + t_id # Simple mock trajectory ID for now
        
        if decision['status'] == 'CONFIRMED': stats['confirmed_plates'] += 1
        elif decision['status'] == 'PROBABLE': stats['probable_plates'] += 1
        else: stats['unknown_plates'] += 1
        
        if decision['text']: stats['plates_recognized'] += 1
        
        track_info = {
            "trajectory_id": traj_id,
            "track_id": t_id,
            "vehicle_class": track_history[t_id]["class"],
            "first_seen_frame": track_history[t_id]["first_seen"],
            "last_seen_frame": track_history[t_id]["last_seen"],
            "plate": decision['text'],
            "plate_status": decision['status'],
            "evidence_score": decision['score'],
            "observations": decision['observations'],
            "frames_seen": track_history[t_id]["frames_seen"],
            "first_confirmed_frame": track_history[t_id].get("first_confirmed_frame"),
            "stable_confirmed_frame": track_history[t_id].get("stable_confirmed_frame")
        }
        vehicle_tracks_log.append(track_info)
        
        if decision['status'] in ['CONFIRMED', 'PROBABLE']:
            anpr_results_log.append(track_info)
            
    vehicles_with_obs = sum(1 for v in voters.values() if len(v.observations) > 0)
    
    summary_log = {
        "run_id": run_id,
        "source_video": os.path.basename(video_path),
        "models": {
            "vehicle_detector": vehicle_model_path,
            "plate_detector": plate_model_path,
            "plate_recognizer": ocr_model_path
        },
        "video_fps": fps,
        "resolution": f"{width}x{height}",
        "frames_processed": frame_idx,
        "vehicle_detections": len(voters),
        "unique_track_ids": len(voters),
        "total_vehicle_bounding_boxes": c_total_vehicle_bounding_boxes,
        "scheduled_plate_candidates": c_plate_yolo_calls_made + c_skipped_by_size,
        "skipped_by_stride": c_skipped_by_stride,
        "skipped_by_size": c_skipped_by_size,
        "plate_yolo_calls_made": c_plate_yolo_calls_made,
        "successful_plate_detections": c_successful_plate_detections,
        "lpr_calls": c_lpr_calls,
        "ocr_observations": total_ocr_observations,
        "vehicles_with_plate_observations": vehicles_with_obs,
        "confirmed_plates": stats['confirmed_plates'],
        "probable_plates": stats['probable_plates'],
        "unknown_vehicles": stats['unknown_plates'],
        "processing_time_sec": round(t1_all - t0_all, 2),
        "effective_fps": round(frame_idx / max((t1_all - t0_all), 1), 2)
    }
    
    # Save JSONs
    t_json_start = time.time()
    with open(output_dir / "detections.json", "w") as f: json.dump(detections_log, f, indent=2)
    with open(output_dir / "recognition_events.json", "w") as f: json.dump(recognition_events_log, f, indent=2)
    with open(output_dir / "vehicle_tracks.json", "w") as f: json.dump(vehicle_tracks_log, f, indent=2)
    with open(output_dir / "anpr_results.json", "w") as f: json.dump(anpr_results_log, f, indent=2)
    with open(output_dir / "summary.json", "w") as f: json.dump(summary_log, f, indent=2)
    t_json_total = time.time() - t_json_start
    
    sys.stderr.write(f"\n--- Validation Complete! ---\n")
    sys.stderr.write(f"Results saved in: {output_dir}\n")
    
    c_ever_confirmed = sum(1 for t in track_history.values() if t.get("ever_confirmed"))
    c_reopened = sum(1 for t in track_history.values() if t.get("reopened"))
    
    profiling_log = {
        "timings": {
            "video_decode_ms_per_frame": round((t_decode_total / max(1, frame_idx)) * 1000, 2),
            "vehicle_yolo_ms_per_frame": round((t_vehicle_total / max(1, frame_idx)) * 1000, 2),
            "plate_yolo_ms_per_crop": round((t_plate_total / max(1, c_plate_yolo_calls_made)) * 1000, 2),
            "lpr_ms_per_crop": round((t_lpr_total / max(1, c_lpr_calls)) * 1000, 2),
            "voter_ms_per_obs": round((t_voter_total / max(1, c_voter_calls)) * 1000, 2),
            "json_logging_ms_total": round(t_json_total * 1000, 2)
        },
        "counts": {
            "total_vehicle_bounding_boxes": c_total_vehicle_bounding_boxes,
            "skipped_by_stride": c_skipped_by_stride,
            "skipped_by_size": c_skipped_by_size,
            "plate_yolo_calls_made": c_plate_yolo_calls_made,
            "successful_plate_detections": c_successful_plate_detections,
            "lpr_calls": c_lpr_calls,
            "voter_calls": c_voter_calls,
            "tracks_ever_confirmed": c_ever_confirmed,
            "tracks_reopened": c_reopened
        },
        "performance": {
            "total_processing_time_sec": round(t1_all - t0_all, 2),
            "effective_fps": summary_log["effective_fps"]
        }
    }
    
    final_output = {
        "summary": summary_log,
        "profiling": profiling_log,
        "results": anpr_results_log
    }
    print(json.dumps(final_output, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", type=str, help="Path to video")
    parser.add_argument("--stride", type=int, default=1, help="Temporal stride")
    parser.add_argument("--min_area", type=int, default=0, help="Min vehicle area (A.10-B)")
    parser.add_argument("--verification_stride", type=int, default=0, help="Stride after CONFIRMED (A.10-C)")
    args = parser.parse_args()
    
    run(args.video_path, args.stride, args.min_area, args.verification_stride)
