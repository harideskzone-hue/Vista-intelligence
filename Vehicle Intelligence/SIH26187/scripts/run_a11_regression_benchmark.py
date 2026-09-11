import os
import sys
import json
import subprocess
from pathlib import Path

def run_pipeline(video_path, stride, min_area, v_stride):
    cmd = [
        "python", "scripts/profile_video_validator.py", video_path, 
        "--stride", str(stride), 
        "--min_area", str(min_area),
        "--verification_stride", str(v_stride)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    run_dir = None
    for line in result.stderr.splitlines():
        if "Results saved in:" in line:
            run_dir = line.split("Results saved in:")[1].strip()
            break
            
    if not run_dir:
        print(f"Error running pipeline on {video_path}")
        print(result.stderr)
        return None, None
        
    # Read summary and vehicle tracks
    with open(os.path.join(run_dir, "summary.json"), "r") as f:
        summary = json.load(f)
    
    with open(os.path.join(run_dir, "vehicle_tracks.json"), "r") as f:
        vehicle_tracks = json.load(f)
        
    profiling = {}
    try:
        profile_out = json.loads(result.stdout)
        if "profiling" in profile_out:
            profiling = profile_out["profiling"]
    except:
        pass
        
    return summary, vehicle_tracks, profiling

def calculate_metrics(summary, vehicle_tracks, profiling, gt_plates):
    fps = summary.get("effective_fps", 0)
    video_fps = summary.get("video_fps", 30)
    total_frames = summary.get("frames_processed", 0)
    resolution = summary.get("resolution", "Unknown")
    duration_sec = total_frames / max(1, video_fps)
    
    plate_calls = summary.get("plate_yolo_calls_made", 0)
    total_boxes = summary.get("total_vehicle_bounding_boxes", 0)
    skipped_by_stride = summary.get("skipped_by_stride", 0)
    skipped_by_size = summary.get("skipped_by_size", 0)
    successful_plate_detections = summary.get("successful_plate_detections", 0)
    lpr_calls = summary.get("lpr_calls", 0)
    
    scheduling_reduction = 0.0
    if total_boxes > 0:
        scheduling_reduction = 1.0 - (plate_calls / total_boxes)
        
    plate_calls_per_frame = plate_calls / max(1, total_frames)
    lpr_calls_per_frame = lpr_calls / max(1, total_frames)
    
    unknown_count = summary.get("unknown_vehicles", 0)
    tracks_no_evidence = sum(1 for v in vehicle_tracks if v.get("observations", 0) == 0)
    short_track_count = sum(1 for v in vehicle_tracks if v.get("frames_seen", 0) < 15)
    long_track_count = sum(1 for v in vehicle_tracks if v.get("frames_seen", 0) >= 15)
    
    confirmed_tracks = [v for v in vehicle_tracks if v.get("plate_status") == "CONFIRMED"]
    
    matched_gt = set()
    correct_confirmations = 0
    false_confirmations = 0
    latency_sec_sum = 0.0
    stable_latency_sec_sum = 0.0
    latency_count = 0
    stable_latency_count = 0
    
    for c in confirmed_tracks:
        plate = c.get("plate")
        if plate in gt_plates:
            matched_gt.add(plate)
            correct_confirmations += 1
            first_frame = c.get("first_confirmed_frame")
            stable_frame = c.get("stable_confirmed_frame")
            if first_frame is not None:
                latency_sec_sum += (first_frame / video_fps)
                latency_count += 1
            if stable_frame is not None:
                stable_latency_sec_sum += (stable_frame / video_fps)
                stable_latency_count += 1
        else:
            false_confirmations += 1
            
    plate_recall = len(matched_gt) / max(1, len(gt_plates)) if gt_plates else 1.0
    missed_recognitions = len(gt_plates) - len(matched_gt)
    avg_latency = latency_sec_sum / max(1, latency_count) if latency_count > 0 else None
    avg_stable_latency = stable_latency_sec_sum / max(1, stable_latency_count) if stable_latency_count > 0 else None
    
    return {
        "fps": fps,
        "total_boxes": total_boxes,
        "skipped_by_stride": skipped_by_stride,
        "skipped_by_size": skipped_by_size,
        "plate_calls": plate_calls,
        "successful_plate_detections": successful_plate_detections,
        "lpr_calls": lpr_calls,
        "scheduling_reduction": scheduling_reduction,
        "plate_calls_per_frame": plate_calls_per_frame,
        "lpr_calls_per_frame": lpr_calls_per_frame,
        "unknown_count": unknown_count,
        "tracks_no_evidence": tracks_no_evidence,
        "short_track_count": short_track_count,
        "long_track_count": long_track_count,
        "confirmed_tracks_count": len(confirmed_tracks),
        "correct_confirmations": correct_confirmations,
        "false_confirmations": false_confirmations,
        "plate_recall": plate_recall,
        "missed_recognitions": missed_recognitions,
        "avg_latency_sec": avg_latency,
        "avg_stable_latency_sec": avg_stable_latency,
        "video_fps": video_fps,
        "resolution": resolution,
        "total_frames": total_frames,
        "duration_sec": duration_sec
    }

def main():
    input_dir = "input/a11_regression"
    os.makedirs(input_dir, exist_ok=True)
    gt_path = os.path.join(input_dir, "ground_truth.json")
    
    if os.path.exists(gt_path):
        with open(gt_path, "r") as f:
            ground_truth = json.load(f)
    else:
        print("No ground_truth.json found in input/")
        ground_truth = {}
        
    video_files = []
    for root, dirs, files in os.walk(input_dir):
        for f in files:
            if f.endswith((".mp4", ".avi")):
                video_files.append(os.path.relpath(os.path.join(root, f), input_dir))
    
    if not video_files:
        print(f"No videos found in {input_dir}/")
        return
        
    results = {}
        
    for vf in video_files:
        video_path = os.path.join(input_dir, vf)
        gt_data = ground_truth.get(vf, {})
        gt_plates = [v["plate"] for v in gt_data.get("vehicles", [])]
        
        print(f"\nProcessing {vf}...")
        
        print("  Running Original Configuration (s=1, a=0, v=1)...")
        sum_orig, trk_orig, prof_orig = run_pipeline(video_path, 1, 0, 1)
        
        print("  Running Optimized Configuration (s=6, a=10000, v=60)...")
        sum_opt, trk_opt, prof_opt = run_pipeline(video_path, 6, 10000, 60)
        
        if not sum_orig or not sum_opt:
            continue
            
        m_orig = calculate_metrics(sum_orig, trk_orig, prof_orig, gt_plates)
        m_opt = calculate_metrics(sum_opt, trk_opt, prof_opt, gt_plates)
        
        compute_reduction = 0.0
        if m_orig["plate_calls"] > 0:
            compute_reduction = ((m_orig["plate_calls"] - m_opt["plate_calls"]) / m_orig["plate_calls"]) * 100
            
        fps_improvement = 0.0
        if m_orig["fps"] > 0:
            fps_improvement = ((m_opt["fps"] - m_orig["fps"]) / m_orig["fps"]) * 100
            
        results[vf] = {
            "Original": m_orig,
            "Optimized": m_opt,
            "Improvements": {
                "compute_reduction_pct": compute_reduction,
                "fps_improvement_pct": fps_improvement
            }
        }
        
    # Write JSON report
    with open("phase_4k_a11_regression_report.json", "w") as f:
        json.dump(results, f, indent=2)
        
    # Write Markdown report
    with open("phase_4k_a11_regression_report.md", "w") as f:
        f.write("# Phase 4K-A.11 Multi-Condition Regression Benchmark\n\n")
        
        for vf, res in results.items():
            f.write(f"## Video: `{vf}`\n\n")
            
            orig = res["Original"]
            opt = res["Optimized"]
            
            f.write(f"**Source Video Metadata:**\n")
            f.write(f"- Resolution: {orig['resolution']}\n")
            f.write(f"- Frame Rate: {orig['video_fps']} FPS\n")
            f.write(f"- Frame Count: {orig['total_frames']} frames\n")
            f.write(f"- Duration: {orig['duration_sec']:.2f} seconds\n\n")
            
            f.write(f"**Overall Improvements:**\n")
            f.write(f"- Scheduling Reduction: **{res['Optimized']['scheduling_reduction']*100:.1f}%**\n")
            f.write(f"- FPS Improvement: **{res['Improvements']['fps_improvement_pct']:.1f}%**\n\n")
            
            f.write("| Metric | Original (s1/a0/v1) | Optimized (s6/a10k/v60) |\n")
            f.write("| :--- | :--- | :--- |\n")
            
            orig = res["Original"]
            opt = res["Optimized"]
            
            f.write(f"| FPS | {orig['fps']} | **{opt['fps']}** |\n")
            f.write(f"| Total Vehicle Boxes | {orig['total_boxes']} | {opt['total_boxes']} |\n")
            f.write(f"| Skipped by Stride | {orig['skipped_by_stride']} | **{opt['skipped_by_stride']}** |\n")
            f.write(f"| Skipped by Size | {orig['skipped_by_size']} | **{opt['skipped_by_size']}** |\n")
            f.write(f"| Plate YOLO calls | {orig['plate_calls']} | **{opt['plate_calls']}** |\n")
            f.write(f"| Successful Plate Detections | {orig['successful_plate_detections']} | **{opt['successful_plate_detections']}** |\n")
            f.write(f"| Plate calls/frame | {orig['plate_calls_per_frame']:.2f} | **{opt['plate_calls_per_frame']:.2f}** |\n")
            f.write(f"| LPR calls | {orig['lpr_calls']} | **{opt['lpr_calls']}** |\n")
            f.write(f"| LPR calls/frame | {orig['lpr_calls_per_frame']:.2f} | **{opt['lpr_calls_per_frame']:.2f}** |\n")
            f.write(f"| Plate Recall | {orig['plate_recall']*100:.1f}% | {opt['plate_recall']*100:.1f}% |\n")
            f.write(f"| Missed Recognitions | {orig['missed_recognitions']} | {opt['missed_recognitions']} |\n")
            f.write(f"| Correct Confirmations | {orig['correct_confirmations']} | {opt['correct_confirmations']} |\n")
            f.write(f"| False Confirmations | {orig['false_confirmations']} | {opt['false_confirmations']} |\n")
            
            lat_o = f"{orig['avg_latency_sec']:.2f}s" if orig['avg_latency_sec'] else "N/A"
            lat_op = f"{opt['avg_latency_sec']:.2f}s" if opt['avg_latency_sec'] else "N/A"
            f.write(f"| Avg First Confirm Latency | {lat_o} | {lat_op} |\n")
            
            slat_o = f"{orig['avg_stable_latency_sec']:.2f}s" if orig['avg_stable_latency_sec'] else "N/A"
            slat_op = f"{opt['avg_stable_latency_sec']:.2f}s" if opt['avg_stable_latency_sec'] else "N/A"
            f.write(f"| Avg Stable Confirm Latency | {slat_o} | {slat_op} |\n")
            
            f.write(f"| UNKNOWN Count | {orig['unknown_count']} | {opt['unknown_count']} |\n")
            f.write(f"| Tracks w/ No Evidence | {orig['tracks_no_evidence']} | {opt['tracks_no_evidence']} |\n")
            f.write(f"| Short Tracks (<15 frames) | {orig['short_track_count']} | {opt['short_track_count']} |\n")
            f.write(f"| Long Tracks (>=15 frames) | {orig['long_track_count']} | {opt['long_track_count']} |\n")
            
            f.write("\n---\n")
            
    print("Benchmark complete! Reports saved to phase_4k_a11_regression_report.json and .md")

if __name__ == "__main__":
    main()
