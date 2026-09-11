import os
import subprocess
import json

def run_ablation():
    # Strides to test: just 6
    stride = 6
    
    # Areas to test
    min_areas = [0, 2500, 5000, 10000, 20000]
    
    video_path = "/Users/hariharans/Documents/Vehicle Intelligence/SIH26187/input/VIDEO-2026-09-09-15-33-40.mp4"
    target_plate = "TN05BT3452"
    
    table_rows = []
    
    for area in min_areas:
        print(f"\n--- Evaluating Stride {stride}, Min Area {area} ---")
        
        # Run profiling script
        cmd = ["python", "scripts/profile_video_validator.py", video_path, "--stride", str(stride), "--min_area", str(area)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        run_dir = None
        for line in result.stderr.splitlines():
            if "Results saved in:" in line:
                run_dir = line.split("Results saved in:")[1].strip()
                break
                
        if not run_dir:
            print(f"Error for area {area}")
            print(result.stderr)
            continue
            
        with open(os.path.join(run_dir, "summary.json"), "r") as f:
            summary = json.load(f)
            
        plate_calls = 0
        lpr_calls = 0
        successful_plates = 0
        crops_skipped = 0
        crops_processed = 0
        total_time = 0.0
        fps = summary.get("effective_fps", 0)
        ocr_observations = summary.get("ocr_observations", 0)
        unknown_vehicles = summary.get("unknown_vehicles", 0)
        
        try:
            profile_out = json.loads(result.stdout)
            if "profiling" in profile_out:
                c = profile_out["profiling"]["counts"]
                p = profile_out["profiling"]["performance"]
                plate_calls = c.get("plate_detector_calls", 0)
                lpr_calls = c.get("lpr_calls", 0)
                successful_plates = c.get("successful_plate_detections", 0)
                crops_skipped = c.get("vehicle_crops_skipped_size", 0)
                crops_processed = c.get("vehicle_crops_processed", 0)
                total_time = p.get("total_processing_time_sec", 0.0)
        except Exception as e:
            print("Could not parse profiling from stdout", e)
            
        # Parse anpr_results.json for incorrect CONFIRMED
        incorrect_confirmed = 0
        with open(os.path.join(run_dir, "anpr_results.json"), "r") as f:
            anpr = json.load(f)
            for track in anpr:
                if track["plate_status"] == "CONFIRMED" and track["plate"] != target_plate:
                    incorrect_confirmed += 1
                    
        # Evaluate benchmark for target plate
        bench_cmd = ["python", "scripts/eval_temporal_benchmark.py", run_dir, target_plate]
        bench_result = subprocess.run(bench_cmd, capture_output=True, text=True)
        
        final_plate = "UNKNOWN"
        decision_frame = "N/A"
        
        for line in bench_result.stdout.splitlines():
            if "Final pipeline decision:" in line:
                final_plate = line.split(":")[-1].strip()
            if "Stable decision reached at frame:" in line:
                decision_frame = line.split(":")[-1].strip()
        
        if final_plate == "CONFIRMED":
            final_plate = target_plate
            
        total_encountered = crops_skipped + crops_processed
            
        table_rows.append({
            "version": f"Area {area}",
            "encountered": total_encountered,
            "skipped": crops_skipped,
            "plate_calls": plate_calls,
            "lpr_calls": lpr_calls,
            "successful_plate": successful_plates,
            "total_time": total_time,
            "fps": fps,
            "final_plate": final_plate,
            "decision_frame": decision_frame,
            "incorrect_confirmed": incorrect_confirmed,
            "unknown_vehicles": unknown_vehicles
        })
        
    print("\n\n--- A.10-B ABLATION RESULTS ---")
    print("| Gate (Area) | Crops Encountered | Crops Skipped | Plate YOLO | LPR calls | Total Time | FPS | Final plate | Stable Frame | Incorrect Confirmed |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in table_rows:
        print(f"| {r['version']} | {r['encountered']} | {r['skipped']} | {r['plate_calls']} | {r['lpr_calls']} | {r['total_time']}s | {r['fps']} | {r['final_plate']} | {r['decision_frame']} | {r['incorrect_confirmed']} |")

if __name__ == "__main__":
    run_ablation()
