import os
import subprocess
import json

def run_ablation():
    # Fixed baseline from A.10-B
    stride = 6
    min_area = 10000
    
    # Verification Strides to test
    v_strides = [6, 12, 24, 30, 60]
    
    video_path = "/Users/hariharans/Documents/Vehicle Intelligence/SIH26187/input/VIDEO-2026-09-09-15-33-40.mp4"
    target_plate = "TN05BT3452"
    
    table_rows = []
    
    for vs in v_strides:
        print(f"\n--- Evaluating Verification Stride {vs} ---")
        
        # Run profiling script
        cmd = [
            "python", "scripts/profile_video_validator.py", video_path, 
            "--stride", str(stride), 
            "--min_area", str(min_area),
            "--verification_stride", str(vs)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        run_dir = None
        for line in result.stderr.splitlines():
            if "Results saved in:" in line:
                run_dir = line.split("Results saved in:")[1].strip()
                break
                
        if not run_dir:
            print(f"Error for v_stride {vs}")
            print(result.stderr)
            continue
            
        with open(os.path.join(run_dir, "summary.json"), "r") as f:
            summary = json.load(f)
            
        plate_calls = 0
        lpr_calls = 0
        fps = summary.get("effective_fps", 0)
        unknown_vehicles = summary.get("unknown_vehicles", 0)
        
        ever_confirmed = 0
        reopened = 0
        
        try:
            profile_out = json.loads(result.stdout)
            if "profiling" in profile_out:
                c = profile_out["profiling"]["counts"]
                plate_calls = c.get("plate_detector_calls", 0)
                lpr_calls = c.get("lpr_calls", 0)
                ever_confirmed = c.get("tracks_ever_confirmed", 0)
                reopened = c.get("tracks_reopened", 0)
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
            
        table_rows.append({
            "version": f"v_stride {vs}",
            "plate_calls": plate_calls,
            "lpr_calls": lpr_calls,
            "fps": fps,
            "final_plate": final_plate,
            "decision_frame": decision_frame,
            "incorrect_confirmed": incorrect_confirmed,
            "unknown_vehicles": unknown_vehicles,
            "ever_confirmed": ever_confirmed,
            "reopened": reopened
        })
        
    print("\n\n--- A.10-C ABLATION RESULTS ---")
    print("| Verif. Stride | Plate YOLO | LPR calls | FPS | Final plate | Stable Frame | Incorrect Confirmed | Reopened Tracks |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in table_rows:
        print(f"| {r['version']} | {r['plate_calls']} | {r['lpr_calls']} | {r['fps']} | {r['final_plate']} | {r['decision_frame']} | {r['incorrect_confirmed']} | {r['reopened']} |")

if __name__ == "__main__":
    run_ablation()
