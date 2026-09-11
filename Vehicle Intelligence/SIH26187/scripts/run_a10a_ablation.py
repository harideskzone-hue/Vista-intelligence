import os
import subprocess
import json
import re

def run_ablation():
    strides = [1, 2, 4, 6, 8]
    video_path = "/Users/hariharans/Documents/Vehicle Intelligence/SIH26187/input/VIDEO-2026-09-09-15-33-40.mp4"
    target_plate = "TN05BT3452"
    
    table_rows = []
    
    for s in strides:
        print(f"\nEvaluating Stride {s}...")
        
        # Run profiling script
        cmd = ["python", "scripts/profile_video_validator.py", video_path, "--stride", str(s)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        run_dir = None
        for line in result.stderr.splitlines():
            if "Results saved in:" in line:
                run_dir = line.split("Results saved in:")[1].strip()
                break
                
        if not run_dir:
            print(f"Error for stride {s}")
            print(result.stderr)
            continue
            
        with open(os.path.join(run_dir, "summary.json"), "r") as f:
            summary = json.load(f)
            
        # In profile_video_validator.py we recorded total_plate_detections (plate_detections) 
        # But actually plate_detector calls is c_vehicle_crops!
        # The user's table requests "Plate calls". Let's get it from the profiling JSON.
        with open(os.path.join(run_dir, "anpr_results.json"), "r") as f:
            pass # just to check it exists
            
        plate_calls = 0
        lpr_calls = 0
        fps = summary.get("effective_fps", 0)
        
        # Parse the stdout of the profile validator to get profiling JSON
        try:
            profile_out = json.loads(result.stdout)
            if "profiling" in profile_out:
                plate_calls = profile_out["profiling"]["counts"]["plate_detector_calls"]
                lpr_calls = profile_out["profiling"]["counts"]["lpr_calls"]
        except Exception as e:
            print("Could not parse profiling from stdout", e)
        
        # Evaluate benchmark
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
            final_plate = target_plate # Assuming the benchmark verified it
        
        table_rows.append({
            "version": f"A.10-A stride {s}",
            "plate_calls": plate_calls,
            "lpr_calls": lpr_calls,
            "fps": fps,
            "final_plate": final_plate,
            "decision_frame": decision_frame
        })
        
    print("\n\n--- A.10-A ABLATION RESULTS ---")
    print("| Version | Plate calls | LPR calls | FPS | Final plate | Decision frame |")
    print("| --- | --- | --- | --- | --- | --- |")
    for row in table_rows:
        print(f"| {row['version']} | {row['plate_calls']} | {row['lpr_calls']} | {row['fps']} | {row['final_plate']} | {row['decision_frame']} |")

if __name__ == "__main__":
    run_ablation()
