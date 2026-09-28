import json
import time
import subprocess
import os

CAM_FILE = "cameras.json"

def set_cameras(active_ids):
    with open(CAM_FILE, "r") as f:
        cams = json.load(f)
    for c in cams:
        c["enabled"] = c["id"] in active_ids
    with open(CAM_FILE, "w") as f:
        json.dump(cams, f, indent=2)

configs = [
    ("Cam 0", ["cam_bf81043c"]),
    ("Cam 1", ["cam_f34fa2d3"]),
    ("Cam 3", ["cam_2251b1fb"]),
    ("Cam 0+1", ["cam_bf81043c", "cam_f34fa2d3"]),
    ("Cam 0+1+3", ["cam_bf81043c", "cam_f34fa2d3", "cam_2251b1fb"])
]

print("Starting Systematic Camera Tests...")

for name, ids in configs:
    print(f"\n=======================")
    print(f"Testing Config: {name}")
    print(f"=======================")
    set_cameras(ids)
    
    # Enable system state
    with open("data/camera_state.json", "w") as f:
        json.dump({"active": True, "updated_at": time.time()}, f)
    
    # Run the live scorer
    proc = subprocess.Popen(["python3", "face_engine/live_scorer.py"], 
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                            
    test_duration = 30 if len(ids) < 3 else 180  # Shortened 3-cam test to 3 minutes for automated run
    start_time = time.time()
    
    while time.time() - start_time < test_duration:
        try:
            line = proc.stdout.readline()
            if not line: break
            line = line.strip()
            if "Negotiated:" in line or "Failed to open" in line or "overread" in line or "FPS" in line or "MATCH" in line:
                print(line)
        except Exception:
            pass
            
    print(f"Stopping test for {name}...")
    proc.terminate()
    proc.wait()

print("\nTests completed.")
