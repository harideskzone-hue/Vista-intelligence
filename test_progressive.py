import json
import time
import subprocess
import os

CAM_FILE = "cameras.json"

# Backup original cameras
import shutil
shutil.copy(CAM_FILE, CAM_FILE + ".bak")

def create_mock_cameras(n):
    cams = []
    for i in range(n):
        cams.append({
            "id": f"cam_mock_{i}",
            "name": f"Mock Camera {i}",
            "source": "/Users/hariharans/Documents/SIH26187/test_video.mp4",
            "type": "IP",
            "enabled": True,
            "boundary_pts": []
        })
    with open(CAM_FILE, "w") as f:
        json.dump(cams, f, indent=2)

print("Starting dev_start.sh in background...")
proc = subprocess.Popen(["./dev_start.sh"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

# Wait for API to start
time.sleep(10)

with open("data/camera_state.json", "w") as f:
    json.dump({"active": True, "updated_at": time.time()}, f)

import threading
def log_reader():
    while True:
        line = proc.stdout.readline()
        if not line: break
        line = line.strip()
        if "TELEMETRY" in line or "error" in line.lower() or "overread" in line:
            print(line)

threading.Thread(target=log_reader, daemon=True).start()

for n in range(3, 9):
    print(f"\n=======================")
    print(f"Testing Config: {n} Cameras")
    print(f"=======================")
    
    # We want to test batch_size=8 as well, but for now we just change camera count.
    # INFERENCE_BATCH_SIZE is read via os.environ by live_scorer.py, but we started it via dev_start.sh
    # By default, it's 4. I will leave it at 4 for this test, then test 8.
    
    create_mock_cameras(n)
    
    # Wait for hot-reload (5s) + stabilization (30s)
    time.sleep(40)

print("Tests completed. Stopping dev_start.sh...")
os.system("pkill -P " + str(proc.pid))
proc.terminate()

# Restore original cameras
shutil.move(CAM_FILE + ".bak", CAM_FILE)
