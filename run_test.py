import subprocess
import time
import os
import json

os.environ["LAZY_ENCODING"] = "true"
os.environ["OPTIMIZE_SCHEDULING"] = "true"
os.environ["INFERENCE_FPS_CAP"] = "10"
os.environ["INFERENCE_BATCH_SIZE"] = "8"

with open("data/camera_state.json", "w") as f:
    json.dump({"active": True, "updated_at": time.time()}, f)

print("Starting 45s Hardware Verification...")
proc = subprocess.Popen(["./dev_start.sh"])
time.sleep(45)
os.system("pkill -P " + str(proc.pid))
proc.terminate()
print("Done.")
