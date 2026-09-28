import os
import subprocess
import threading
import sys

# 1. Setup Environment Variables
os.environ["PORT"] = os.environ.get("PORT", "10000")
os.environ["SIH26187_DATA"] = os.path.abspath("data")
os.environ["SIH26187_VEHICLE_INT"] = os.path.abspath("Vehicle Intelligence/SIH26187")
os.makedirs(os.environ["SIH26187_DATA"], exist_ok=True)

# 2. Start the AI Engine (Live Scorer) as a background process
def start_live_scorer():
    print("Starting VISTA AI Engine in background...")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath("face_api") + ":" + os.path.abspath(".")
    subprocess.run(["python", "-u", "face_engine/live_scorer.py"], env=env)

threading.Thread(target=start_live_scorer, daemon=True).start()

# 3. Export FastAPI app for Uvicorn
sys.path.insert(0, os.path.abspath("face_api"))
from face_api.run import app

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    print(f"Starting Uvicorn server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
