import os
import subprocess
import threading
import sys
import time

# 1. Setup Environment Variables for Hugging Face
os.environ["PORT"] = "7860" # Hugging Face maps this port to the public URL
os.environ["SIH26187_DATA"] = os.path.abspath("data")
os.environ["SIH26187_VEHICLE_INT"] = os.path.abspath("Vehicle Intelligence/SIH26187")

# Ensure data directory exists
os.makedirs(os.environ["SIH26187_DATA"], exist_ok=True)

# 2. Start the AI Engine (Live Scorer) as a background process
def start_live_scorer():
    print("Starting VISTA AI Engine in background...")
    # Add face_engine to PYTHONPATH so it finds its modules
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath("face_api") + ":" + os.path.abspath(".")
    subprocess.run(["python", "-u", "face_engine/live_scorer.py"], env=env)

threading.Thread(target=start_live_scorer, daemon=True).start()

# 3. Expose the FastAPI app
# Hugging Face Spaces (Gradio SDK) will look for an object named 'app' in 'app.py'
# and run it using Uvicorn if it detects it's a FastAPI app!
sys.path.insert(0, os.path.abspath("face_api"))

# Import the main FastAPI application instance from our existing code
from face_api.run import app
