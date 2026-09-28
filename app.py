import os
import subprocess
import threading
import sys
import gradio as gr
from fastapi import FastAPI

# 1. Setup Environment Variables for Hugging Face
os.environ["PORT"] = "7860"
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

# 3. Import FastAPI app
sys.path.insert(0, os.path.abspath("face_api"))
from face_api.run import app as fastapi_app

# 4. Create dummy ZeroGPU Gradio block to satisfy Hugging Face Space supervisor
import spaces

@spaces.GPU
def _dummy_gpu():
    return "GPU Ready"

with gr.Blocks() as demo:
    gr.Markdown("# VISTA AI - System Running\nThe FastAPI server is handling requests in the background.")
    btn = gr.Button("Wake GPU")
    out = gr.Textbox()
    btn.click(_dummy_gpu, inputs=[], outputs=[out])

# Mount Gradio into FastAPI
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

