with open("run_phase7f_endurance.py", "r") as f:
    content = f.read()

import re

old_wrapper_code = r"""    def write_wrapper(self):
        wrapper_code = r\"\"\"import time
import threading
import json
import cv2
import traceback
import sys
import psutil
import os
sys.path.append("face_engine")
import live_scorer

# INJECT JSON FIX FOR MISSING IMPORT IN LIVE SCORER
live_scorer.json = json

original_process_frame = live_scorer.CameraProcessor.process_frame
fps_lock = threading.Lock()
fps_values = []

def patched_process_frame(self):
    original_process_frame(self)
    if hasattr(self, 'display_fps'):
        with fps_lock:
            fps_values.append(self.display_fps)

live_scorer.CameraProcessor.process_frame = patched_process_frame

def monitor_telemetry():
    start_time = time.time()
    proc = psutil.Process()
    with open("wrapper_telemetry.log", "w") as f:
        while True:
            time.sleep(10)
            now = time.time()
            elapsed_min = (now - start_time) / 60.0
            cpu = proc.cpu_percent()
            rss = proc.memory_info().rss / (1024 * 1024)
            with fps_lock:
                if fps_values:
                    avg_fps = sum(fps_values) / len(fps_values)
                    fps_values.clear()
                else:
                    avg_fps = 0.0
            msg = f"[{time.strftime('%H:%M:%S')}] Min {elapsed_min:.1f}: CPU={cpu:.1f}%, RSS={rss:.1f}MB, FPS={avg_fps:.1f}\n"
            f.write(msg)
            f.flush()

threading.Thread(target=monitor_telemetry, daemon=True).start()

try:
    live_scorer.main()
except Exception as e:
    with open("wrapper_crash.log", "w") as f:
        traceback.print_exc(file=f)
\"\"\"
        with open("endurance_wrapper.py", "w") as f:
            f.write(wrapper_code)"""

new_wrapper_code = r"""    def write_wrapper(self):
        wrapper_code = r\"\"\"import time
import threading
import traceback
import sys
import psutil
import os
import subprocess
sys.path.append("face_engine")
import live_scorer

# Log configuration environment to verify resolved variables
env_output = subprocess.check_output("env | grep -E 'INFERENCE|OPTIMIZE|LAZY'", shell=True, text=True)
print("\n--- Resolved Runtime Environment ---")
print(env_output)
print("------------------------------------\n")

def monitor_telemetry():
    start_time = time.time()
    proc = psutil.Process()
    with open("wrapper_telemetry.log", "w") as f:
        while True:
            time.sleep(10)
            now = time.time()
            elapsed_min = (now - start_time) / 60.0
            cpu = proc.cpu_percent()
            rss = proc.memory_info().rss / (1024 * 1024)
            
            # Observational telemetry (no monkey patching)
            fps_values = []
            if hasattr(live_scorer, 'processors'):
                for cam_id, processor in live_scorer.processors.items():
                    if hasattr(processor, 'display_fps'):
                        fps_values.append(processor.display_fps)
            avg_fps = sum(fps_values) / len(fps_values) if fps_values else 0.0
            
            queue_depth = 0
            if hasattr(live_scorer, '_sync_processors') and hasattr(live_scorer._sync_processors, 'executor'):
                if hasattr(live_scorer._sync_processors.executor, '_work_queue'):
                    queue_depth = live_scorer._sync_processors.executor._work_queue.qsize()
            
            reconnects = 0
            if hasattr(live_scorer, 'retry_counts'):
                reconnects = sum(live_scorer.retry_counts.values())
            
            msg = f"[{time.strftime('%H:%M:%S')}] Min {elapsed_min:.1f}: CPU={cpu:.1f}%, RSS={rss:.1f}MB, FPS={avg_fps:.1f}, Queue={queue_depth}, Reconnects={reconnects}\n"
            f.write(msg)
            f.flush()

threading.Thread(target=monitor_telemetry, daemon=True).start()

try:
    live_scorer.main()
except Exception as e:
    with open("wrapper_crash.log", "w") as f:
        traceback.print_exc(file=f)
\"\"\"
        with open("endurance_wrapper.py", "w") as f:
            f.write(wrapper_code)"""

content = content.replace(old_wrapper_code, new_wrapper_code)

# Change cam file
content = content.replace('self.cam_file = "cameras.json"', 'self.cam_file = "/tmp/sih26187_phase7f_cameras.json"')

with open("run_phase7f_endurance.py", "w") as f:
    f.write(content)
print("Replaced wrapper and cam file!")
