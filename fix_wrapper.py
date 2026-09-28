with open("run_phase7f_endurance.py", "r") as f:
    content = f.read()

new_wrapper = r"""    def write_wrapper(self):
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

original_process_frame = live_scorer.CameraProcessor.process_frame
fps_lock = threading.Lock()
fps_values = []
reconnects_ref = [0]

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
                avg_fps = sum(fps_values) / len(fps_values) if fps_values else 0.0
                fps_values.clear()
            
            queue_depth = 0
            if hasattr(live_scorer, '_sync_processors') and hasattr(live_scorer._sync_processors, 'executor'):
                if hasattr(live_scorer._sync_processors.executor, '_work_queue'):
                    queue_depth = live_scorer._sync_processors.executor._work_queue.qsize()
            
            reconnects = 0
            if hasattr(live_scorer, 'retry_counts'):
                reconnects = sum(live_scorer.retry_counts.values())
            
            msg = f"{elapsed_min:.1f},{cpu:.1f},{rss:.1f},{avg_fps:.1f},{queue_depth},{reconnects}\n"
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

import re
content = re.sub(r'    def write_wrapper\(self\):.*?        with open\("endurance_wrapper.py", "w"\) as f:\n            f.write\(wrapper_code\)', new_wrapper, content, flags=re.DOTALL)

with open("run_phase7f_endurance.py", "w") as f:
    f.write(content)

print("Replaced wrapper with monkey patch observing!")
