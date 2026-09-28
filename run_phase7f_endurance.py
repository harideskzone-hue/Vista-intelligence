import subprocess
import time
import sys
import os
import json
import psutil
import traceback

class Phase7fRunner:
    def __init__(self, mode="mps_batch", duration_mins=35):
        self.mode = mode
        self.duration = duration_mins * 60
        self.mtx = None
        self.publishers = []
        self.api = None
        self.live = None
        self.cam_file = "/tmp/sih26187_phase7f_cameras.json"
        
    def write_cams(self, num_cams=12):
        cams = [{"id": f"cam{i}", "source": f"rtsp://localhost:8554/cam{i}", "label": f"Cam {i}", "enabled": True} for i in range(1, num_cams+1)]
        with open(self.cam_file, "w") as f:
            json.dump(cams, f)

    def start_mediamtx(self):
        self.mtx = subprocess.Popen(["./mediamtx"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(3)

    def start_publishers(self, num_cams=12):
        for i in range(1, num_cams+1):
            cmd = ["ffmpeg", "-stream_loop", "-1", "-re", "-i", "test_video.mp4", "-c", "copy", "-f", "rtsp", f"rtsp://localhost:8554/cam{i}"]
            p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.publishers.append(p)
            print(f"[{time.strftime('%H:%M:%S')}] Started publisher cam{i}")

    def write_wrapper(self):
        wrapper_code = r"""import time
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
"""
        with open("endurance_wrapper.py", "w") as f:
            f.write(wrapper_code)

    def start_system(self):
        env = os.environ.copy()
        env["INFERENCE_MODE"] = self.mode
        if self.mode == "mps_batch":
            env["INFERENCE_BATCH_SIZE"] = "1"
        env["OPTIMIZE_SCHEDULING"] = "true"
        env["LAZY_ENCODING"] = "true"
        env["CAMERAS_JSON_PATH"] = os.path.abspath(self.cam_file)
        env["PYTHONPATH"] = ".:face_engine:face_api"
        env["PYTHONUNBUFFERED"] = "1"
        
        self.api = subprocess.Popen([sys.executable, "face_api/run.py"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(2)
        
        self.write_wrapper()
        
        with open("live_out.log", "w") as out:
            self.live = subprocess.Popen([sys.executable, "endurance_wrapper.py"], env=env, stdout=out, stderr=subprocess.STDOUT)
            
        print(f"[{time.strftime('%H:%M:%S')}] System started")

    def monitor(self):
        start_time = time.time()
        cam_drop_schedule = {10: 0, 20: 4, 30: 8} 
        
        with open("phase7f_data.csv", "w") as logfile:
            logfile.write("Min,CPU,RSS,FPS,Queue,Reconnects\n")
            
            while time.time() - start_time < self.duration:
                elapsed_min = (time.time() - start_time) / 60.0
                
                # Check for scheduled camera drops
                for m, idx in list(cam_drop_schedule.items()):
                    if elapsed_min >= m and idx < len(self.publishers) and self.publishers[idx] is not None:
                        self.publishers[idx].terminate()
                        self.publishers[idx] = None
                        msg = f"[{time.strftime('%H:%M:%S')}] Stopped publisher cam{idx+1}"
                        print(msg)
                        del cam_drop_schedule[m]

                # Print telemetry
                if os.path.exists("wrapper_telemetry.log"):
                    with open("wrapper_telemetry.log", "r") as f:
                        lines = f.readlines()
                        if lines:
                            last_line = lines[-1].strip()
                            print(f"[{time.strftime('%H:%M:%S')}] {last_line}")
                            logfile.write(last_line + "\n")
                            logfile.flush()
                            # clear it so we don't double print
                            open("wrapper_telemetry.log", "w").close()

                time.sleep(10)
                
    def cleanup(self):
        if self.live: self.live.terminate()
        if self.api: self.api.terminate()
        for p in self.publishers:
            if p: p.terminate()
        if self.mtx: self.mtx.terminate()
        
        os.system("pkill -f endurance_wrapper.py")
        os.system("pkill -f face_api/run.py")
        os.system("pkill -f mediamtx")
        os.system("pkill -f ffmpeg")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        dur = int(sys.argv[1])
    else:
        dur = 35
    runner = Phase7fRunner(mode="mps_batch", duration_mins=dur)
    runner.write_cams(12)
    runner.start_mediamtx()
    runner.start_publishers(12)
    runner.start_system()
    try:
        runner.monitor()
    finally:
        runner.cleanup()
