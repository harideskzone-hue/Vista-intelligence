import subprocess, sys, os, time
env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"
env["PYTHONPATH"] = ".:face_engine:face_api"
env["INFERENCE_MODE"] = "mps_batch"
env["OPTIMIZE_SCHEDULING"] = "true"
env["LAZY_ENCODING"] = "true"
env["INFERENCE_BATCH_SIZE"] = "1"

print("Starting api...")
api = subprocess.Popen([sys.executable, "face_api/run.py"], env=env)
time.sleep(3)
import requests
requests.post("http://localhost:5001/api/camera/start", timeout=2)

print("Starting wrapper...")
wrapper = subprocess.Popen([sys.executable, "endurance_wrapper.py"], env=env)
time.sleep(15)
api.terminate()
wrapper.terminate()
