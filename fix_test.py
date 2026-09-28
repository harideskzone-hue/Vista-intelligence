import subprocess, sys, os

env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"
env["PYTHONPATH"] = ".:face_engine:face_api"

print("Testing face_api/run.py directly...")
p_api = subprocess.Popen([sys.executable, "face_api/run.py"], env=env, stdout=open("api_debug.log", "w"), stderr=subprocess.STDOUT)

import time
time.sleep(2)

print("Testing endurance_wrapper.py directly...")
p_live = subprocess.Popen([sys.executable, "endurance_wrapper.py"], env=env, stdout=open("live_debug.log", "w"), stderr=subprocess.STDOUT)

time.sleep(10)
p_api.terminate()
p_live.terminate()
