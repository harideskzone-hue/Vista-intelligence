import os, sys, subprocess, time, signal
env = os.environ.copy()
env["PYTHONPATH"] = ".:face_engine:face_api"
env["INFERENCE_MODE"] = "legacy"
p = subprocess.Popen([sys.executable, "endurance_wrapper.py"], env=env)
time.sleep(10)
os.kill(p.pid, signal.SIGTERM)
p.wait()
print("Wrapper exited with code:", p.returncode)
