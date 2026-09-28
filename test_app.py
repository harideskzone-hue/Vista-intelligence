import subprocess, sys, os
env = os.environ.copy()
env["PYTHONPATH"] = ".:face_api"
env["CAMERAS_JSON_PATH"] = "cameras.json"
env["SIH26187_DATA"] = "data"
app = subprocess.Popen([sys.executable, "face_engine/live_scorer.py"], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in app.stdout:
    print(line, end="")
app.wait()
