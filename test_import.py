import os, sys
env = os.environ.copy()
os.environ["PYTHONPATH"] = ".:face_engine:face_api"
sys.path.append(".")
sys.path.append("face_engine")
sys.path.append("face_api")

print("Importing get_correlation_engine...")
try:
    from edge.events.correlation import get_correlation_engine
    print("SUCCESS")
except Exception as e:
    print("ERR:", e)
