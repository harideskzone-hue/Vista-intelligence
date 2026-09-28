import sys, os
sys.path.append("face_engine")
from live_scorer import CentralInferenceWorker
try:
    print("About to call CentralInferenceWorker")
    w = CentralInferenceWorker(batch_size=1)
    print("Success")
except Exception as e:
    print("Exception caught:", e)
