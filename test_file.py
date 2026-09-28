import sys, os
sys.path.append("face_engine")
import live_scorer
print("LIVE SCORER FILE:", live_scorer.__file__)
print("CAMERAS JSON PATH:", live_scorer._CAMERAS_JSON)
