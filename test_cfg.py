import os, sys, json
_CAMERAS_JSON = os.environ.get(
    "CAMERAS_JSON_PATH",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath('face_engine/live_scorer.py'))), "cameras.json")
)
print("PATH:", _CAMERAS_JSON)
try:
    with open(_CAMERAS_JSON, 'r') as f:
        data = json.load(f)
    print("DATA LEN:", len([c for c in data if isinstance(c, dict) and c.get('enabled', True)]))
except Exception as e:
    print("ERR:", e)
