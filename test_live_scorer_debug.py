import sys, os, time, threading
os.environ["PYTHONPATH"] = ".:face_engine:face_api"
sys.path.append("face_engine")

import live_scorer

original_get_active_cameras = live_scorer._get_active_cameras
def debug_get_active_cameras():
    print(f"[DEBUG] _CAMERAS_JSON is {live_scorer._CAMERAS_JSON}")
    try:
        with open(live_scorer._CAMERAS_JSON, 'r') as f:
            print("[DEBUG] File contents:", f.read()[:50], "...")
    except Exception as e:
        print("[DEBUG] Failed to read:", e)
    res = original_get_active_cameras()
    print("[DEBUG] _get_active_cameras returned:", len(res), "cameras")
    return res

live_scorer._get_active_cameras = debug_get_active_cameras
live_scorer._is_system_active = lambda: True

def force_stop():
    time.sleep(5)
    os._exit(0)
threading.Thread(target=force_stop, daemon=True).start()

print("[DEBUG] Starting live_scorer.main()")
live_scorer.main()
