import requests
import time
import threading

def start_server():
    import os
    os.system("./dev_start.sh")

t = threading.Thread(target=start_server, daemon=True)
t.start()

time.sleep(15) # Wait for it to start
try:
    print("Fetching port 5002 directly...")
    res1 = requests.get("http://127.0.0.1:5002/video_feed/usb_cam_0", stream=True, timeout=5)
    print("5002 Status:", res1.status_code)
    
    print("Fetching port 5001 proxy...")
    res2 = requests.get("http://127.0.0.1:5001/api/stream/usb_cam_0", stream=True, timeout=5)
    print("5001 Status:", res2.status_code)
except Exception as e:
    print("Error:", e)

import os
os.system("pkill -f live_scorer")
os.system("pkill -f face_api")
