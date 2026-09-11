
import requests
import base64
import cv2
import numpy as np
import json

# Create a black dummy image
img = np.zeros((100, 100, 3), dtype=np.uint8)
_, buf = cv2.imencode('.jpg', img)
b64_str = base64.b64encode(buf).decode('utf-8')
data_url = f"data:image/jpeg;base64,{b64_str}"

print("Testing /api/search with POST...")
try:
    resp = requests.post(
        'http://127.0.0.1:8000/api/search',  # Assuming default port 8000 based on run.py typical usage
        json={'img': data_url}
    )
    print(f"Status: {resp.status_code}")
    print(f"Response: {resp.text}")
except Exception as e:
    print(f"Failed: {e}")
