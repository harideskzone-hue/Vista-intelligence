
import requests  # type: ignore # pyre-ignore # pyright: ignore
import base64
import cv2  # type: ignore # pyre-ignore # pyright: ignore
import numpy as np  # type: ignore # pyre-ignore # pyright: ignore

# Create a dummy image (random noise or solid color)
# Using white image to ensure detection if we were using a real detector, 
# but for logic test, just needs to be valid image data.
img = np.zeros((200, 200, 3), dtype=np.uint8) + 255
cv2.circle(img, (100, 100), 50, (0, 0, 255), -1) # Draw a red circle
_, buf = cv2.imencode('.jpg', img)
b64_str = base64.b64encode(buf).decode('utf-8')
data_url = f"data:image/jpeg;base64,{b64_str}"

print("Testing /api/add with POST...")
score = 0.99
name = "Script Test User"

payload = {
    "img": data_url,
    "score": score
    # Note: /api/add does NOT accept 'name' in body currently based on my reading, 
    # but my frontend logic calls /api/name immediately after. 
    # Let's test just the add first.
}

try:
    resp = requests.post(
        'http://127.0.0.1:5000/api/add', 
        json=payload
    )
    print(f"Add Status: {resp.status_code}")
    print(f"Add Response: {resp.text}")
    
    data = resp.json()
    if data.get('success'):
        person_id = data.get('person_id')
        print(f"Person ID: {person_id}")
        
        # Test Naming
        print("Testing /api/name...")
        name_payload = {"person_id": person_id, "name": name}
        name_resp = requests.post('http://127.0.0.1:5000/api/name', json=name_payload)
        print(f"Name Status: {name_resp.status_code}")
        print(f"Name Response: {name_resp.text}")
        
    else:
        print("Add failed, skipping name test.")

except Exception as e:
    print(f"Failed: {e}")
