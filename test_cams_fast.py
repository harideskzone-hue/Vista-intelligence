import cv2
import time

for i in range(4):
    print(f"\n--- Testing index {i} ---")
    cap = cv2.VideoCapture(i)
    if not cap.isOpened():
        print(f"Index {i}: Failed to open")
        continue
    
    # Try reading a frame
    success = False
    for _ in range(10):
        ret, frame = cap.read()
        if ret and frame is not None:
            success = True
            break
        time.sleep(0.1)
    
    if success:
        print(f"Index {i}: SUCCESS ({frame.shape})")
    else:
        print(f"Index {i}: Opened, but failed to read frame")
    cap.release()
