import cv2

print("Scanning for available cameras (indices 0 to 5)...")
for i in range(6):
    cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"Camera {i}: SUCCESS. Resolution: {frame.shape[1]}x{frame.shape[0]}")
        else:
            print(f"Camera {i}: Opened but failed to read frame.")
        cap.release()
    else:
        print(f"Camera {i}: Failed to open.")
