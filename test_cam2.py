import cv2

print("Testing exact sequence in live_scorer.py...")
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
if not cap.isOpened():
    print("Failed to open.")
else:
    print("Opened. Setting resolution to 640x360...")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    print("Reading frame...")
    ret, frame = cap.read()
    print("Read frame:", ret, frame.shape if ret else "None")
    
cap.release()
