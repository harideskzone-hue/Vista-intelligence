import cv2

print("Testing 640x480...")
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
ret, frame = cap.read()
print("640x480 Read frame:", ret, frame.shape if ret else "None")
cap.release()
