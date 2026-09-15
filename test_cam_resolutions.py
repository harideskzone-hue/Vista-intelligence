import cv2

print("Testing 1280x720...")
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
ret, frame = cap.read()
print("1280x720 Read frame:", ret, frame.shape if ret else "None")
cap.release()

print("\nTesting 1920x1080...")
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
ret, frame = cap.read()
print("1920x1080 Read frame:", ret, frame.shape if ret else "None")
cap.release()

print("\nTesting no property changes...")
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
ret, frame = cap.read()
print("Default Read frame:", ret, frame.shape if ret else "None")
cap.release()
