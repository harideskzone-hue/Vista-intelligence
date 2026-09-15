import cv2
import sys

print("Testing camera 0 without CAP_AVFOUNDATION...")
cap1 = cv2.VideoCapture(0)
if cap1.isOpened():
    ret1, frame1 = cap1.read()
    print("Cap1 Opened. Read frame:", ret1, frame1.shape if ret1 else "None")
else:
    print("Cap1 Failed to open.")
cap1.release()

print("\nTesting camera 0 with CAP_AVFOUNDATION...")
cap2 = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
if cap2.isOpened():
    ret2, frame2 = cap2.read()
    print("Cap2 Opened. Read frame:", ret2, frame2.shape if ret2 else "None")
else:
    print("Cap2 Failed to open.")
cap2.release()
