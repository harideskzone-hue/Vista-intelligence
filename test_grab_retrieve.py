import cv2
import time

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Testing grab() vs retrieve() performance...")
t0 = time.time()
for i in range(100):
    cap.grab()
t1 = time.time()
print(f"100 grabs took: {t1-t0:.3f}s")

for i in range(100):
    cap.grab()
    cap.retrieve()
t2 = time.time()
print(f"100 grabs+retrieves took: {t2-t1:.3f}s")
cap.release()
