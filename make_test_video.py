import cv2
import numpy as np

print("Generating 15s synthetic 1080p test video...")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('test_video.mp4', fourcc, 30.0, (1920, 1080))

for i in range(450):
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    # Draw something moving so it's not totally static
    x = int(1920 * (i / 450.0))
    cv2.circle(frame, (x, 540), 100, (0, 255, 0), -1)
    cv2.putText(frame, f"Frame {i}", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,255,255), 3)
    out.write(frame)

out.release()
print("test_video.mp4 created successfully.")
