import cv2
import numpy as np

original = cv2.imencode
def patched(ext, img, params=None):
    res = original(ext, img, params)
    return res

cv2.imencode = patched

img = np.zeros((100, 100, 3), dtype=np.uint8)
res = cv2.imencode('.jpg', img)
print("Type of res:", type(res))
print("Type of res[0]:", type(res[0]))
print("Type of res[1]:", type(res[1]))
