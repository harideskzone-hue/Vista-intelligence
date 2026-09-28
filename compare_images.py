import cv2
import numpy as np
import sys

def mse(imageA, imageB):
    if imageA.shape != imageB.shape:
        return -1
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

img0 = cv2.imread('/Users/hariharans/.gemini/antigravity-ide/brain/f3daadea-3376-439d-a7d4-77d9950cc54b/scratch/cam_0.jpg')
img1 = cv2.imread('/Users/hariharans/.gemini/antigravity-ide/brain/f3daadea-3376-439d-a7d4-77d9950cc54b/scratch/cam_1.jpg')

if img0 is None or img1 is None:
    print("Could not load images.")
    sys.exit(1)

err = mse(img0, img1)
print(f"MSE between cam_0 and cam_1: {err:.2f}")

if err > 0 and err < 2000:
    print("Images are VERY similar (likely the exact same camera feed looking at the same scene)")
elif err >= 2000:
    print("Images are DIFFERENT")
else:
    print("Images are EXACTLY IDENTICAL")
