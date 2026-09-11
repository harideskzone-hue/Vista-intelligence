import cv2
import numpy as np

class AdaptiveEnhancer:
    """
    Enhances low-light or night images adaptively to improve Face Detection
    and Recognition accuracy without altering the original evidence.
    """
    def __init__(self):
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # clipLimit=2.0 is conservative to avoid overly harsh noise amplification
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def enhance(self, frame: np.ndarray, mode: str = "AUTO") -> np.ndarray:
        """
        Enhance the image based on mode:
        - DAY: return original frame
        - AUTO: enhance only if the image is too dark
        - NIGHT: always enhance
        """
        if mode == "DAY":
            return frame

        # Check brightness
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)

        if mode == "AUTO" and mean_brightness > 85:
            # Image is bright enough, skip enhancement to avoid artifacts
            return frame

        # Convert to LAB color space for luminance-only enhancement
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE to L-channel
        cl = self.clahe.apply(l)

        # Merge channels back
        limg = cv2.merge((cl, a, b))

        # Convert back to BGR
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return enhanced
