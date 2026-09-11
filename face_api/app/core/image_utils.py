import base64
import numpy as np
import cv2
from typing import Optional, Tuple

def decode_base64_image(base64_str: str) -> Optional[np.ndarray]:
    """
    Decode base64 string to OpenCV image.
    
    Args:
        base64_str: Base64 encoded image string (with or without data:image/jpeg;base64, prefix)
        
    Returns:
        decoded OpenCV image or None if decoding fails
    """
    try:
        if ',' in base64_str:
            base64_str = base64_str.split(',')[1]
        
        img_bytes = base64.b64decode(base64_str)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return image
    except Exception:
        return None

def encode_image_to_bytes(image: np.ndarray, format: str = '.jpg') -> Optional[bytes]:
    """
    Encode OpenCV image to bytes.
    """
    try:
        _, buffer = cv2.imencode(format, image)
        return buffer.tobytes()
    except Exception:
        return None
