import cv2
import numpy as np
from typing import Tuple, List, Optional
from fast_plate_ocr import LicensePlateRecognizer

from core.config import config

class PlateOCR:
    """
    Production License Plate Recognition Engine using fine-tuned FastPlateOCR ONNX model.
    """
    def __init__(
        self,
        onnx_model_path: str = config.ocr_model_path,
        plate_config_path: str = config.ocr_config_path
    ):
        self.onnx_model_path = str(onnx_model_path)
        self.plate_config_path = str(plate_config_path)
        self.recognizer = LicensePlateRecognizer(
            onnx_model_path=self.onnx_model_path,
            plate_config_path=self.plate_config_path
        )

    def recognize(self, plate_crop: np.ndarray) -> Tuple[str, float, List[float]]:
        """
        Runs OCR on a BGR plate crop.
        
        Returns:
            (plate_text, mean_confidence, char_probs)
        """
        if plate_crop is None or plate_crop.size == 0:
            return "", 0.0, []

        try:
            rgb_crop = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB)
            preds = self.recognizer.run(rgb_crop)
            if not preds:
                return "", 0.0, []

            plate_text = getattr(preds[0], 'plate', "").replace("_", "").strip()
            probs = getattr(preds[0], 'char_probs', [])
            mean_conf = float(np.mean(probs)) if probs else (1.0 if plate_text else 0.0)

            return plate_text, mean_conf, probs
        except Exception:
            return "", 0.0, []
