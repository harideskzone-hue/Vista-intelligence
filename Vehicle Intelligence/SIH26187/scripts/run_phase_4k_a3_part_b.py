import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import pandas as pd
from tqdm import tqdm
import time
import json
import Levenshtein
from collections import defaultdict
import subprocess
import os

from anpr.plate_format import PlateFormatValidator

def normalize_name(name):
    n = str(name).lower()
    for char in [' ', '(', ')', '-', '_']:
        n = n.replace(char, '')
    return n

def load_ground_truth():
    INPUT_DIR = Path("input/indian_anpr_zenodo")
    excel_path = INPUT_DIR / "number_plate.xlsx"
    images_dir = INPUT_DIR / "images"
    
    df = pd.read_excel(excel_path)
    col_img = df.columns[0]
    col_gt = df.columns[1]
    
    actual_files = list(images_dir.rglob("*"))
    actual_files = [f for f in actual_files if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    
    norm_to_file = {}
    for f in actual_files:
        norm_to_file[normalize_name(f.stem)] = f
        
    mapped = {}
    for _, row in df.iterrows():
        img_val = str(row[col_img]).strip()
        gt_val = str(row[col_gt]).strip().upper().replace(" ", "")
        stem = Path(img_val).stem
        norm = normalize_name(stem)
        if norm in norm_to_file:
            mapped[str(norm_to_file[norm])] = gt_val
            
    return mapped

# OCR Engines
import easyocr
import pytesseract
import logging
logging.getLogger('ppocr').setLevel(logging.ERROR)
from paddleocr import PaddleOCR

def main():
    gt_map = load_ground_truth()
    detector = YOLO("anpr/models/best.pt")
    
    print("Initializing OCR engines...")
    easy_reader = easyocr.Reader(['en'], gpu=True)
    paddle_reader = PaddleOCR(lang='en')
    
    engines = ["EasyOCR", "PaddleOCR", "Tesseract"]
    
    metrics = {e: defaultdict(int) for e in engines}
    latencies = {e: [] for e in engines}
    confusions = {e: defaultdict(int) for e in engines}
    
    for img_path_str, gt_text in tqdm(gt_map.items(), desc="Running OCR Benchmark"):
        img = cv2.imread(img_path_str)
        if img is None: continue
        
        results = detector(img, verbose=False)
        if len(results) == 0 or len(results[0].boxes) == 0:
            continue # Only benchmark detected plates
            
        best_box = max(results[0].boxes, key=lambda b: b.conf[0].item())
        x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
        plate_crop = img[max(0, y1):min(img.shape[0], y2), max(0, x1):min(img.shape[1], x2)]
        
        if plate_crop.size == 0: continue
            
        for e in engines:
            metrics[e]["detected"] += 1
            metrics[e]["expected_chars"] += len(gt_text)
            
        # 1. EasyOCR
        t0 = time.time()
        easy_res = easy_reader.readtext(plate_crop)
        latencies["EasyOCR"].append(time.time() - t0)
        
        raw_easy = ""
        if easy_res:
            best = max(easy_res, key=lambda x: x[2])
            raw_easy = best[1].upper().replace(" ", "")
            
        # 2. PaddleOCR
        t0 = time.time()
        pad_res = paddle_reader.ocr(plate_crop)
        latencies["PaddleOCR"].append(time.time() - t0)
        
        raw_pad = ""
        if pad_res and isinstance(pad_res, dict) and 'rec_texts' in pad_res:
            raw_pad = "".join(pad_res['rec_texts']).upper().replace(" ", "")
        elif pad_res and isinstance(pad_res, list) and pad_res[0]:
            texts = [line[1][0] for line in pad_res[0]]
            raw_pad = "".join(texts).upper().replace(" ", "")
            
        # 3. Tesseract
        t0 = time.time()
        raw_tess = pytesseract.image_to_string(plate_crop, config='--psm 7').strip().upper().replace(" ", "")
        latencies["Tesseract"].append(time.time() - t0)
        
        preds = {"EasyOCR": raw_easy, "PaddleOCR": raw_pad, "Tesseract": raw_tess}
        
        for e, raw_pred in preds.items():
            fmt = PlateFormatValidator.validate(raw_pred)
            norm_pred = fmt.normalized_text or raw_pred
            
            if norm_pred:
                metrics[e]["ocr_non_empty"] += 1
                
            lev = Levenshtein.distance(norm_pred, gt_text)
            metrics[e]["levenshtein"] += lev
            
            if norm_pred == gt_text:
                metrics[e]["exact"] += 1
            
            if fmt.normalized_text == gt_text:
                metrics[e]["normalized_exact"] += 1
                
            ops = Levenshtein.editops(norm_pred, gt_text)
            for op, p_idx, gt_idx in ops:
                if op == 'replace':
                    confusions[e]["substitution"] += 1
                elif op == 'insert':
                    confusions[e]["insertion"] += 1
                elif op == 'delete':
                    confusions[e]["deletion"] += 1
                    
    print("\n| Metric | EasyOCR | PaddleOCR | Tesseract |")
    print("| --- | ---: | ---: | ---: |")
    
    def val(m, total, dec=2):
        return f"{m/total*100:.{dec}f}%" if total > 0 else "0.00%"
        
    row_empty = "| OCR non-empty | " + " | ".join([val(metrics[e]["ocr_non_empty"], metrics[e]["detected"]) for e in engines]) + " |"
    
    cer = []
    for e in engines:
        lev = metrics[e]["levenshtein"]
        exp = metrics[e]["expected_chars"]
        cer.append(val(lev, exp))
    row_cer = "| CER | " + " | ".join(cer) + " |"
    
    row_exact = "| Exact plate accuracy | " + " | ".join([val(metrics[e]["exact"], metrics[e]["detected"]) for e in engines]) + " |"
    row_norm_exact = "| Correctly normalized exact accuracy | " + " | ".join([val(metrics[e]["normalized_exact"], metrics[e]["detected"]) for e in engines]) + " |"
    
    row_sub = "| Substitution | " + " | ".join([str(confusions[e]["substitution"]) for e in engines]) + " |"
    row_ins = "| Insertion | " + " | ".join([str(confusions[e]["insertion"]) for e in engines]) + " |"
    row_del = "| Deletion | " + " | ".join([str(confusions[e]["deletion"]) for e in engines]) + " |"
    
    row_lat = "| Mean latency/image | " + " | ".join([f"{np.mean(latencies[e])*1000:.1f}ms" for e in engines]) + " |"
    
    for r in [row_empty, row_cer, row_exact, row_norm_exact, row_sub, row_ins, row_del, row_lat]:
        print(r)
        
    print(f"\nEvaluation Environment: M-Series Mac, MPS backend enabled (if applicable).")

if __name__ == "__main__":
    main()
