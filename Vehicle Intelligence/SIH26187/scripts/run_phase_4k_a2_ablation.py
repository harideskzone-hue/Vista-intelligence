import os
import cv2
import json
import gc
import torch
import pandas as pd
from pathlib import Path
import numpy as np
from ultralytics import YOLO
import easyocr
import Levenshtein
from collections import defaultdict
from tqdm import tqdm

from anpr.plate_quality import PlateQualityScorer
from anpr.plate_format import PlateFormatValidator

INPUT_DIR = Path("input/indian_anpr_zenodo")
OUTPUT_DIR = Path("output/phase_4k_a2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VARIANTS = [
    "Current baseline",
    "Grayscale + 2x",
    "CLAHE",
    "Otsu",
    "Adaptive",
    "Contrast",
    "Sharpening"
]

def normalize_name(name):
    n = str(name).lower()
    for char in [' ', '(', ')', '-', '_']:
        n = n.replace(char, '')
    return n

def load_ground_truth():
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

def apply_variant(crop, variant):
    if variant == "Current baseline":
        return crop
        
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    
    if variant == "Grayscale + 2x":
        return cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        
    elif variant == "CLAHE":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        return clahe.apply(gray)
        
    elif variant == "Otsu":
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh
        
    elif variant == "Adaptive":
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
    elif variant == "Contrast":
        return cv2.convertScaleAbs(gray, alpha=1.5, beta=0)
        
    elif variant == "Sharpening":
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        return cv2.filter2D(crop, -1, kernel)
        
    return crop

def run_ablation():
    gt_map = load_ground_truth()
    
    print(f"1,696 of 1,700 dataset annotations were deterministically mapped one-to-one.")
    print(f"Four records with ambiguous filename correspondence were excluded rather than force-mapped.\n")
    
    detector = YOLO("anpr/models/best.pt")
    reader = easyocr.Reader(['en'], gpu=True)
    
    # Metrics structures
    # variant -> metric -> value
    metrics = {v: defaultdict(int) for v in VARIANTS + ["Best-of oracle"]}
    confusion_matrix = defaultdict(int) # (pred_char, gt_char) -> count
    
    # quality bucket -> variant -> metric -> value
    quality_metrics = {
        "HIGH": {v: defaultdict(int) for v in VARIANTS},
        "MEDIUM": {v: defaultdict(int) for v in VARIANTS},
        "LOW": {v: defaultdict(int) for v in VARIANTS}
    }
    
    for img_path_str, gt_text in tqdm(gt_map.items(), desc="Running Ablation"):
        img = cv2.imread(img_path_str)
        if img is None:
            continue
            
        for v in VARIANTS + ["Best-of oracle"]:
            metrics[v]["total"] += 1
            metrics[v]["expected_chars"] += len(gt_text)
            
        # 1. Detect
        results = detector(img, verbose=False)
        plate_crop = None
        
        if len(results) > 0 and len(results[0].boxes) > 0:
            best_box = max(results[0].boxes, key=lambda b: b.conf[0].item())
            x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
            plate_crop = img[max(0, y1):min(img.shape[0], y2), max(0, x1):min(img.shape[1], x2)]
            
        if plate_crop is None or plate_crop.size == 0:
            for v in VARIANTS + ["Best-of oracle"]:
                metrics[v]["levenshtein"] += len(gt_text)
            continue
            
        for v in VARIANTS + ["Best-of oracle"]:
            metrics[v]["detected"] += 1
            
        # Quality score
        q_score = PlateQualityScorer.compute(plate_crop)
        if q_score >= 0.70: q_bucket = "HIGH"
        elif q_score >= 0.40: q_bucket = "MEDIUM"
        else: q_bucket = "LOW"
        
        for v in VARIANTS:
            quality_metrics[q_bucket][v]["total"] += 1
            quality_metrics[q_bucket][v]["expected_chars"] += len(gt_text)
            quality_metrics[q_bucket][v]["detected"] += 1
            
        # 2. Variants
        oracle_best_lev = len(gt_text)
        oracle_exact = False
        oracle_non_empty = False
        
        for variant in VARIANTS:
            v_crop = apply_variant(plate_crop, variant)
            ocr_results = reader.readtext(v_crop)
            
            raw_text = ""
            if ocr_results:
                best_ocr = max(ocr_results, key=lambda x: x[2])
                raw_text = best_ocr[1].upper().replace(" ", "")
                
            fmt_result = PlateFormatValidator.validate(raw_text)
            pred_text = fmt_result.normalized_text or raw_text
            
            if pred_text:
                metrics[variant]["ocr_non_empty"] += 1
                quality_metrics[q_bucket][variant]["ocr_non_empty"] += 1
                oracle_non_empty = True
                
            lev_dist = Levenshtein.distance(pred_text, gt_text)
            metrics[variant]["levenshtein"] += lev_dist
            quality_metrics[q_bucket][variant]["levenshtein"] += lev_dist
            
            if pred_text == gt_text:
                metrics[variant]["exact"] += 1
                quality_metrics[q_bucket][variant]["exact"] += 1
                oracle_exact = True
                oracle_best_lev = 0
            else:
                oracle_best_lev = min(oracle_best_lev, lev_dist)
                
            # Track character confusion ONLY for the Current baseline
            if variant == "Current baseline":
                ops = Levenshtein.editops(pred_text, gt_text)
                for op, p_idx, gt_idx in ops:
                    if op == 'replace':
                        confusion_matrix[(pred_text[p_idx], gt_text[gt_idx])] += 1
                    elif op == 'insert':
                        confusion_matrix[('', gt_text[gt_idx])] += 1
                    elif op == 'delete':
                        confusion_matrix[(pred_text[p_idx], '')] += 1
                        
        # Oracle Tracking
        if oracle_non_empty: metrics["Best-of oracle"]["ocr_non_empty"] += 1
        metrics["Best-of oracle"]["levenshtein"] += oracle_best_lev
        if oracle_exact: metrics["Best-of oracle"]["exact"] += 1
        
        # Memory Management to prevent leak
        del results, plate_crop, img
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
    print("\n| Variant | Detected | OCR non-empty | CER | Detection-conditioned exact | End-to-end exact |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for v in VARIANTS + ["Best-of oracle"]:
        m = metrics[v]
        det = m["detected"]
        total = m["total"]
        non_empty = m["ocr_non_empty"]
        lev = m["levenshtein"]
        exact = m["exact"]
        exp_chars = m["expected_chars"]
        
        cer = (lev / exp_chars * 100) if exp_chars > 0 else 0
        det_exact = (exact / det * 100) if det > 0 else 0
        e2e_exact = (exact / total * 100) if total > 0 else 0
        
        v_label = v + "*" if v == "Best-of oracle" else v
        print(f"| {v_label} | {det} | {non_empty} | {cer:.2f}% | {det_exact:.2f}% | {e2e_exact:.2f}% |")
        
    print("\n* Best-of oracle is diagnostic only, representing the theoretical ceiling if a perfect ensemble routing existed.\n")
    
    print("### Initial Diagnostic Quality Buckets\n")
    for bucket in ["HIGH", "MEDIUM", "LOW"]:
        print(f"**{bucket} QUALITY**")
        print("| Variant | Images | CER | Det-conditioned Exact |")
        print("| --- | ---: | ---: | ---: |")
        for v in VARIANTS:
            m = quality_metrics[bucket][v]
            det = m["detected"]
            lev = m["levenshtein"]
            exact = m["exact"]
            exp_chars = m["expected_chars"]
            
            cer = (lev / exp_chars * 100) if exp_chars > 0 else 0
            det_exact = (exact / det * 100) if det > 0 else 0
            print(f"| {v} | {det} | {cer:.2f}% | {det_exact:.2f}% |")
        print("")
        
    print("### Alignment-Aware Character Confusion (Current Baseline)\n")
    print("| Operation | Prediction | Ground Truth | Count |")
    print("| --- | --- | --- | ---: |")
    sorted_confusions = sorted(confusion_matrix.items(), key=lambda x: x[1], reverse=True)
    for (pred_c, gt_c), count in sorted_confusions[:25]:
        if pred_c == '' and gt_c != '': op = "Deletion"
        elif pred_c != '' and gt_c == '': op = "Insertion"
        else: op = "Substitution"
        
        pred_disp = pred_c if pred_c else "[NONE]"
        gt_disp = gt_c if gt_c else "[NONE]"
        print(f"| {op} | {pred_disp} | {gt_disp} | {count} |")

if __name__ == "__main__":
    run_ablation()
