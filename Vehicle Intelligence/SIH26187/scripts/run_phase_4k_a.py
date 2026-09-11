import os
import cv2
import json
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
import easyocr
import Levenshtein
from tqdm import tqdm

from anpr.plate_quality import PlateQualityScorer
from anpr.plate_format import PlateFormatValidator
from anpr.fuzzy_temporal_voter import FuzzyTemporalVoter

INPUT_DIR = Path("input/indian_anpr_zenodo")
OUTPUT_DIR = Path("output/phase_4k")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_ground_truth():
    xlsx_files = list(INPUT_DIR.rglob("*.xlsx"))
    if not xlsx_files:
        raise FileNotFoundError("Could not find the Excel file in the dataset directory.")
        
    df = pd.read_excel(xlsx_files[0])
    
    # We expect columns like "Image Name" and "Number Plate"
    # Let's standardize them
    col_img = None
    col_text = None
    for c in df.columns:
        if "image" in str(c).lower(): col_img = c
        if "number" in str(c).lower() or "plate" in str(c).lower(): col_text = c
        
    if not col_img or not col_text:
        # Fallback to index 0 and 1
        col_img = df.columns[0]
        col_text = df.columns[1]
        
    gt_map = {}
    for _, row in df.iterrows():
        img_name = str(row[col_img]).strip()
        plate_text = str(row[col_text]).strip().upper().replace(" ", "")
        
        # Ensure image extension
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_name += '.jpg'  # Most Zenodo images are JPGs
            
        gt_map[img_name] = plate_text
        
    img_dir = None
    for d in INPUT_DIR.rglob("*"):
        if d.is_dir() and list(d.glob("*.jpg")):
            img_dir = d
            break
            
    if not img_dir:
        raise FileNotFoundError("Could not find images directory.")
        
    return gt_map, img_dir

def run_benchmark():
    gt_map, img_dir = load_ground_truth()
    
    print(f"Loaded {len(gt_map)} ground truth annotations.")
    
    # Initialize models
    detector = YOLO("anpr/models/best.pt")
    reader = easyocr.Reader(['en'], gpu=True)
    scorer = PlateQualityScorer()
    
    metrics = {
        "images_evaluated": 0,
        "plate_detected": 0,
        "ocr_non_empty": 0,
        "valid_indian_format": 0,
        "exact_raw_accuracy": 0,
        "correct_confirmed": 0,
        "false_confirmed": 0,
        "safe_unknown": 0,
        "incorrect_unconfirmed": 0,
        "total_levenshtein": 0,
        "total_chars_expected": 0
    }
    
    failures = []
    
    for img_name, gt_text in tqdm(gt_map.items(), desc="Evaluating ANPR Pipeline"):
        img_path = img_dir / img_name
        if not img_path.exists():
            continue
            
        metrics["images_evaluated"] += 1
        
        img = cv2.imread(str(img_path))
        if img is None: continue
        
        # 1. Detection
        results = detector(img, verbose=False)
        det_conf = 0.0
        plate_crop = None
        
        if len(results) > 0 and len(results[0].boxes) > 0:
            best_box = max(results[0].boxes, key=lambda b: b.conf[0].item())
            det_conf = best_box.conf[0].item()
            x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
            plate_crop = img[max(0, y1):min(img.shape[0], y2), max(0, x1):min(img.shape[1], x2)]
            
        if plate_crop is None or plate_crop.size == 0:
            # Nothing detected
            metrics["safe_unknown"] += 1
            metrics["total_chars_expected"] += len(gt_text)
            metrics["total_levenshtein"] += len(gt_text)
            failures.append((gt_text, "UNKNOWN", "SAFE-UNKNOWN"))
            continue
            
        metrics["plate_detected"] += 1
        
        # 2. Quality Score
        quality = PlateQualityScorer.compute(plate_crop)
        
        # 3. OCR
        ocr_results = reader.readtext(plate_crop)
        raw_text = ""
        ocr_conf = 0.0
        
        if ocr_results:
            # Sort by area or confidence, or take highest conf
            best_ocr = max(ocr_results, key=lambda x: x[2])
            raw_text = best_ocr[1].upper().replace(" ", "")
            ocr_conf = best_ocr[2]
            
        if not raw_text:
            metrics["safe_unknown"] += 1
            metrics["total_chars_expected"] += len(gt_text)
            metrics["total_levenshtein"] += len(gt_text)
            failures.append((gt_text, "UNKNOWN", "SAFE-UNKNOWN"))
            continue
            
        metrics["ocr_non_empty"] += 1
        
        # 4. Format Validation
        fmt_result = PlateFormatValidator.validate(raw_text)
        normalized_text = fmt_result.normalized_text or raw_text
        
        if fmt_result.format_score >= 0.8:
            metrics["valid_indian_format"] += 1
            
        # Character Error Rate component
        lev_dist = Levenshtein.distance(normalized_text, gt_text)
        metrics["total_chars_expected"] += len(gt_text)
        metrics["total_levenshtein"] += lev_dist
        
        if normalized_text == gt_text:
            metrics["exact_raw_accuracy"] += 1
            
        # 5. Fuzzy Temporal Voting
        # The user requested to keep production logic unchanged.
        # This voter will require 3 observations to hit CONFIRMED.
        voter = FuzzyTemporalVoter()
        voter.add(normalized_text, ocr_conf, det_conf, quality, 0)
        decision = voter.decide()
        
        # 6. Evaluation Logic
        status_label = ""
        pred_text = decision.text if decision.status == "CONFIRMED" else normalized_text
        
        if decision.status == "CONFIRMED":
            if decision.text == gt_text:
                metrics["correct_confirmed"] += 1
                status_label = "CORRECT"
            else:
                metrics["false_confirmed"] += 1
                status_label = "FALSE-CONFIRMED"
        else:
            if normalized_text == gt_text:
                status_label = "CORRECT-UNCONFIRMED"
            else:
                metrics["incorrect_unconfirmed"] += 1
                status_label = "INCORRECT"
                
        # Record everything not perfectly correct/confirmed for the log
        if status_label != "CORRECT":
            failures.append((gt_text, pred_text, status_label))
            
    # Compute final rates
    N = metrics["images_evaluated"]
    if N == 0:
        print("No images evaluated.")
        return
        
    print("\n=== PHASE 4K-A METRICS ===")
    print(f"Images evaluated           : {N}")
    print(f"Plate detected             : {metrics['plate_detected']} ({metrics['plate_detected']/N*100:.1f}%)")
    print(f"OCR non-empty rate         : {metrics['ocr_non_empty']} ({metrics['ocr_non_empty']/N*100:.1f}%)")
    
    cer = metrics["total_levenshtein"] / metrics["total_chars_expected"] if metrics["total_chars_expected"] > 0 else 1.0
    print(f"Character Error Rate (CER) : {cer*100:.2f}%")
    print(f"Exact full-plate accuracy  : {metrics['exact_raw_accuracy']} ({metrics['exact_raw_accuracy']/N*100:.1f}%)")
    print(f"Valid Indian-format rate   : {metrics['valid_indian_format']} ({metrics['valid_indian_format']/N*100:.1f}%)")
    
    print("\n--- SAFETY METRICS ---")
    print(f"Correct CONFIRMED rate     : {metrics['correct_confirmed']} ({metrics['correct_confirmed']/N*100:.1f}%)")
    print(f"False CONFIRMED Plate Rate : {metrics['false_confirmed']} ({metrics['false_confirmed']/N*100:.1f}%)")
    
    unknown_rate = metrics["safe_unknown"] / N
    print(f"SAFE UNKNOWN rate          : {metrics['safe_unknown']} ({unknown_rate*100:.1f}%)")
    print(f"Incorrect but unconfirmed  : {metrics['incorrect_unconfirmed']} ({metrics['incorrect_unconfirmed']/N*100:.1f}%)")
    
    print("\n=== EXAMPLES OF FAILURES ===")
    print(f"{'Ground Truth':<15} {'Prediction':<15} {'Status'}")
    print("-" * 50)
    for i, (gt, pred, status) in enumerate(failures[:20]):
        print(f"{gt:<15} {pred:<15} {status}")
        
    print("\nBenchmark complete. Dataset metadata:")
    print("Dataset: Number Plate Number Identification Dataset")
    print("Source: Zenodo DOI 10.5281/zenodo.13954136")
    print("Location: Bapatla, Andhra Pradesh, India")
    print("Ground truth: Dataset-provided Excel plate mappings")

if __name__ == "__main__":
    run_benchmark()
