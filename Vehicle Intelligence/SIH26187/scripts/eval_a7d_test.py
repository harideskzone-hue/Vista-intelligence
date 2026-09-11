import os
import cv2
import csv
import json
import time
import numpy as np
from pathlib import Path
import Levenshtein
import torch

import fast_plate_ocr

def load_test_set():
    # Load the 197 test images and their ground truth
    # The prepare_lpr_dataset.py generated test_annotations.csv with image_path, plate_text
    test_csv = "data/fast_plate_ocr/test_annotations.csv"
    items = []
    with open(test_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            items.append({
                "image_path": row["image_path"],
                "norm_label": row["plate_text"]
            })
    return items

import easyocr
import pytesseract

def get_quality(img_path, gt, reader):
    img = cv2.imread(img_path)
    # easyocr
    res_easy = reader.readtext(img, detail=0)
    res_easy = "".join(res_easy)
    res_easy = "".join([c for c in res_easy if c.isalnum()]).upper()
    
    # tesseract
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    res_tess = pytesseract.image_to_string(gray, config='--psm 7')
    res_tess = "".join([c for c in res_tess if c.isalnum()]).upper()
    
    # fuzz
    dist_easy = Levenshtein.distance(gt, res_easy)
    dist_tess = Levenshtein.distance(gt, res_tess)
    
    fuzz_easy = max(0.0, 1.0 - (dist_easy / max(len(gt), 1)))
    fuzz_tess = max(0.0, 1.0 - (dist_tess / max(len(gt), 1)))
    
    best_fuzz = max(fuzz_easy, fuzz_tess)
    if best_fuzz == 1.0: return "HIGH"
    elif best_fuzz >= 0.8: return "MEDIUM"
    return "LOW"

def run_evaluation():
    test_items = load_test_set()
    reader = easyocr.Reader(['en'], gpu=torch.backends.mps.is_available()) if torch.backends.mps.is_available() else easyocr.Reader(['en'])

    
    from fast_plate_ocr import LicensePlateRecognizer
    
    zero_shot_model = LicensePlateRecognizer(
        onnx_model_path="data/fast_plate_ocr/models/cct_xs_v2_global.onnx",
        plate_config_path="data/fast_plate_ocr/models/cct_xs_v2_global_plate_config.yaml"
    )
    
    # Find the best.onnx inside fine_tuned
    import glob
    best_weights = glob.glob("data/fast_plate_ocr/models/fine_tuned/*/best.onnx")[0]
    fine_tuned_model = LicensePlateRecognizer(
        onnx_model_path=best_weights,
        plate_config_path="data/fast_plate_ocr/models/cct_xs_v2_global_plate_config.yaml"
    )
    
    
    results = {
        "Zero-Shot": {"exact": 0, "norm_exact": 0, "total": len(test_items), "dist": 0, "chars": 0, "time": 0, "empty": 0, "high_exact": 0, "high_total": 0, "med_exact": 0, "med_total": 0, "low_exact": 0, "low_total": 0},
        "Fine-Tuned": {"exact": 0, "norm_exact": 0, "total": len(test_items), "dist": 0, "chars": 0, "time": 0, "empty": 0, "high_exact": 0, "high_total": 0, "med_exact": 0, "med_total": 0, "low_exact": 0, "low_total": 0}
    }
    
    for item in test_items:
        # The image path in CSV is relative like 'images/filename'
        # But we run from project root, so we prepend 'data/fast_plate_ocr'
        img_path = os.path.join("data/fast_plate_ocr", item["image_path"])
        img = cv2.imread(img_path)
        gt = item["norm_label"]
        quality = get_quality(img_path, gt, reader)
        
        # Evaluate Zero-Shot
        t0 = time.time()
        preds = zero_shot_model.run(img)
        res_z = getattr(preds[0], 'plate', "") if preds else ""
        t1 = time.time()
        res_z = res_z.replace("_", "")
        results["Zero-Shot"]["time"] += (t1 - t0)
        
        if res_z == gt:
            results["Zero-Shot"]["exact"] += 1
            if quality == "HIGH": results["Zero-Shot"]["high_exact"] += 1
            if quality == "MEDIUM": results["Zero-Shot"]["med_exact"] += 1
            if quality == "LOW": results["Zero-Shot"]["low_exact"] += 1
        
        results["Zero-Shot"]["dist"] += Levenshtein.distance(gt, res_z)
        results["Zero-Shot"]["chars"] += len(gt)
        if len(res_z) == 0:
            results["Zero-Shot"]["empty"] += 1
            
        if quality == "HIGH": results["Zero-Shot"]["high_total"] += 1
        if quality == "MEDIUM": results["Zero-Shot"]["med_total"] += 1
        if quality == "LOW": results["Zero-Shot"]["low_total"] += 1
            
        # Evaluate Fine-Tuned
        t0 = time.time()
        preds_f = fine_tuned_model.run(img)
        res_f = getattr(preds_f[0], 'plate', "") if preds_f else ""
        t1 = time.time()
        res_f = res_f.replace("_", "")
        results["Fine-Tuned"]["time"] += (t1 - t0)
        
        if res_f == gt:
            results["Fine-Tuned"]["exact"] += 1
            if quality == "HIGH": results["Fine-Tuned"]["high_exact"] += 1
            if quality == "MEDIUM": results["Fine-Tuned"]["med_exact"] += 1
            if quality == "LOW": results["Fine-Tuned"]["low_exact"] += 1
            
        results["Fine-Tuned"]["dist"] += Levenshtein.distance(gt, res_f)
        results["Fine-Tuned"]["chars"] += len(gt)
        if len(res_f) == 0:
            results["Fine-Tuned"]["empty"] += 1
            
        if quality == "HIGH": results["Fine-Tuned"]["high_total"] += 1
        if quality == "MEDIUM": results["Fine-Tuned"]["med_total"] += 1
        if quality == "LOW": results["Fine-Tuned"]["low_total"] += 1

    # Print results
    print(f"Total test images: {len(test_items)}")
    for model_name, metrics in results.items():
        total = metrics["total"]
        print(f"\nModel: {model_name}")
        print(f"Exact Accuracy: {metrics['exact']} / {total} ({(metrics['exact']/total)*100:.2f}%)")
        print(f"CER: {(metrics['dist']/metrics['chars'])*100:.2f}%")
        print(f"Empty Predictions: {metrics['empty']}")
        avg_latency = metrics['time'] / total * 1000
        print(f"Avg Latency: {avg_latency:.2f} ms")
        
        h_t = metrics['high_total']
        if h_t > 0: print(f"HIGH Quality Exact: {metrics['high_exact']} / {h_t} ({(metrics['high_exact']/h_t)*100:.2f}%)")
        m_t = metrics['med_total']
        if m_t > 0: print(f"MEDIUM Quality Exact: {metrics['med_exact']} / {m_t} ({(metrics['med_exact']/m_t)*100:.2f}%)")
        l_t = metrics['low_total']
        if l_t > 0: print(f"LOW Quality Exact: {metrics['low_exact']} / {l_t} ({(metrics['low_exact']/l_t)*100:.2f}%)")

if __name__ == "__main__":
    run_evaluation()
