import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import easyocr
import pandas as pd
import math
from tqdm import tqdm

from anpr.plate_quality import PlateQualityScorer
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

def create_contact_sheet(images, labels, output_path, cols=5):
    if not images: return
    rows = math.ceil(len(images) / cols)
    
    w, h = 300, 200
    resized = []
    for img, lbl in zip(images, labels):
        if img is None or img.size == 0: continue
        r = cv2.resize(img, (w, h - 50))
        canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
        canvas[:h-50, :, :] = r
        
        # Split label by newline if any
        lines = lbl.split('\n')
        for i, line in enumerate(lines):
            cv2.putText(canvas, line, (10, h - 35 + i*15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)
        resized.append(canvas)
        
    while len(resized) < rows * cols:
        resized.append(np.ones((h, w, 3), dtype=np.uint8) * 255)
        
    grid = []
    for r in range(rows):
        row_imgs = resized[r*cols:(r+1)*cols]
        grid.append(np.hstack(row_imgs))
    
    sheet = np.vstack(grid)
    cv2.imwrite(str(output_path), sheet)

def main():
    gt_map = load_ground_truth()
    detector = YOLO("anpr/models/best.pt")
    reader = easyocr.Reader(['en'], gpu=True)
    
    missed_det = []
    missed_lbl = []
    
    correct_det = []
    correct_lbl = []
    
    hq_ocr_fail = []
    hq_ocr_fail_lbl = []
    
    exact_ocr = []
    exact_ocr_lbl = []
    
    for img_path_str, gt_text in tqdm(gt_map.items(), desc="Auditing"):
        if len(missed_det) >= 50 and len(correct_det) >= 50 and len(hq_ocr_fail) >= 50 and len(exact_ocr) >= 50:
            break
            
        img = cv2.imread(img_path_str)
        if img is None: continue
        
        orig_img = img.copy()
        results = detector(img, verbose=False)
        
        if len(results) == 0 or len(results[0].boxes) == 0:
            if len(missed_det) < 50:
                missed_det.append(orig_img)
                missed_lbl.append(f"GT: {gt_text}\nDetection Failed")
            continue
            
        best_box = max(results[0].boxes, key=lambda b: b.conf[0].item())
        x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
        plate_crop = orig_img[max(0, y1):min(orig_img.shape[0], y2), max(0, x1):min(orig_img.shape[1], x2)]
        
        # Draw box for correct_det
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if len(correct_det) < 50:
            correct_det.append(img)
            correct_lbl.append(f"GT: {gt_text}\nDetected")
            
        q_score = PlateQualityScorer.compute(plate_crop)
        ocr_results = reader.readtext(plate_crop)
        
        raw_text = ""
        if ocr_results:
            best_ocr = max(ocr_results, key=lambda x: x[2])
            raw_text = best_ocr[1].upper().replace(" ", "")
            
        fmt = PlateFormatValidator.validate(raw_text)
        pred = fmt.normalized_text or raw_text
        
        if pred == gt_text:
            if len(exact_ocr) < 50:
                exact_ocr.append(plate_crop)
                exact_ocr_lbl.append(f"GT: {gt_text}\nExact Match")
        else:
            if q_score >= 0.70 and len(hq_ocr_fail) < 50:
                hq_ocr_fail.append(plate_crop)
                hq_ocr_fail_lbl.append(f"GT: {gt_text}\nPred: {pred}\nQ: {q_score:.2f}")

    OUT_DIR = Path("output/phase_4k_a3")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    create_contact_sheet(missed_det, missed_lbl, OUT_DIR / "missed_detections.jpg")
    create_contact_sheet(correct_det, correct_lbl, OUT_DIR / "correct_detections.jpg")
    create_contact_sheet(hq_ocr_fail, hq_ocr_fail_lbl, OUT_DIR / "hq_ocr_failures.jpg")
    create_contact_sheet(exact_ocr, exact_ocr_lbl, OUT_DIR / "exact_ocr_successes.jpg")
    
    print("Contact sheets generated in output/phase_4k_a3/")

if __name__ == "__main__":
    main()
