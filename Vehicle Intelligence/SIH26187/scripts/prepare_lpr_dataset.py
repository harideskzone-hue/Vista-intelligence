import os
import cv2
import csv
import json
import random
import glob
import pandas as pd
from collections import defaultdict, Counter
from ultralytics import YOLO
import sys

# Paths
INPUT_DIR = "input/indian_anpr_zenodo"
INPUT_IMG_DIR = os.path.join(INPUT_DIR, "images")
EXCEL_PATH = os.path.join(INPUT_DIR, "number_plate.xlsx")
OUTPUT_DIR = "data/fast_plate_ocr"
OUTPUT_IMG_DIR = os.path.join(OUTPUT_DIR, "images")

os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)

# Settings
ALPHABET = set("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_")
MAX_LENGTH = 10
SEED = 42
random.seed(SEED)

def normalize_label(label):
    norm = "".join(ch for ch in label if ch.isalnum()).upper()
    return norm

def validate_label(label):
    if not label:
        return False, "Empty label"
    if len(label) > MAX_LENGTH:
        return False, f"Length > {MAX_LENGTH}"
    for ch in label:
        if ch not in ALPHABET:
            return False, f"Invalid character: {ch}"
    return True, "Valid"

def main():
    print("Loading YOLO detector...")
    detector = YOLO("anpr/models/best.pt")

    df = pd.read_excel(EXCEL_PATH)
    col_img = df.columns[0]
    col_gt = df.columns[1]
    
    img_paths = list(glob.glob(os.path.join(INPUT_IMG_DIR, "*")))
    norm_to_file = {}
    for f in img_paths:
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            norm_name = os.path.splitext(os.path.basename(f))[0].lower().replace(" ", "").replace("(", "").replace(")", "").replace("-", "").replace("_", "")
            norm_to_file[norm_name] = f
    
    # Store data
    # We will group by normalized label to prevent leakage.
    # groups[label] = [ { 'img_path', 'crop_filename', 'original_label', ... } ]
    groups = defaultdict(list)
    invalid_records = []
    
    print(f"Processing annotations...")
    processed_count = 0
    crop_count = 0
    
    for idx, row in df.iterrows():
        img_name = str(row[col_img])
        gt_text = str(row[col_gt])
        if pd.isna(row[col_gt]) or not gt_text.strip() or gt_text.lower() == 'nan':
            continue
            
        img_stem = os.path.splitext(img_name)[0]
        norm_name = img_stem.lower().replace(" ", "").replace("(", "").replace(")", "").replace("-", "").replace("_", "")
        if norm_name not in norm_to_file:
            continue
            
        img_path = norm_to_file[norm_name]
        basename = os.path.basename(img_path)
        name, _ = os.path.splitext(basename)
        
        raw_label = gt_text.strip()
        norm_label = normalize_label(raw_label)
        
        valid, reason = validate_label(norm_label)
        if not valid:
            invalid_records.append({"file": basename, "raw": raw_label, "norm": norm_label, "reason": reason})
            continue

        processed_count += 1
        
        # YOLO Crop
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        res = detector(img, verbose=False)[0]
        if len(res.boxes) == 0:
            continue # No plate detected
            
        best_box = max(res.boxes, key=lambda b: b.conf[0].item())
        x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
        crop = img[y1:y2, x1:x2]
        
        if crop.size == 0:
            continue
            
        crop_count += 1
        crop_filename = f"{name}_crop.jpg"
        crop_path = os.path.join(OUTPUT_IMG_DIR, crop_filename)
        cv2.imwrite(crop_path, crop)
        
        groups[norm_label].append({
            "original_img": basename,
            "crop_filename": crop_filename,
            "raw_label": raw_label,
            "norm_label": norm_label
        })

    print(f"\nExtracted {crop_count} valid crops from {processed_count} valid annotations.")
    
    # Split preventing leakage (split by group)
    labels = list(groups.keys())
    random.shuffle(labels)
    
    train_split, val_split, test_split = [], [], []
    train_target = int(0.70 * crop_count)
    val_target = int(0.15 * crop_count)
    
    for label in labels:
        items = groups[label]
        if len(train_split) < train_target:
            train_split.extend(items)
        elif len(val_split) < val_target:
            val_split.extend(items)
        else:
            test_split.extend(items)
            
    # Output CSVs
    def write_csv(split_name, items):
        filepath = os.path.join(OUTPUT_DIR, f"{split_name}_annotations.csv")
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["image_path", "plate_text"])
            for item in items:
                rel_path = os.path.join("images", item["crop_filename"])
                writer.writerow([rel_path, item["norm_label"]])
        return filepath
        
    write_csv("train", train_split)
    write_csv("val", val_split)
    write_csv("test", test_split)
    
    # Audit Stats
    all_items = train_split + val_split + test_split
    lengths = [len(item["norm_label"]) for item in all_items]
    chars = [ch for item in all_items for ch in item["norm_label"]]
    
    audit = {
        "total_crops": len(all_items),
        "split_counts": {
            "train": len(train_split),
            "val": len(val_split),
            "test": len(test_split)
        },
        "split_percentages": {
            "train": round(len(train_split) / max(1, len(all_items)) * 100, 2),
            "val": round(len(val_split) / max(1, len(all_items)) * 100, 2),
            "test": round(len(test_split) / max(1, len(all_items)) * 100, 2)
        },
        "plate_lengths": dict(Counter(lengths)),
        "character_frequencies": dict(Counter(chars)),
        "invalid_excluded": len(invalid_records),
        "invalid_examples": invalid_records[:10]
    }
    
    with open(os.path.join(OUTPUT_DIR, "audit_report.json"), "w") as f:
        json.dump(audit, f, indent=2)
        
    print("\nDataset split complete. Audit report saved to", os.path.join(OUTPUT_DIR, "audit_report.json"))

if __name__ == "__main__":
    main()
