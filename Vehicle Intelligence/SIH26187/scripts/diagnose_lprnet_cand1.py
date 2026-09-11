import sys
import os
import time
import json
import numpy as np
import pandas as pd
import torch
import cv2
from pathlib import Path
from ultralytics import YOLO

def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

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

class LPRNetCandidate:
    def __init__(self, name, repo_path, weight_path, chars, lpr_max_len, dropout_rate=0.5):
        self.name = name
        self.repo_path = repo_path
        self.weight_path = weight_path
        self.chars = chars
        self.lpr_max_len = lpr_max_len
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.dropout_rate = dropout_rate
        self.model = None
        self.load_model()
        
    def load_model(self):
        sys.path.insert(0, self.repo_path)
        try:
            from model.LPRNet import build_lprnet
            self.model = build_lprnet(lpr_max_len=self.lpr_max_len, phase=False, class_num=len(self.chars), dropout_rate=self.dropout_rate)
            self.model.load_state_dict(torch.load(self.weight_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"[{self.name}] Failed to load: {e}")
            self.model = None
        finally:
            sys.path.pop(0)
            if "model.LPRNet" in sys.modules:
                del sys.modules["model.LPRNet"]
            if "model" in sys.modules:
                del sys.modules["model"]
            
    def decode(self, preds):
        preds = preds.cpu().detach().numpy()
        preds = preds[0] # batch size 1
        
        # Calculate confidences (softmax max over character classes)
        # Note: LPRNet raw output is logits, need to apply softmax
        probs = np.exp(preds) / np.sum(np.exp(preds), axis=0, keepdims=True)
        max_probs = np.max(probs, axis=0)
        max_idx = np.argmax(preds, axis=0)
        
        res = ""
        pre_c = -1
        blank_idx = len(self.chars) - 1
        
        char_confidences = []
        for i, c in enumerate(max_idx):
            if c != pre_c:
                if c != blank_idx:
                    res += self.chars[c]
                    char_confidences.append(max_probs[i])
            pre_c = c
            
        avg_conf = float(np.mean(char_confidences)) if char_confidences else 0.0
        return res, avg_conf

    def predict(self, img):
        if self.model is None:
            return "", 0.0
        # resize to 94x24
        img = cv2.resize(img, (94, 24))
        # normalize
        img = img.astype('float32')
        img -= 127.5
        img *= 0.0078125
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            preds = self.model(img)
        return self.decode(preds)

def get_error_type(gt, pred):
    if len(pred) == 0:
        return "Empty output"
    elif len(gt) > len(pred) + 2:
        return "Severe Deletion (Truncation)"
    elif len(pred) > len(gt) + 2:
        return "Severe Insertion (Over-prediction)"
    else:
        # Check if it's mostly substitutions
        diff = len(pred) - len(gt)
        dist = levenshtein(gt, pred)
        if abs(diff) <= 2 and dist <= 3:
            return "Minor Substitution/Confusion"
        elif abs(diff) <= 2 and dist > 3:
            return "Major Substitution (Garbage)"
        else:
            return "Mixed Error"

def main():
    gt_map = load_ground_truth()
    detector = YOLO("anpr/models/best.pt")
    
    CHARS_CAND1 = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "-"]
    cand1 = LPRNetCandidate(
        name="Candidate 1 (Indian_LPR)",
        repo_path="/Users/hariharans/.gemini/antigravity-ide/brain/eee8ce25-20f1-4bc8-ab9e-d6c00f8b08ad/scratch/LPR_Candidates/Indian_LPR/src/License_Plate_Recognition",
        weight_path="/Users/hariharans/.gemini/antigravity-ide/brain/eee8ce25-20f1-4bc8-ab9e-d6c00f8b08ad/scratch/LPR_Candidates/Indian_LPR/weights/best_lprnet.pth",
        chars=CHARS_CAND1,
        lpr_max_len=18
    )

    from anpr.plate_quality import PlateQualityScorer
    scorer = PlateQualityScorer()

    results = []
    
    from tqdm import tqdm
    for img_path_str, gt_text in tqdm(gt_map.items()):
        img = cv2.imread(img_path_str)
        if img is None:
            continue
            
        res = detector(img, verbose=False)[0]
        boxes = res.boxes
        if len(boxes) == 0:
            continue
            
        best_box = max(boxes, key=lambda b: b.conf[0].item())
        x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
        plate_crop = img[y1:y2, x1:x2]
        if plate_crop.size == 0:
            continue
            
        score = scorer.compute(plate_crop)
        if score > 0.8: bucket = "HIGH"
        elif score > 0.5: bucket = "MEDIUM"
        else: bucket = "LOW"

        if cand1.model is None:
            continue
            
        try:
            raw_pred, conf = cand1.predict(plate_crop)
            norm_pred = "".join(ch for ch in raw_pred if ch.isalnum())
            norm_gt = "".join(ch for ch in gt_text if ch.isalnum())
            
            is_correct = (norm_pred == norm_gt)
            
            error_type = "Correct" if is_correct else get_error_type(norm_gt, norm_pred)
            
            results.append({
                "img_path": img_path_str,
                "gt": norm_gt,
                "pred": norm_pred,
                "conf": conf,
                "quality_score": score,
                "bucket": bucket,
                "is_correct": is_correct,
                "error_type": error_type,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2
            })
                
        except Exception as e:
            pass

    df = pd.DataFrame(results)
    
    out_dir = Path("output/phase_4k_a5_diagnosis")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Bucket breakdown
    print("\\n### Overall Breakdown")
    print(f"Total evaluated: {len(df)}")
    print(f"Correct: {df['is_correct'].sum()} ({df['is_correct'].mean()*100:.2f}%)")
    print(f"Incorrect: {(~df['is_correct']).sum()}")
    
    print("\\n### Quality Bucket Accuracy")
    for b in ["HIGH", "MEDIUM", "LOW"]:
        b_df = df[df["bucket"] == b]
        if len(b_df) > 0:
            acc = b_df['is_correct'].mean() * 100
            print(f"{b}: {len(b_df)} crops, {acc:.2f}% accuracy")
            
    print("\\n### Error Typology (out of incorrect crops)")
    err_df = df[~df["is_correct"]]
    counts = err_df["error_type"].value_counts()
    for err_type, count in counts.items():
        print(f"{err_type}: {count} ({count/len(err_df)*100:.2f}%)")
        
    print("\\n### Sample Output (High Quality but Wrong)")
    hq_wrong = df[(df["bucket"] == "HIGH") & (~df["is_correct"])]
    for _, row in hq_wrong.head(15).iterrows():
        print(f"GT: {row['gt']:<15} Pred: {row['pred']:<15} Conf: {row['conf']:.3f} Type: {row['error_type']}")

    print("\\n### Sample Output (Low Quality and Wrong)")
    lq_wrong = df[(df["bucket"] == "LOW") & (~df["is_correct"])]
    for _, row in lq_wrong.head(10).iterrows():
        print(f"GT: {row['gt']:<15} Pred: {row['pred']:<15} Conf: {row['conf']:.3f} Type: {row['error_type']}")

    # Save random 50 correct
    correct_df = df[df["is_correct"]].sample(min(50, len(df[df["is_correct"]])), random_state=42)
    correct_df.to_csv(out_dir / "correct_50.csv", index=False)
    
    # Save random 100 wrong
    wrong_df = err_df.sample(min(100, len(err_df)), random_state=42)
    wrong_df.to_csv(out_dir / "wrong_100.csv", index=False)

if __name__ == '__main__':
    main()
