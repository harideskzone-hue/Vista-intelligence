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
            print(f"[{self.name}] Successfully loaded weights.")
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
        preds = np.argmax(preds, axis=0)
        res = ""
        pre_c = -1
        blank_idx = len(self.chars) - 1
        for c in preds:
            if c != pre_c:
                if c != blank_idx:
                    res += self.chars[c]
            pre_c = c
        return res

    def predict(self, img):
        if self.model is None:
            return ""
        # resize to 94x24
        img = cv2.resize(img, (94, 24))
        if "Candidate 2" in self.name:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = np.expand_dims(img, axis=-1)
        # normalize
        img = img.astype('float32')
        img -= 127.5
        img *= 0.0078125
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            preds = self.model(img)
        return self.decode(preds)

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

    CHARS_CAND2 = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    cand2 = LPRNetCandidate(
        name="Candidate 2 (muskaarora4446)",
        repo_path="/Users/hariharans/.gemini/antigravity-ide/brain/eee8ce25-20f1-4bc8-ab9e-d6c00f8b08ad/scratch/LPR_Candidates/LPRnet",
        weight_path="/Users/hariharans/.gemini/antigravity-ide/brain/eee8ce25-20f1-4bc8-ab9e-d6c00f8b08ad/scratch/LPR_Candidates/LPRnet/weights/Final_LPRNet_model.pth",
        chars=CHARS_CAND2,
        lpr_max_len=18
    )

    candidates = [cand1, cand2]

    metrics = {
        c.name: {
            "eval_count": 0,
            "exact_match": 0,
            "norm_exact_match": 0,
            "substitutions": 0,
            "insertions": 0,
            "deletions": 0,
            "empty": 0,
            "latencies": [],
            "inference_failures": 0,
            "q_high": {"eval": 0, "correct": 0},
            "q_medium": {"eval": 0, "correct": 0},
            "q_low": {"eval": 0, "correct": 0}
        } for c in candidates
    }
    
    from anpr.plate_quality import PlateQualityScorer
    scorer = PlateQualityScorer()

    print("Evaluating crops...")
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
        if score > 0.8: bucket = "high"
        elif score > 0.5: bucket = "medium"
        else: bucket = "low"

        for c in candidates:
            if c.model is None:
                continue
                
            metrics[c.name]["eval_count"] += 1
            metrics[c.name][f"q_{bucket}"]['eval'] += 1
            t0 = time.time()
            try:
                pred = c.predict(plate_crop)
                metrics[c.name]["latencies"].append(time.time() - t0)
                
                if not pred:
                    metrics[c.name]["empty"] += 1
                else:
                    norm_pred = "".join(ch for ch in pred if ch.isalnum())
                    norm_gt = "".join(ch for ch in gt_text if ch.isalnum())
                    
                    if pred == gt_text:
                        metrics[c.name]["exact_match"] += 1
                    if norm_pred == norm_gt:
                        metrics[c.name]["norm_exact_match"] += 1
                        metrics[c.name][f"q_{bucket}"]['correct'] += 1
                        
                    dist = levenshtein(norm_gt, norm_pred)
                    diff = len(norm_pred) - len(norm_gt)
                    if diff > 0:
                        metrics[c.name]["insertions"] += diff
                        metrics[c.name]["substitutions"] += max(0, dist - diff)
                    elif diff < 0:
                        metrics[c.name]["deletions"] += abs(diff)
                        metrics[c.name]["substitutions"] += max(0, dist - abs(diff))
                    else:
                        metrics[c.name]["substitutions"] += dist

            except Exception as e:
                import traceback
                traceback.print_exc()
                metrics[c.name]["inference_failures"] += 1
                
    # Report
    print("\\n\\n### Benchmark Results")
    print("| Metric | Candidate 1 | Candidate 2 |")
    print("| --- | ---: | ---: |")
    for metric_name, display_name in [
        ("eval_count", "Evaluated crops"),
        ("exact_match", "Exact accuracy"),
        ("norm_exact_match", "Normalized exact"),
        ("empty", "Empty output"),
        ("substitutions", "Substitutions"),
        ("insertions", "Insertions"),
        ("deletions", "Deletions"),
        ("inference_failures", "Inference failures")
    ]:
        v1 = metrics[cand1.name][metric_name]
        v2 = metrics[cand2.name][metric_name]
        
        if metric_name in ["exact_match", "norm_exact_match", "empty"]:
            p1 = (v1 / metrics[cand1.name]["eval_count"] * 100) if metrics[cand1.name]["eval_count"] > 0 else 0
            p2 = (v2 / metrics[cand2.name]["eval_count"] * 100) if metrics[cand2.name]["eval_count"] > 0 else 0
            print(f"| {display_name} | {v1} ({p1:.2f}%) | {v2} ({p2:.2f}%) |")
        else:
            print(f"| {display_name} | {v1} | {v2} |")
            
    # Latencies
    mean1 = np.mean(metrics[cand1.name]["latencies"]) * 1000 if metrics[cand1.name]["latencies"] else 0
    mean2 = np.mean(metrics[cand2.name]["latencies"]) * 1000 if metrics[cand2.name]["latencies"] else 0
    med1 = np.median(metrics[cand1.name]["latencies"]) * 1000 if metrics[cand1.name]["latencies"] else 0
    med2 = np.median(metrics[cand2.name]["latencies"]) * 1000 if metrics[cand2.name]["latencies"] else 0
    
    print(f"| Mean latency/image | {mean1:.1f}ms | {mean2:.1f}ms |")
    print(f"| Median latency/image | {med1:.1f}ms | {med2:.1f}ms |")
    
    size1 = os.path.getsize(cand1.weight_path) / (1024*1024) if os.path.exists(cand1.weight_path) else 0
    size2 = os.path.getsize(cand2.weight_path) / (1024*1024) if os.path.exists(cand2.weight_path) else 0
    print(f"| Model size | {size1:.1f}MB | {size2:.1f}MB |")
    
    print("\\n### Quality Buckets (Normalized Exact Accuracy)")
    for q in ["high", "medium", "low"]:
        c1_q = metrics[cand1.name][f"q_{q}"]
        c2_q = metrics[cand2.name][f"q_{q}"]
        c1_acc = (c1_q["correct"] / c1_q["eval"] * 100) if c1_q["eval"] > 0 else 0
        c2_acc = (c2_q["correct"] / c2_q["eval"] * 100) if c2_q["eval"] > 0 else 0
        print(f"| {q.upper()} | {c1_acc:.2f}% | {c2_acc:.2f}% |")

if __name__ == '__main__':
    main()
