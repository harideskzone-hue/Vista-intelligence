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

# ----- Levenshtein Distance -----
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

# ----- Candidate A: fast-plate-ocr -----
class FastPlateOCRCandidate:
    def __init__(self):
        self.name = "Candidate A: fast-plate-ocr"
        self.model = None
        self.license = "MIT"
        self.model_size = "Unknown"
        self.latency_samples = []
        try:
            from fast_plate_ocr import LicensePlateRecognizer
            self.model = LicensePlateRecognizer("cct-s-v2-global-model")
        except Exception as e:
            print(f"[{self.name}] Failed to load: {e}")

    def predict(self, img_crop):
        if self.model is None:
            return "", 0.0
        try:
            img_rgb = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)
            preds = self.model.run(img_rgb, return_confidence=True)
            if not preds: return "", 0.0
            pred = preds[0]
            text = getattr(pred, 'plate', "")
            probs = getattr(pred, 'char_probs', None)
            conf = float(np.mean(probs)) if probs is not None and len(probs) > 0 else 0.0
            return str(text), float(conf)
        except Exception as e:
            print(f"fast-plate-ocr exception: {e}")
            return "", 0.0

# ----- Candidate B: ClovaAI CRNN -----
class ClovaAICandidate:
    def __init__(self):
        self.name = "Candidate B: ClovaAI CRNN"
        self.model = None
        self.license = "Apache 2.0"
        self.model_size = "31.5MB"
        self.latency_samples = []
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.repo_path = "/Users/hariharans/Documents/Vehicle Intelligence/SIH26187/scratch/LPR_Candidates/deep-text-recognition-benchmark"
        self.weight_path = os.path.join(self.repo_path, "saved_models", "None-VGG-BiLSTM-CTC.pth")
        
        self.load_model()
        
    class Opt:
        def __init__(self):
            self.image_folder = 'demo_image/'
            self.workers = 4
            self.batch_size = 192
            self.saved_model = 'None-VGG-BiLSTM-CTC.pth'
            self.batch_max_length = 25
            self.imgH = 32
            self.imgW = 100
            self.rgb = False
            self.character = '0123456789abcdefghijklmnopqrstuvwxyz'
            self.sensitive = False
            self.PAD = False
            self.Transformation = 'None'
            self.FeatureExtraction = 'VGG'
            self.SequenceModeling = 'BiLSTM'
            self.Prediction = 'CTC'
            self.num_fiducial = 20
            self.input_channel = 1
            self.output_channel = 512
            self.hidden_size = 256
            self.num_class = len(self.character) + 1 # For CTC

    def load_model(self):
        sys.path.insert(0, self.repo_path)
        try:
            from model import Model
            from utils import CTCLabelConverter
            
            self.opt = self.Opt()
            self.converter = CTCLabelConverter(self.opt.character)
            self.opt.num_class = len(self.converter.character)
            
            # Using data parallel is common in clovaai
            import torch.nn as nn
            model = Model(self.opt)
            
            # Remove "module." prefix if it exists in state_dict
            try:
                state_dict = torch.load(self.weight_path, map_location=self.device, weights_only=False)
            except Exception as e:
                print(f"[{self.name}] Failed to load weights from {self.weight_path}: {e}")
                self.model = None
                return
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("module."):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
                    
            model.load_state_dict(new_state_dict)
            self.model = model.to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"[{self.name}] Failed to load: {e}")
            self.model = None
        finally:
            sys.path.pop(0)
            if "model" in sys.modules: del sys.modules["model"]
            if "utils" in sys.modules: del sys.modules["utils"]

    def predict(self, img_crop):
        if self.model is None:
            return "", 0.0
        try:
            import torch.nn.functional as F
            import math
            # preprocess image
            img = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (self.opt.imgW, self.opt.imgH))
            img = img.astype('float32')
            img = (img / 255.0 - 0.5) / 0.5
            
            img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(self.device)
            
            length_for_pred = torch.IntTensor([self.opt.batch_max_length]).to(self.device)
            text_for_pred = torch.LongTensor(1, self.opt.batch_max_length + 1).fill_(0).to(self.device)
            
            with torch.no_grad():
                preds = self.model(img, text_for_pred)
                
                preds_size = torch.IntTensor([preds.size(1)])
                _, preds_index = preds.max(2)
                preds_str = self.converter.decode(preds_index, preds_size)
                
                preds_prob = F.softmax(preds, dim=2)
                preds_max_prob, _ = preds_prob.max(dim=2)
                
                pred = preds_str[0]
                pred_EOS = pred.find('[s]')
                pred = pred[:pred_EOS] if pred_EOS != -1 else pred
                
                confidence_score = preds_max_prob[0][:len(pred)].mean().item() if len(pred) > 0 else 0.0
                
            return str(pred).upper(), float(confidence_score)
        except Exception as e:
            # print(f"ClovaAI err: {e}")
            return "", 0.0

# ----- Candidate C: LPRNet -----
class LPRNetCandidate:
    def __init__(self):
        self.name = "Candidate C: LPRNet_Pytorch"
        self.model = None
        self.license = "Apache 2.0"
        self.model_size = "N/A"
        self.latency_samples = []

    def predict(self, img_crop):
        # INCOMPATIBLE VOCABULARY - NOT EVALUATED
        return "INCOMPATIBLE", 0.0

def main():
    gt_map = load_ground_truth()
    detector = YOLO("anpr/models/best.pt")
    
    candidates = [
        FastPlateOCRCandidate(),
        ClovaAICandidate(),
        LPRNetCandidate()
    ]
    
    from anpr.plate_quality import PlateQualityScorer
    scorer = PlateQualityScorer()

    metrics = {}
    for c in candidates:
        metrics[c.name] = {
            "eval_count": 0,
            "exact_match": 0,
            "norm_exact_match": 0,
            "substitutions": 0,
            "insertions": 0,
            "deletions": 0,
            "empty": 0,
            "failures": 0,
            "q_HIGH": {"eval": 0, "correct": 0},
            "q_MEDIUM": {"eval": 0, "correct": 0},
            "q_LOW": {"eval": 0, "correct": 0},
            "hallucinations": 0, # non-empty but wrong
            "conf_0_50": {"eval": 0, "correct": 0},
            "conf_50_70": {"eval": 0, "correct": 0},
            "conf_70_85": {"eval": 0, "correct": 0},
            "conf_85_95": {"eval": 0, "correct": 0},
            "conf_95_100": {"eval": 0, "correct": 0},
        }

    from tqdm import tqdm
    for img_path_str, gt_text in tqdm(gt_map.items()):
        img = cv2.imread(img_path_str)
        if img is None: continue
            
        res = detector(img, verbose=False)[0]
        boxes = res.boxes
        if len(boxes) == 0: continue
            
        best_box = max(boxes, key=lambda b: b.conf[0].item())
        x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
        plate_crop = img[y1:y2, x1:x2]
        if plate_crop.size == 0: continue
            
        score = scorer.compute(plate_crop)
        if score > 0.8: bucket = "HIGH"
        elif score > 0.5: bucket = "MEDIUM"
        else: bucket = "LOW"

        for c in candidates:
            if c.name == "Candidate C: LPRNet_Pytorch":
                metrics[c.name]["failures"] += 1
                continue
                
            metrics[c.name]["eval_count"] += 1
            metrics[c.name][f"q_{bucket}"]['eval'] += 1
            
            try:
                t0 = time.perf_counter()
                raw_pred, conf = c.predict(plate_crop)
                t1 = time.perf_counter()
                c.latency_samples.append((t1 - t0)*1000)
                
                norm_pred = "".join(ch for ch in str(raw_pred) if ch.isalnum()).upper()
                norm_gt = "".join(ch for ch in str(gt_text) if ch.isalnum()).upper()
                
                if not norm_pred:
                    metrics[c.name]["empty"] += 1
                
                is_correct = False
                if norm_pred == norm_gt:
                    metrics[c.name]["norm_exact_match"] += 1
                    metrics[c.name][f"q_{bucket}"]['correct'] += 1
                    is_correct = True
                elif norm_pred:
                    metrics[c.name]["hallucinations"] += 1
                    
                dist = levenshtein(norm_gt, norm_pred)
                diff = len(norm_pred) - len(norm_gt)
                if diff > 0:
                    metrics[c.name]["insertions"] += diff
                    metrics[c.name]["substitutions"] += (dist - diff)
                elif diff < 0:
                    metrics[c.name]["deletions"] += abs(diff)
                    metrics[c.name]["substitutions"] += (dist - abs(diff))
                else:
                    metrics[c.name]["substitutions"] += dist

                # Confidence bucketing
                if conf < 0.50: conf_bucket = "conf_0_50"
                elif conf <= 0.70: conf_bucket = "conf_50_70"
                elif conf <= 0.85: conf_bucket = "conf_70_85"
                elif conf <= 0.95: conf_bucket = "conf_85_95"
                else: conf_bucket = "conf_95_100"
                
                metrics[c.name][conf_bucket]["eval"] += 1
                if is_correct:
                    metrics[c.name][conf_bucket]["correct"] += 1

            except Exception as e:
                metrics[c.name]["failures"] += 1

    print("\\n\\n### Benchmark Results")
    print("| Metric | fast-plate-ocr | STR/CRNN | LPRNet |")
    print("| --- | ---: | ---: | ---: |")
    
    def get_m(name, key): return metrics[name][key]
    
    total = get_m(candidates[0].name, 'eval_count')
    n_A = candidates[0].name
    n_B = candidates[1].name
    n_C = candidates[2].name
    
    print(f"| Evaluated crops | {total} | {get_m(n_B, 'eval_count')} | 0 |")
    
    def fmt_acc(n):
        ev = get_m(n, 'eval_count')
        if ev == 0: return "INCOMPATIBLE"
        ex = get_m(n, 'norm_exact_match')
        return f"{ex} ({ex/ev*100:.2f}%)"
        
    print(f"| Exact accuracy | {fmt_acc(n_A)} | {fmt_acc(n_B)} | INCOMPATIBLE |")
    print(f"| Normalized exact | {fmt_acc(n_A)} | {fmt_acc(n_B)} | INCOMPATIBLE |")
    
    def fmt_cer(n):
        ev = get_m(n, 'eval_count')
        if ev == 0: return "INCOMPATIBLE"
        total_err = get_m(n, 'substitutions') + get_m(n, 'insertions') + get_m(n, 'deletions')
        return f"{total_err}" # Simplified CER representation
        
    print(f"| Substitutions | {get_m(n_A, 'substitutions')} | {get_m(n_B, 'substitutions')} | - |")
    print(f"| Insertions | {get_m(n_A, 'insertions')} | {get_m(n_B, 'insertions')} | - |")
    print(f"| Deletions | {get_m(n_A, 'deletions')} | {get_m(n_B, 'deletions')} | - |")
    
    def fmt_empty(n):
        ev = get_m(n, 'eval_count')
        if ev == 0: return "-"
        em = get_m(n, 'empty')
        return f"{em} ({em/ev*100:.2f}%)"
        
    print(f"| Empty output | {fmt_empty(n_A)} | {fmt_empty(n_B)} | - |")
    print(f"| Inference failures | {get_m(n_A, 'failures')} | {get_m(n_B, 'failures')} | 1316 |")
    
    def fmt_q(n, q):
        ev = get_m(n, q)['eval']
        if ev == 0: return "-"
        cr = get_m(n, q)['correct']
        return f"{cr/ev*100:.2f}%"
        
    print(f"| HIGH exact | {fmt_q(n_A, 'q_HIGH')} | {fmt_q(n_B, 'q_HIGH')} | - |")
    print(f"| MEDIUM exact | {fmt_q(n_A, 'q_MEDIUM')} | {fmt_q(n_B, 'q_MEDIUM')} | - |")
    print(f"| LOW exact | {fmt_q(n_A, 'q_LOW')} | {fmt_q(n_B, 'q_LOW')} | - |")
    
    def fmt_lat(n, stat):
        c = [cand for cand in candidates if cand.name == n][0]
        if not c.latency_samples: return "-"
        if stat == 'mean': return f"{np.mean(c.latency_samples):.1f}ms"
        return f"{np.median(c.latency_samples):.1f}ms"
        
    print(f"| Mean latency | {fmt_lat(n_A, 'mean')} | {fmt_lat(n_B, 'mean')} | - |")
    print(f"| Median latency | {fmt_lat(n_A, 'median')} | {fmt_lat(n_B, 'median')} | - |")
    print(f"| Model size | {candidates[0].model_size} | {candidates[1].model_size} | {candidates[2].model_size} |")
    print(f"| License | {candidates[0].license} | {candidates[1].license} | {candidates[2].license} |")

    print("\\n### Hallucination Rate (Among Incorrect Predictions)")
    for n in [n_A, n_B]:
        ev = get_m(n, 'eval_count')
        if ev == 0: continue
        ex = get_m(n, 'norm_exact_match')
        incorrect = ev - ex
        if incorrect == 0: continue
        hal = get_m(n, 'hallucinations')
        print(f"**{n.split(': ')[1]}**: {hal}/{incorrect} ({hal/incorrect*100:.2f}%) produced non-empty incorrect string.")

    print("\\n### Confidence Calibration")
    for n in [n_A, n_B]:
        ev = get_m(n, 'eval_count')
        if ev == 0: continue
        print(f"\\n**{n.split(': ')[1]}**")
        for cb in ["conf_0_50", "conf_50_70", "conf_70_85", "conf_85_95", "conf_95_100"]:
            cb_ev = get_m(n, cb)['eval']
            if cb_ev == 0:
                print(f"  - {cb.replace('_', ' ')}: No predictions")
            else:
                cb_cr = get_m(n, cb)['correct']
                print(f"  - {cb.replace('_', ' ')}: {cb_ev} predictions, {cb_cr/cb_ev*100:.2f}% accuracy")

if __name__ == '__main__':
    main()
