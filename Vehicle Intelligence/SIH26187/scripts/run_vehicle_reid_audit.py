import json
from pathlib import Path
import cv2
import torch
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

OUTPUT_DIR = Path("output/phase_4i")
VIDEO_FILE = Path("vehicle_ai/source/compressed_output_TrafficPolice.mp4")

with open(OUTPUT_DIR / "manual_annotations.json") as f:
    annotations = json.load(f)

with open(OUTPUT_DIR / "track_audit.json") as f:
    audit_data = json.load(f)
tracks = audit_data["tracks"]

with open(OUTPUT_DIR / "merger_audit.json") as f:
    merger_data = {f"{m['trackA']}->{m['trackB']}": m for m in json.load(f)}

# Initialize Re-ID model (ResNet50 baseline)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
model.fc = torch.nn.Identity() # Remove classification head
model = model.to(device)
model.eval()

transform = T.Compose([
    T.ToPILImage(),
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_crops_for_track(track_id, max_samples=3):
    traj = tracks[str(track_id)]["trajectory"]
    if len(traj) <= max_samples:
        samples = traj
    else:
        indices = np.linspace(0, len(traj)-1, max_samples, dtype=int)
        samples = [traj[i] for i in indices]
    return samples

frames_to_extract = defaultdict(list)
pair_samples = {}

for pair_id in annotations.keys():
    trackA, trackB = pair_id.split("->")
    samplesA = extract_crops_for_track(trackA)
    samplesB = extract_crops_for_track(trackB)
    
    pair_samples[pair_id] = {"A": samplesA, "B": samplesB}
    for s in samplesA: frames_to_extract[s[0]].append(s)
    for s in samplesB: frames_to_extract[s[0]].append(s)

# Read video and extract crops
cap = cv2.VideoCapture(str(VIDEO_FILE))
frame_idx = 0
extracted_crops = {} 

while True:
    ret, frame = cap.read()
    if not ret: break
    frame_idx += 1
    
    if frame_idx in frames_to_extract:
        for s in frames_to_extract[frame_idx]:
            _, cx, cy, w, h = s
            x1 = max(0, int(cx - w/2))
            y1 = max(0, int(cy - h/2))
            x2 = min(frame.shape[1], int(cx + w/2))
            y2 = min(frame.shape[0], int(cy + h/2))
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                extracted_crops[tuple(s)] = crop

cap.release()

# Compute embeddings
embeddings = {}
with torch.no_grad():
    for s_tuple, crop in extracted_crops.items():
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = transform(crop_rgb).unsqueeze(0).to(device)
        emb = model(tensor)
        emb = F.normalize(emb, p=2, dim=1)
        embeddings[s_tuple] = emb.cpu().squeeze()

def get_similarity(samplesA, samplesB):
    embsA = [embeddings[tuple(s)] for s in samplesA if tuple(s) in embeddings]
    embsB = [embeddings[tuple(s)] for s in samplesB if tuple(s) in embeddings]
    
    if not embsA or not embsB: return 0.0
    
    max_sim = 0.0
    for ea in embsA:
        for eb in embsB:
            sim = F.cosine_similarity(ea.unsqueeze(0), eb.unsqueeze(0)).item()
            if sim > max_sim: max_sim = sim
    return max_sim

print("=== APPEARANCE RE-ID AUDIT ===")
results = []
for pair_id, gt in annotations.items():
    if gt["label"] == "UNCERTAIN": continue
    
    sim = get_similarity(pair_samples[pair_id]["A"], pair_samples[pair_id]["B"])
    geo_score = merger_data[pair_id]["overall_score"] if pair_id in merger_data else 0.0
    
    print(f"Track {pair_id}")
    print(f"Geometry score       : {geo_score:.2f}")
    print(f"Appearance similarity: {sim:.2f}")
    print(f"Ground truth         : {gt['label']}")
    print()
    
    results.append({
        "pair": pair_id,
        "gt": gt["label"],
        "geo": geo_score,
        "app": sim
    })

thresholds = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
print("=== APPEARANCE THRESHOLD ANALYSIS ===")
print(f"{'Similarity':>10} | {'Predicted SAME':>14} | {'TP':>4} | {'FP':>4} | {'Precision':>9} | {'Recall':>6}")
print("-" * 65)

for t in thresholds:
    pred_same = 0
    tp, fp, fn = 0, 0, 0
    for r in results:
        is_pred = r["app"] >= t
        is_gt = r["gt"] == "SAME"
        
        if is_pred:
            pred_same += 1
            if is_gt: tp += 1
            else: fp += 1
        else:
            if is_gt: fn += 1
            
    prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    print(f"{t:10.2f} | {pred_same:14d} | {tp:4d} | {fp:4d} | {prec:9.2f} | {rec:6.2f}")

print("\n=== COMBINED ASSOCIATION ANALYSIS ===")
print("Rule: Geometry >= 0.50 AND Appearance >= T")
print(f"{'App Thresh':>10} | {'Predicted SAME':>14} | {'TP':>4} | {'FP':>4} | {'Precision':>9} | {'Recall':>6}")
print("-" * 65)

for t in thresholds:
    pred_same = 0
    tp, fp, fn = 0, 0, 0
    for r in results:
        is_pred = (r["geo"] >= 0.50) and (r["app"] >= t)
        is_gt = r["gt"] == "SAME"
        
        if is_pred:
            pred_same += 1
            if is_gt: tp += 1
            else: fp += 1
        else:
            if is_gt: fn += 1
            
    prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    print(f"{t:10.2f} | {pred_same:14d} | {tp:4d} | {fp:4d} | {prec:9.2f} | {rec:6.2f}")
