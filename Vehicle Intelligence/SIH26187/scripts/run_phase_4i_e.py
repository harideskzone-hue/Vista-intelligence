import json
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
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

print("Downloading/Loading ONNX model...")
model_path = hf_hub_download(repo_id='occurra/vehicle_vit_clip_reid', filename='vehicle_vit_clip_reid.onnx')
session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

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

print("Extracting crops from video...")
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

def preprocess(img):
    img = cv2.resize(img, (256, 256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, axis=0)

print("Computing embeddings...")
embeddings = {}
for s_tuple, crop in extracted_crops.items():
    inp = preprocess(crop)
    emb = session.run([output_name], {input_name: inp})[0][0]
    emb = emb / np.linalg.norm(emb)
    embeddings[s_tuple] = emb

def get_similarities(samplesA, samplesB):
    embsA = [embeddings[tuple(s)] for s in samplesA if tuple(s) in embeddings]
    embsB = [embeddings[tuple(s)] for s in samplesB if tuple(s) in embeddings]
    if not embsA or not embsB: return 0.0, 0.0
    
    sims = []
    for ea in embsA:
        for eb in embsB:
            sims.append(np.dot(ea, eb))
            
    sims.sort(reverse=True)
    max_sim = sims[0]
    top3_mean = np.mean(sims[:3]) if len(sims) >= 3 else np.mean(sims)
    return float(max_sim), float(top3_mean)

print("\n=== RE-ID MODEL ===")
print("Model: occurra/vehicle_vit_clip_reid")
print("Weights/dataset: Vehicle ReID (ViT CLIP backbone)")
emb_dim = list(embeddings.values())[0].shape[0] if embeddings else "Unknown"
print(f"Embedding dimension: {emb_dim}\n")

print("=== PAIR RESULTS ===")
results = []
for pair_id, gt in annotations.items():
    if gt["label"] == "UNCERTAIN": continue
    
    max_sim, mean_top3 = get_similarities(pair_samples[pair_id]["A"], pair_samples[pair_id]["B"])
    geo_score = merger_data[pair_id]["overall_score"] if pair_id in merger_data else 0.0
    
    print(f"Track {pair_id}")
    print(f"Geometry score    : {geo_score:.2f}")
    print(f"Max Re-ID         : {max_sim:.2f}")
    print(f"Mean Top-3 Re-ID  : {mean_top3:.2f}")
    print(f"Ground truth      : {gt['label']}")
    print()
    
    results.append({
        "pair": pair_id,
        "gt": gt["label"],
        "geo": geo_score,
        "app_max": max_sim,
        "app_mean3": mean_top3
    })

thresholds = [0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]

print("=== APPEARANCE THRESHOLD ANALYSIS (Using Mean Top-3) ===")
print(f"{'Threshold':>10} | {'TP':>4} | {'FP':>4} | {'Precision':>9} | {'Recall':>6}")
print("-" * 46)
for t in thresholds:
    tp, fp, fn = 0, 0, 0
    for r in results:
        is_pred = r["app_mean3"] >= t
        is_gt = r["gt"] == "SAME"
        if is_pred:
            if is_gt: tp += 1
            else: fp += 1
        else:
            if is_gt: fn += 1
    prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    print(f"{t:10.2f} | {tp:4d} | {fp:4d} | {prec:9.2f} | {rec:6.2f}")

print("\n=== COMBINED ASSOCIATION ANALYSIS ===")
print("Rule: Geometry >= 0.50 AND Mean Top-3 Appearance >= T")
print(f"{'App Thresh':>10} | {'TP':>4} | {'FP':>4} | {'Precision':>9} | {'Recall':>6}")
print("-" * 46)
for t in thresholds:
    tp, fp, fn = 0, 0, 0
    for r in results:
        is_pred = (r["geo"] >= 0.50) and (r["app_mean3"] >= t)
        is_gt = r["gt"] == "SAME"
        if is_pred:
            if is_gt: tp += 1
            else: fp += 1
        else:
            if is_gt: fn += 1
    prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    print(f"{t:10.2f} | {tp:4d} | {fp:4d} | {prec:9.2f} | {rec:6.2f}")

print("\n=== KNOWN FALSE POSITIVES (DIFFERENT) ===")
for r in results:
    if r["pair"] in ["4->19", "4->21", "123->159"]:
        print(f"Track {r['pair']} (GT: {r['gt']}) -> Max Re-ID: {r['app_max']:.2f}, Mean Top-3: {r['app_mean3']:.2f}")

print("\n=== STATUS ===")
print("Re-ID Benchmark Complete")
