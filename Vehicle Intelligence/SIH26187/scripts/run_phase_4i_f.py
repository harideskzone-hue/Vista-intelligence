import json
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort
import torch
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as T
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from collections import defaultdict
from integration.trajectory_association import TrajectoryAssociator

OUTPUT_DIR = Path("output/phase_4i")
VIDEO_FILE = Path("vehicle_ai/source/compressed_output_TrafficPolice.mp4")

with open(OUTPUT_DIR / "manual_annotations.json") as f:
    annotations = json.load(f)

with open(OUTPUT_DIR / "track_audit.json") as f:
    audit_data = json.load(f)
tracks = audit_data["tracks"]

with open(OUTPUT_DIR / "merger_audit.json") as f:
    merger_data = {f"{m['trackA']}->{m['trackB']}": m for m in json.load(f)}

print("Loading models...")
# 1. Vehicle ReID ONNX
model_path = hf_hub_download(repo_id='occurra/vehicle_vit_clip_reid', filename='vehicle_vit_clip_reid.onnx')
session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# 2. Generic ResNet50
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
res_model = resnet50(weights=ResNet50_Weights.DEFAULT)
res_model.fc = torch.nn.Identity()
res_model = res_model.to(device)
res_model.eval()

transform_res = T.Compose([
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

# Also add 75->110 explicitly if not in merger_data
if "75->110" not in merger_data:
    merger_data["75->110"] = {
        "trackA": "75", "trackB": "110",
        "frame_gap": tracks["110"]["first_frame"] - tracks["75"]["last_frame"],
        "overall_score": 0.0,
        "class_id": tracks["75"]["class_id"]
    }

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

def preprocess_onnx(img):
    img = cv2.resize(img, (256, 256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, axis=0)

print("Computing embeddings...")
embeddings_veh = {}
embeddings_gen = {}
with torch.no_grad():
    for s_tuple, crop in extracted_crops.items():
        # Vehicle ReID
        inp = preprocess_onnx(crop)
        emb = session.run([output_name], {input_name: inp})[0][0]
        emb = emb / np.linalg.norm(emb)
        embeddings_veh[s_tuple] = emb
        
        # Generic ReID
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = transform_res(crop_rgb).unsqueeze(0).to(device)
        emb_gen = res_model(tensor)
        emb_gen = F.normalize(emb_gen, p=2, dim=1).cpu().squeeze().numpy()
        embeddings_gen[s_tuple] = emb_gen

def get_mean_top3(samplesA, samplesB, emb_dict):
    embsA = [emb_dict[tuple(s)] for s in samplesA if tuple(s) in emb_dict]
    embsB = [emb_dict[tuple(s)] for s in samplesB if tuple(s) in emb_dict]
    if not embsA or not embsB: return 0.0
    
    sims = []
    for ea in embsA:
        for eb in embsB:
            sims.append(np.dot(ea, eb))
            
    sims.sort(reverse=True)
    return float(np.mean(sims[:3])) if len(sims) >= 3 else float(np.mean(sims))

print("\n=== TRAJECTORY ASSOCIATION BENCHMARK ===\n")
print(f"Input ByteTrack IDs: {len(tracks)}")
print(f"Candidate pairs: {len(annotations)}\n")

# A. Geometry Only Evaluator
geom_associator = TrajectoryAssociator()
# B. Generic ReID Evaluator
gen_associator = TrajectoryAssociator()
# C. Vehicle ReID Evaluator (Proposed)
veh_associator = TrajectoryAssociator()

# Feed tracks
for pair_id, gt in annotations.items():
    tA, tB = pair_id.split("->")
    m_data = merger_data.get(pair_id, {})
    
    gap = m_data.get("frame_gap", 100)
    geom_score = m_data.get("overall_score", 0.0)
    class_match = (tracks[tA]["class_id"] == tracks[tB]["class_id"])
    
    # 1. Geometry only -> we fake ReID = 1.0 so it passes the ReID gate automatically
    geom_associator.evaluate_candidate(tA, tB, gap, geom_score, 1.0, class_match)
    
    # 2. Generic ReID
    sim_gen = get_mean_top3(pair_samples[pair_id]["A"], pair_samples[pair_id]["B"], embeddings_gen)
    gen_associator.evaluate_candidate(tA, tB, gap, geom_score, sim_gen, class_match)
    
    # 3. Vehicle ReID
    sim_veh = get_mean_top3(pair_samples[pair_id]["A"], pair_samples[pair_id]["B"], embeddings_veh)
    veh_associator.evaluate_candidate(tA, tB, gap, geom_score, sim_veh, class_match)


def evaluate_associator(associator, name):
    tp, fp, fn = 0, 0, 0
    correct_review = 0
    
    print(f"--- {name} ---")
    print(f"{'PAIR':<10} {'GAP':<5} {'GEOM':<5} {'RE-ID':<6} {'CLASS':<6} {'DECISION':<10} {'TRAJ'}")
    print("-" * 55)
    
    counts = {"SAME": 0, "REVIEW": 0, "DIFFERENT": 0}
    
    for log in associator.audit_log:
        pid = f"{log['trackA']}->{log['trackB']}"
        gt_label = annotations[pid]["label"]
        dec = log["decision"]
        counts[dec] += 1
        
        reid_str = f"{log['reid']:.2f}" if log['reid'] < 1.0 or name == "Geometry Only Baseline" else "1.00"
        
        print(f"{pid:<10} {log['gap']:<5} {log['geom']:<5.2f} {reid_str:<6} {'SAME' if log['class_match'] else 'DIFF':<6} {dec:<10} {log['trajB']}")
        
        if gt_label == "UNCERTAIN":
            pass # ignore
        elif dec == "SAME":
            if gt_label == "SAME": tp += 1
            else: fp += 1
        else:
            if gt_label == "SAME": fn += 1
            if dec == "REVIEW" and gt_label == "DIFFERENT": correct_review += 1
            
    prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    print(f"\nDecision Summary:")
    print(f"SAME:       {counts['SAME']}")
    print(f"REVIEW:     {counts['REVIEW']}")
    print(f"DIFFERENT:  {counts['DIFFERENT']}")
    print(f"Automatic merges: {counts['SAME']}")
    
    print(f"\nManual Ground Truth Evaluation:")
    print(f"TP = {tp}")
    print(f"FP = {fp}")
    print(f"Precision = {prec:.2f}")
    print(f"Recall = {rec:.2f}")
    print(f"Correctly deferred to REVIEW/DIFF: {correct_review}")
    print("\n" + "="*55 + "\n")

evaluate_associator(geom_associator, "A. Geometry-only Baseline")
evaluate_associator(gen_associator, "B. Generic ImageNet Re-ID Baseline")
evaluate_associator(veh_associator, "C. Proposed System (Geometry + Vehicle Re-ID)")

print("=== FINAL OUTPUT ===")
print(f"Generated {veh_associator.next_id - 1} persistent Trajectory IDs from {len(tracks)} ByteTrack track fragments.")
