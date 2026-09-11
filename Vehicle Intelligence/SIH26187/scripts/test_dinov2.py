import json
from pathlib import Path
import cv2
import torch
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np

OUTPUT_DIR = Path("output/phase_4i")
VIDEO_FILE = Path("vehicle_ai/source/compressed_output_TrafficPolice.mp4")

with open(OUTPUT_DIR / "manual_annotations.json") as f:
    annotations = json.load(f)

with open(OUTPUT_DIR / "track_audit.json") as f:
    audit_data = json.load(f)
tracks = audit_data["tracks"]

# Load DINOv2
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
model = model.to(device)
model.eval()

transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_crops(track_id):
    traj = tracks[str(track_id)]["trajectory"]
    max_samples = 3
    if len(traj) <= max_samples:
        samples = traj
    else:
        indices = np.linspace(0, len(traj)-1, max_samples, dtype=int)
        samples = [traj[i] for i in indices]
    return samples

frames_to_extract = {}
for pair_id in annotations.keys():
    tA, tB = pair_id.split("->")
    for s in extract_crops(tA): frames_to_extract.setdefault(s[0], []).append(s)
    for s in extract_crops(tB): frames_to_extract.setdefault(s[0], []).append(s)

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

embeddings = {}
with torch.no_grad():
    for s_tuple, crop in extracted_crops.items():
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = transform(crop_rgb).unsqueeze(0).to(device)
        emb = model(tensor)
        emb = F.normalize(emb, p=2, dim=1)
        embeddings[s_tuple] = emb.cpu().squeeze()

def get_similarities(tA, tB):
    sA = extract_crops(tA)
    sB = extract_crops(tB)
    embsA = [embeddings[tuple(s)] for s in sA if tuple(s) in embeddings]
    embsB = [embeddings[tuple(s)] for s in sB if tuple(s) in embeddings]
    if not embsA or not embsB: return 0.0, 0.0
    
    sims = []
    for ea in embsA:
        for eb in embsB:
            sims.append(F.cosine_similarity(ea.unsqueeze(0), eb.unsqueeze(0)).item())
    
    sims.sort(reverse=True)
    max_sim = sims[0]
    top3_mean = np.mean(sims[:3]) if len(sims) >= 3 else np.mean(sims)
    return max_sim, top3_mean

print("=== DINOv2 TEST ===")
for pair_id, gt in annotations.items():
    if gt["label"] == "UNCERTAIN": continue
    tA, tB = pair_id.split("->")
    max_s, mean_s = get_similarities(tA, tB)
    print(f"{pair_id} ({gt['label']}) | Max: {max_s:.2f} | MeanTop3: {mean_s:.2f}")

