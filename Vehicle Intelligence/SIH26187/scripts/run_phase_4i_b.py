import json
import math
from pathlib import Path
import cv2
import numpy as np

OUTPUT_DIR = Path("output/phase_4i")
AUDIT_FILE = OUTPUT_DIR / "track_audit.json"
VIDEO_FILE = Path("vehicle_ai/source/compressed_output_TrafficPolice.mp4")
CROPS_DIR = OUTPUT_DIR / "merges"
CROPS_DIR.mkdir(parents=True, exist_ok=True)

with open(AUDIT_FILE) as f:
    data = json.load(f)

tracks = data["tracks"]
candidates = data["candidates"]

# Manually add the 75 -> 110 candidate for explicit evaluation
candidates.append({
    "trackA": "75",
    "trackB": "110",
    "class_id": tracks["75"]["class_id"],
    "frame_gap": tracks["110"]["first_frame"] - tracks["75"]["last_frame"],
    "distance": 0,
    "size_ratio_w": 0,
    "size_ratio_h": 0,
    "score": 0
})

def calculate_scores(statA, statB):
    if statA["class_id"] != statB["class_id"]:
        return 0, 0, 0, 0, "class mismatch"
        
    gap = statB["first_frame"] - statA["last_frame"]
    if gap <= 0:
        return 0, 0, 0, 0, "negative or zero gap"
        
    # Temporal
    if gap <= 5: t_score = 1.0
    elif gap <= 15: t_score = 0.7
    elif gap <= 40: t_score = 0.4
    else: t_score = 0.0
    
    # Spatial
    _, ax, ay, aw, ah = statA["trajectory"][-1]
    _, bx, by, bw, bh = statB["trajectory"][0]
    
    dist = math.hypot(bx - ax, by - ay)
    norm_dist = dist / max(aw, bw)
    
    if norm_dist < 0.2: s_score = 1.0
    elif norm_dist < 0.5: s_score = 0.7
    elif norm_dist < 1.0: s_score = 0.4
    else: s_score = 0.0
    
    # Size
    rw = max(aw, bw) / min(aw, bw) if min(aw, bw) > 0 else float('inf')
    rh = max(ah, bh) / min(ah, bh) if min(ah, bh) > 0 else float('inf')
    max_r = max(rw, rh)
    
    if max_r < 1.1: sz_score = 1.0
    elif max_r < 1.3: sz_score = 0.7
    elif max_r < 1.6: sz_score = 0.4
    else: sz_score = 0.0
    
    if t_score == 0 or s_score == 0 or sz_score == 0:
        overall = 0.0
    else:
        overall = (t_score + s_score + sz_score) / 3.0
        
    return t_score, s_score, sz_score, overall, gap

results = []
frames_to_extract = set()

for c in candidates:
    idA = str(c["trackA"])
    idB = str(c["trackB"])
    
    statA = tracks[idA]
    statB = tracks[idB]
    
    t_score, s_score, sz_score, overall, gap = calculate_scores(statA, statB)
    
    if overall == 0.0:
        classification = "NOT_MERGED"
        if gap > 40:
            reason = "gap exceeds continuity window"
        elif s_score == 0:
            reason = "spatial distance too large"
        elif sz_score == 0:
            reason = "size change too large"
        else:
            reason = "class mismatch or overlap"
    elif overall >= 0.8:
        classification = "CANDIDATE_SAME_VEHICLE"
        reason = "high confidence"
    elif overall >= 0.5:
        classification = "REVIEW"
        reason = "moderate confidence"
    else:
        classification = "DIFFERENT_VEHICLE"
        reason = "low confidence"
        
    results.append({
        "trackA": idA,
        "trackB": idB,
        "class_id": statA["class_id"],
        "frame_gap": gap,
        "temporal_score": t_score,
        "spatial_score": s_score,
        "size_score": sz_score,
        "overall_score": overall,
        "classification": classification,
        "reason": reason,
        "frameA": statA["last_frame"],
        "frameB": statB["first_frame"],
        "trajA": statA["trajectory"][-1],
        "trajB": statB["trajectory"][0]
    })
    
    frames_to_extract.add(statA["last_frame"])
    frames_to_extract.add(statB["first_frame"])

# Extract crops
cap = cv2.VideoCapture(str(VIDEO_FILE))
frame_idx = 0
frames_dict = {}

while True:
    ret, frame = cap.read()
    if not ret: break
    frame_idx += 1
    if frame_idx in frames_to_extract:
        frames_dict[frame_idx] = frame.copy()
cap.release()

for res in results:
    fA = res["frameA"]
    fB = res["frameB"]
    
    _, cxA, cyA, wA, hA = res["trajA"]
    _, cxB, cyB, wB, hB = res["trajB"]
    
    def get_crop(img, cx, cy, w, h):
        x1 = max(0, int(cx - w/2))
        y1 = max(0, int(cy - h/2))
        x2 = min(img.shape[1], int(cx + w/2))
        y2 = min(img.shape[0], int(cy + h/2))
        return img[y1:y2, x1:x2]
        
    cropA = get_crop(frames_dict[fA], cxA, cyA, wA, hA)
    cropB = get_crop(frames_dict[fB], cxB, cyB, wB, hB)
    
    if cropA.size == 0 or cropB.size == 0:
        continue
    
    max_h = max(cropA.shape[0], cropB.shape[0])
    padA = max_h - cropA.shape[0]
    padB = max_h - cropB.shape[0]
    
    cropA_pad = np.pad(cropA, ((0, padA), (0,0), (0,0)), mode='constant')
    cropB_pad = np.pad(cropB, ((0, padB), (0,0), (0,0)), mode='constant')
    
    divider = np.zeros((max_h, 10, 3), dtype=np.uint8)
    divider[:] = (0, 0, 255) # Red divider
    combo = np.concatenate([cropA_pad, divider, cropB_pad], axis=1)
    
    cv2.imwrite(str(CROPS_DIR / f"merge_{res['trackA']}_{res['trackB']}.jpg"), combo)

print("=== TRAJECTORY MERGER VALIDATION ===")
for res in results:
    print(f"Track {res['trackA']} -> {res['trackB']}")
    print(f"  class compatibility : {'PASS' if res['reason'] != 'class mismatch' else 'FAIL'}")
    print(f"  frame gap           : {res['frame_gap']}")
    print(f"  spatial score       : {res['spatial_score']:.2f}")
    print(f"  size score          : {res['size_score']:.2f}")
    print(f"  temporal score      : {res['temporal_score']:.2f}")
    print(f"  overall score       : {res['overall_score']:.2f}")
    print(f"  classification      : {res['classification']}")
    if res['classification'] == 'NOT_MERGED':
        print(f"  reason              : {res['reason']}")
    print()

with open(OUTPUT_DIR / "merger_audit.json", "w") as f:
    json.dump(results, f, indent=2)

print("STATUS: MERGER VALIDATION COMPLETE")
