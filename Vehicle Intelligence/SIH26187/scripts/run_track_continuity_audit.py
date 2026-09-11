from pathlib import Path
from collections import defaultdict
import json
import math
import cv2
from ultralytics import YOLO

VIDEO = Path("vehicle_ai/source/compressed_output_TrafficPolice.mp4")
MODEL = YOLO("yolo11n.pt")
CLASSES = [2, 3, 5, 7]
OUTPUT_DIR = Path("output/phase_4i")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(str(VIDEO))
if not cap.isOpened():
    raise RuntimeError(f"Unable to open {VIDEO}")

# track_id -> statistics
tracks = defaultdict(lambda: {
    "class_id": None,
    "first_frame": None,
    "last_frame": None,
    "detection_count": 0,
    "trajectory": [] # list of (frame, cx, cy, w, h)
})

frame_count = 0
while True:
    success, frame = cap.read()
    if not success:
        break
    frame_count += 1

    results = MODEL.track(
        source=frame, classes=CLASSES, conf=0.40, persist=True, tracker="bytetrack.yaml", verbose=False
    )
    if not results or results[0].boxes is None or results[0].boxes.id is None:
        continue

    ids = results[0].boxes.id.int().cpu().tolist()
    classes = results[0].boxes.cls.int().cpu().tolist()
    boxes = results[0].boxes.xyxy.int().cpu().tolist()

    for t_id, c_id, bbox in zip(ids, classes, boxes):
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = x2 - x1
        h = y2 - y1

        stats = tracks[t_id]
        if stats["first_frame"] is None:
            stats["first_frame"] = frame_count
            stats["class_id"] = c_id # Use first class assignment for simplicity
        
        stats["last_frame"] = frame_count
        stats["detection_count"] += 1
        stats["trajectory"].append((frame_count, cx, cy, w, h))

cap.release()

# Continuity audit rules
MAX_FRAME_GAP = 40
MAX_DISTANCE = 150.0
MAX_SIZE_RATIO = 1.6

candidates = []

track_ids = sorted(tracks.keys())
for idA in track_ids:
    for idB in track_ids:
        if idA == idB: continue
        
        statA = tracks[idA]
        statB = tracks[idB]
        
        # Track B must start AFTER Track A ends
        if statB["first_frame"] <= statA["last_frame"]:
            continue
            
        frame_gap = statB["first_frame"] - statA["last_frame"]
        if frame_gap > MAX_FRAME_GAP:
            continue
            
        if statA["class_id"] != statB["class_id"]:
            continue
            
        # Get last position of A and first position of B
        _, ax, ay, aw, ah = statA["trajectory"][-1]
        _, bx, by, bw, bh = statB["trajectory"][0]
        
        dist = math.hypot(bx - ax, by - ay)
        if dist > MAX_DISTANCE:
            continue
            
        size_ratio_w = max(aw, bw) / min(aw, bw) if min(aw, bw) > 0 else float('inf')
        size_ratio_h = max(ah, bh) / min(ah, bh) if min(ah, bh) > 0 else float('inf')
        
        if size_ratio_w > MAX_SIZE_RATIO or size_ratio_h > MAX_SIZE_RATIO:
            continue
            
        score = dist + (frame_gap * 2.0) # Simple heuristic score, lower is better
        
        candidates.append({
            "trackA": idA,
            "trackB": idB,
            "class_id": statA["class_id"],
            "frame_gap": frame_gap,
            "distance": round(dist, 2),
            "size_ratio_w": round(size_ratio_w, 2),
            "size_ratio_h": round(size_ratio_h, 2),
            "score": round(score, 2)
        })

candidates.sort(key=lambda x: x["score"])

# Write JSON Audit
with open(OUTPUT_DIR / "track_audit.json", "w") as f:
    json.dump({"tracks": tracks, "candidates": candidates}, f, indent=2)

# Write human-readable TXT
with open(OUTPUT_DIR / "candidate_merges.txt", "w") as f:
    f.write(f"ByteTrack IDs              : {len(tracks)}\n")
    f.write(f"Candidate fragmented pairs : {len(candidates)}\n\n")
    for c in candidates:
        f.write("Candidate:\n")
        f.write(f"Track {c['trackA']} -> Track {c['trackB']}\n")
        f.write(f"class: {c['class_id']}\n")
        f.write(f"frame gap: {c['frame_gap']}\n")
        f.write(f"distance: {c['distance']} pixels\n")
        f.write(f"size ratio: W={c['size_ratio_w']}, H={c['size_ratio_h']}\n")
        f.write(f"score: {c['score']}\n\n")

print("\n=== TRACK CONTINUITY AUDIT ===")
print(f"ByteTrack IDs              : {len(tracks)}")
print(f"Candidate fragmented pairs : {len(candidates)}")
print()

# Print top 15 candidates for inspection
for c in candidates[:15]:
    print("Candidate:")
    print(f"Track {c['trackA']} -> Track {c['trackB']}")
    print(f"class: {c['class_id']}")
    print(f"frame gap: {c['frame_gap']}")
    print(f"distance: {c['distance']} pixels")
    print(f"size ratio: W={c['size_ratio_w']}, H={c['size_ratio_h']}")
    print(f"score: {c['score']}")
    print()

print("STATUS: AUDIT COMPLETE")
