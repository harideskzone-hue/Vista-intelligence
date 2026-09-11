import sys
import cv2
import json
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO

from anpr.temporal_voter import TemporalPlateVoter
from anpr.fuzzy_temporal_voter import FuzzyTemporalVoter
from anpr.plate_quality import PlateQualityScorer
from anpr.plate_format import PlateFormatValidator
from anpr.ocr import PlateOCR

# Ground truth is empty
GROUND_TRUTH = {}

VIDEO = Path("vehicle_ai/source/compressed_output_TrafficPolice.mp4")
PLATE_MODEL = Path("anpr/models/best.pt")
OUTPUT = Path("output/phase_4h")
TRACKS_DIR = OUTPUT / "tracks"

OUTPUT.mkdir(parents=True, exist_ok=True)
TRACKS_DIR.mkdir(parents=True, exist_ok=True)

print("=== PHASE 4H: REAL TEMPORAL ANPR EXPERIMENT ===")

vehicle_model = YOLO("yolo11n.pt")
plate_model = YOLO(str(PLATE_MODEL))
ocr_engine = PlateOCR(languages=["en"], gpu=False)

exact_voters = defaultdict(lambda: TemporalPlateVoter(min_observations=3, min_confidence=0.50))
fuzzy_voters = defaultdict(lambda: FuzzyTemporalVoter(min_observations=3, min_confidence=0.50))

# To track statistics
track_stats = defaultdict(lambda: {
    "plate_observations": 0,
    "ocr_observations": 0,
    "raw_candidates": set(),
    "quality_scores": []
})

cap = cv2.VideoCapture(str(VIDEO))
if not cap.isOpened():
    raise RuntimeError(f"Unable to open {VIDEO}")

frame_index = 0
vehicle_observations = 0
plate_observations = 0
ocr_observations = 0
FRAME_INTERVAL = 3
MIN_VEHICLE_WIDTH = 100
MIN_VEHICLE_HEIGHT = 70

while True:
    success, frame = cap.read()
    if not success:
        break
    frame_index += 1

    results = vehicle_model.track(
        source=frame, classes=[2, 3, 5, 7], conf=0.40, persist=True, tracker="bytetrack.yaml", verbose=False
    )
    if not results or results[0].boxes is None or results[0].boxes.id is None:
        continue

    result = results[0]
    track_ids = result.boxes.id.int().cpu().tolist()
    boxes = result.boxes.xyxy.int().cpu().tolist()
    confidences = result.boxes.conf.float().cpu().tolist()

    for track_id, bbox, vehicle_conf in zip(track_ids, boxes, confidences):
        vehicle_observations += 1
        x1, y1, x2, y2 = bbox
        if (x2 - x1) < MIN_VEHICLE_WIDTH or (y2 - y1) < MIN_VEHICLE_HEIGHT:
            continue
        if frame_index % FRAME_INTERVAL != 0:
            continue

        vehicle_crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
        if vehicle_crop.size == 0:
            continue

        plate_results = plate_model.predict(source=vehicle_crop, conf=0.25, imgsz=640, verbose=False)
        best_plate = None
        for plate_result in plate_results:
            if plate_result.boxes is None or len(plate_result.boxes) == 0:
                continue
            for box in plate_result.boxes:
                pconf = float(box.conf[0].item())
                coords = box.xyxy[0].cpu().tolist()
                if best_plate is None or pconf > best_plate[0]:
                    best_plate = (pconf, coords)

        if best_plate is None:
            continue

        plate_observations += 1
        track_stats[track_id]["plate_observations"] += 1
        pconf, coords = best_plate
        px1, py1, px2, py2 = map(int, coords)
        ph, pw = vehicle_crop.shape[:2]
        px1 = max(0, min(px1, pw - 1))
        py1 = max(0, min(py1, ph - 1))
        px2 = max(0, min(px2, pw))
        py2 = max(0, min(py2, ph))

        if px2 <= px1 or py2 <= py1:
            continue

        plate_crop = vehicle_crop[py1:py2, px1:px2]
        if plate_crop.size == 0:
            continue

        # Quality Gate
        quality_score = PlateQualityScorer.compute(plate_crop)
        track_stats[track_id]["quality_scores"].append(quality_score)
        
        # Save sample crops up to 5 per track
        if track_stats[track_id]["plate_observations"] <= 5:
            t_dir = TRACKS_DIR / f"track_{track_id}"
            t_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(t_dir / f"frame_{frame_index}_plate.jpg"), plate_crop)

        # Do not discard, but weight by quality score later
        candidates = ocr_engine.read(plate_crop)
        for candidate in candidates:
            ocr_observations += 1
            track_stats[track_id]["ocr_observations"] += 1
            track_stats[track_id]["raw_candidates"].add(candidate.text)
            
            # Exact Voter
            exact_voters[track_id].add(
                text=candidate.text,
                ocr_confidence=candidate.confidence,
                detector_confidence=pconf,
                frame_index=frame_index
            )
            
            # Fuzzy Voter
            fuzzy_voters[track_id].add(
                text=candidate.text,
                ocr_confidence=candidate.confidence,
                detector_confidence=pconf,
                quality_score=quality_score,
                frame_index=frame_index
            )

cap.release()

# Reporting
exact_confirmed = 0
exact_probable = 0
exact_unknown = 0
exact_valid_format = 0

fuzzy_confirmed = 0
fuzzy_probable = 0
fuzzy_unknown = 0
fuzzy_valid_format = 0

evaluation_data = []

print("\n=== TRACK REPORT ===")
for track_id in sorted(fuzzy_voters.keys()):
    exact_decision = exact_voters[track_id].decide()
    fuzzy_decision = fuzzy_voters[track_id].decide()

    if exact_decision.status == "CONFIRMED": exact_confirmed += 1
    elif exact_decision.status == "PROBABLE": exact_probable += 1
    else: exact_unknown += 1
    
    if PlateFormatValidator.validate(exact_decision.text or "").format_score >= 0.8:
        exact_valid_format += 1

    if fuzzy_decision.status == "CONFIRMED": fuzzy_confirmed += 1
    elif fuzzy_decision.status == "PROBABLE": fuzzy_probable += 1
    else: fuzzy_unknown += 1

    if PlateFormatValidator.validate(fuzzy_decision.text or "").format_score >= 0.8:
        fuzzy_valid_format += 1

    q_scores = track_stats[track_id]["quality_scores"]
    avg_q = sum(q_scores)/len(q_scores) if q_scores else 0

    print(f"Track {track_id}")
    print(f"  Raw OCR candidates     : {', '.join(track_stats[track_id]['raw_candidates'])}")
    print(f"  Plate observations     : {track_stats[track_id]['plate_observations']}")
    print(f"  OCR observations       : {track_stats[track_id]['ocr_observations']}")
    print(f"  Best exact-voter result: {exact_decision.text} ({exact_decision.status})")
    print(f"  Best fuzzy-voter result: {fuzzy_decision.text} ({fuzzy_decision.status})")
    print(f"  Fuzzy confidence       : {fuzzy_decision.confidence:.3f}")
    if hasattr(fuzzy_decision, 'format_type'):
        print(f"  Format score/type      : {PlateFormatValidator.validate(fuzzy_decision.text or '').format_score:.2f} / {fuzzy_decision.format_type}")
    print(f"  Quality statistics     : avg={avg_q:.2f}, min={min(q_scores) if q_scores else 0:.2f}, max={max(q_scores) if q_scores else 0:.2f}")
    print(f"  Final status           : EXACT={exact_decision.status} | FUZZY={fuzzy_decision.status}")
    print()
    
    evaluation_data.append({
        "track_id": track_id,
        "exact_result": exact_decision.text,
        "exact_status": exact_decision.status,
        "fuzzy_result": fuzzy_decision.text,
        "fuzzy_status": fuzzy_decision.status,
        "fuzzy_confidence": fuzzy_decision.confidence,
        "plate_obs": track_stats[track_id]['plate_observations'],
        "ocr_obs": track_stats[track_id]['ocr_observations'],
        "avg_quality": avg_q
    })

with open(OUTPUT / "evaluation.json", "w") as f:
    json.dump(evaluation_data, f, indent=2)

summary = f"""
=== PHASE 4H BENCHMARK SUMMARY ===
Total frames           : {frame_index}
Vehicle tracks         : {len(fuzzy_voters)}
Vehicle observations   : {vehicle_observations}
Plate observations     : {plate_observations}
OCR observations       : {ocr_observations}

Exact voter:
    CONFIRMED          : {exact_confirmed}
    PROBABLE           : {exact_probable}
    UNKNOWN            : {exact_unknown}

Fuzzy voter:
    CONFIRMED          : {fuzzy_confirmed}
    PROBABLE           : {fuzzy_probable}
    UNKNOWN            : {fuzzy_unknown}

Valid-format outputs:
    Exact              : {exact_valid_format}
    Fuzzy              : {fuzzy_valid_format}

Ground-truth entries:
    {len(GROUND_TRUTH)}

Accuracy:
    NOT AVAILABLE
"""

with open(OUTPUT / "benchmark.txt", "w") as f:
    f.write(summary)

print(summary)
print("STATUS: EXPERIMENT COMPLETE")
