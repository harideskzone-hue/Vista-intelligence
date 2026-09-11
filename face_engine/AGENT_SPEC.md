# Best Photo Picker v2 — Agent Specification
> v2 — replaces all previous specs entirely.

## SYSTEM_IDENTITY
project:         best-photo-picker v2
person_detector: YOLO26n-person  (yolo26n-person.pt)
face_detector:   YOLO26n-face    (yolo26n-face.pt)
face_landmarker: MediaPipe FaceLandmarker 0.10.14 on FACE CROP
pose_landmarker: MediaPipe PoseLandmarker 0.10.14 on FULL IMAGE

## CRITICAL_ARCHITECTURE_DECISIONS
1. YOLO26n-person on FULL image → persons
2. YOLO26n-face on PERSON CROP  → faces within each person
3. MediaPipe FaceLandmarker on FACE CROP (512px min) → 478 landmarks
4. MediaPipe PoseLandmarker on FULL image → 33 landmarks
5. Body group runs ONCE per image, shared across all faces
6. Each face scored independently
7. Best face = highest score → image score
8. Hard fail ONLY: zero persons OR all faces fail frame check
9. Never artificially enhance image (no denoise, no sharpen)
10. All detectors init ONCE at startup, passed as arguments

## FILE_MAP
pose_scorer/
├── config.py
├── preprocessor.py        PIL EXIF-aware load + blur check
├── frame_check.py         per-face edge/offset (SKIP not REJECT)
├── detection/
│   ├── __init__.py        two-stage pipeline
│   ├── yolo_detector.py   YOLO26n-person + YOLO26n-face wrappers
│   └── crop.py            bbox → padded crop + coord mapping
├── face_group/
│   ├── __init__.py        runs on FACE CROP
│   ├── eye_openness.py
│   ├── gaze_direction.py
│   ├── head_pose.py       FOV-based camera matrix — not naive focal=image_w
│   └── smile.py
├── body_group/
│   ├── __init__.py        runs on FULL image
│   ├── body_orientation.py
│   ├── posture.py
│   ├── shoulder_symmetry.py
│   ├── hand_position.py
│   └── leg_position.py
├── aggregator.py
├── scorer.py
└── reporter.py

models/
├── yolo26n-person.pt
├── yolo26n-face.pt
├── face_landmarker.task
└── pose_landmarker_full.task

## DEPENDENCIES
pip install ultralytics mediapipe==0.10.14 opencv-python==4.9.0.80 numpy==1.26.4 scipy==1.13.0 Pillow==10.3.0 pandas==2.2.2 tqdm==4.66.4

## MODEL_DOWNLOADS
mkdir -p models
wget https://github.com/akanametov/yolo-face/releases/download/v1.0.0/yolo26n-person.pt -O models/yolo26n-person.pt
wget https://github.com/akanametov/yolo-face/releases/download/v1.0.0/yolo26n-face.pt   -O models/yolo26n-face.pt
wget https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task -O models/face_landmarker.task
wget https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task -O models/pose_landmarker_full.task


## MODULE_CONTRACT
Every module file MUST return exactly this dict. Agents generating module code must conform to this.

```python
def score_<module_name>(...) -> dict:
    return {
        "score":       float,   # 0.0 to 100.0
        "confidence":  float,   # 0.0 to 1.0
        "detail":      str,     # human-readable reason, include numeric values
        "skipped":     bool,    # True if landmarks missing or confidence < MIN_TO_SCORE
        "skip_reason": str,     # only populated when skipped=True, else ""
    }
```

**Rules:**
- Never raise exceptions — catch all errors, return skipped=True with skip_reason
- Never import from sibling modules — only from config.py and stdlib/numpy/cv2
- Never hard-code thresholds — always read from config.py
- Confidence must reflect landmark quality, not score quality

---

## GROUP_CONTRACT
Each `face_group/__init__.py` and `body_group/__init__.py` MUST return:

```python
{
    "group_score":      float | None,  # None if ALL modules skipped
    "group_confidence": float,          # mean confidence of non-skipped modules
    "modules": {
        "<module_name>": { ...MODULE_CONTRACT... },
        ...
    }
}
```

---

## PIPELINE_STAGES
Stages run in order. A HARD_REJECT at any stage stops processing immediately.

```
Stage 1: PREFLIGHT
  Input:  image path
  Checks: file readable, min resolution (480x480), not blurry (Laplacian var > 80), person exists
  Output: (original_image, mediapipe_image) or PREFLIGHT_FAIL

Stage 2: FRAME_CHECK
  Input:  pose_landmarks
  Checks: body centre within horizontal offset threshold, no edge cutoffs on LEFT/RIGHT/TOP
  Output: {status: PASS|REJECT, offset_score: float, violations: [...]} or HARD_REJECT

Stage 3: FACE_GROUP  [parallel with Stage 4]
  Input:  face_landmarks, image dimensions
  Runs:   eye_openness, gaze_direction, head_pose, smile
  Output: GROUP_CONTRACT result

Stage 4: BODY_GROUP  [parallel with Stage 3]
  Input:  pose_landmarks
  Runs:   body_orientation, posture, shoulder_symmetry, hand_position, leg_position
  Output: GROUP_CONTRACT result

Stage 5: AGGREGATION
  Input:  frame_offset_score, face_group_result, body_group_result
  Output: final_score (0-100), score_band string

Stage 6: REPORT
  Input:  all stage results
  Output: per-image dict appended to results list
```

---

## LANDMARK_INDEX_TABLE

#

## SCORING_TABLES

#

## WEIGHTS_TABLE
```python
GROUP_WEIGHTS = {
    "frame_offset": 0.10,
    "face_group":   0.55,
    "body_group":   0.35,
}

FACE_MODULE_WEIGHTS = {
    "eye_openness":   0.30,
    "gaze_direction": 0.25,
    "head_pose":      0.25,
    "smile":          0.20,
}

BODY_MODULE_WEIGHTS = {
    "body_orientation":  0.30,
    "posture":           0.25,
    "shoulder_symmetry": 0.20,
    "hand_position":     0.15,
    "leg_position":      0.10,
}
```

---

## FRAME_CHECK_RULES
```yaml
edge_threshold: 0.02          # landmark within 2% of frame edge = cutoff
reject_top: true
reject_left: true
reject_right: true
reject_bottom: false           # legs out of frame is acceptable, flag only

body_centre:
  landmarks: [11, 12, 23, 24]  # shoulders + hips
  formula: mean(x), mean(y)

horizontal_offset_reject: user-defined in config  # default 0.20
vertical_offset_reject: null   # disabled by default

top_edge_landmarks:   [0, 7, 8]           # nose, ears
left_edge_landmarks:  [11, 23, 15, 27]    # l-shoulder, l-hip, l-wrist, l-ankle
right_edge_landmarks: [12, 24, 16, 28]    # r-shoulder, r-hip, r-wrist, r-ankle
```

---

## AGGREGATION_FORMULA
```
# Level 1: within a group
group_score = Σ(score × weight × confidence) / Σ(weight × confidence)
# skipped modules excluded from both numerator and denominator

# Confidence adjustment
if confidence < LOW_THRESHOLD (0.75):
    effective_confidence = confidence * 0.6

# Level 2: across groups
# If a group_score is None (all modules skipped), redistribute that group's weight
final_score = Σ(group_score × group_weight) / Σ(active_group_weights)
```

---

## OUTPUT_SCHEMA
```jsonc
// results.json — array of these objects, sorted by rank ascending
{
  "image":       "string — filename",
  "status":      "SCORED | REJECTED | PREFLIGHT_FAIL",
  "reject_reason": "string — only when status != SCORED",
  "final_score": "float | null",
  "score_band":  "Excellent | Good | Acceptable | Poor | Very Poor | null",
  "rank":        "int | null",

  "frame_check": {
    "status":             "PASS | REJECT",
    "offset_score":       "float",
    "body_centre":        { "x": "float", "y": "float" },
    "offset_from_centre": { "x": "float", "y": "float" },
    "violations":         "array of {edge, landmark, value}",
    "bottom_flag":        "bool"
  },

  "face_group": {
    "group_score":      "float | null",
    "group_confidence": "float",
    "modules": {
      "eye_openness":   "MODULE_CONTRACT",
      "gaze_direction": "MODULE_CONTRACT",
      "head_pose":      "MODULE_CONTRACT",
      "smile":          "MODULE_CONTRACT"
    }
  },

  "body_group": {
    "group_score":      "float | null",
    "group_confidence": "float",
    "modules": {
      "body_orientation":  "MODULE_CONTRACT",
      "posture":           "MODULE_CONTRACT",
      "shoulder_symmetry": "MODULE_CONTRACT",
      "hand_position":     "MODULE_CONTRACT",
      "leg_position":      "MODULE_CONTRACT"
    }
  }
}
```

---

## SCORE_BANDS
```
90–100  Excellent     Strong candidate for best photo
75–90   Good          Minor imperfections
60–75   Acceptable    Noticeable issues, still usable
40–60   Poor          Significant problems
0–40    Very Poor     Likely unusable
```

---

## SKIP_VS_REJECT_RULES
```
landmark.presence < 0.55          → skip this module (not zero, not penalised)
all modules in a group skipped     → group_score = None
both groups None                   → HARD REJECT (no person detected)
frame edge violation               → HARD REJECT (any single left/right/top)
horizontal offset > threshold      → HARD REJECT
preflight fail                     → PREFLIGHT_FAIL (not scored, not ranked)
```

---

## RESOLUTION_STRATEGY
```python
# Always downscale before MediaPipe. Keep original for output.
TARGET_LONG_EDGE = 1280  # px
scale = min(1.0, TARGET_LONG_EDGE / max(h, w))
mediapipe_image = cv2.resize(original, (int(w*scale), int(h*scale)))
# Pass mediapipe_image to detectors, pass original to reporter
```

---

## MEDIAPIPE_INIT_PATTERN
```python
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

face_detector = mp_vision.FaceLandmarker.create_from_options(
    mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path='models/face_landmarker.task'),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
    )
)

pose_detector = mp_vision.PoseLandmarker.create_from_options(
    mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path='models/pose_landmarker_full.task'),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
    )
)
```

---

## CLI_INTERFACE
```bash
python scorer.py --input ./photos --output ./results
python scorer.py --input ./photos/photo_001.jpg --output ./results --debug
```