# Best Photo Picker — Upgrade Instructions
> This document targets the existing codebase in Temp_MC exactly as it stands.
> Every instruction references a specific file, function, and line.
> Do not rewrite files from scratch. Apply targeted changes only.
> Read every section before touching any file.

---

## CODEBASE STATE ASSESSMENT

### What Is Working Correctly
```
preprocessor.py          ✓ EXIF rotation, blur check, resolution check
frame_check.py           ✓ edge detection, horizontal offset, bottom flag only
detection/crop.py        ✓ asymmetric person crop, face crop, landmark mapping
detection/yolo_detector  ✓ person + face detection, person count gates
detection/__init__.py    ✓ imports correct
scorer.py                ✓ pipeline stages 1–9, gates 2b/6b/7b all present
aggregator.py            ✓ confidence weighting, level-2 redistribution, _aggregate_group
config.py                ✓ all keys present, validate() exists
head_pose.py             ✓ FOV camera matrix, solvePnP, correct landmark IDs
gaze_direction.py        ✓ yaw+pitch gates, h+v offset, width/height normalised
eye_openness.py          ✓ coordinate validity, EAR formula, asymmetry penalty
smile.py                 ✓ scale-invariant mouth_ratio, corner_lift thresholds
```

### What Needs Upgrading — Priority Order
```
UPGRADE 1  body_group/__init__.py      group_score hardcoded None — critical
UPGRADE 2  face_group/__init__.py      group_score hardcoded None — critical
UPGRADE 3  body_orientation.py        wrong confidence source (presence on world lms)
UPGRADE 4  body_orientation.py        winding ambiguity — verify nz is positive
UPGRADE 5  posture.py                 raw coordinate thresholds, wrong confidence source
UPGRADE 6  shoulder_symmetry.py       raw coordinate thresholds, wrong confidence source
UPGRADE 7  hand_position.py           raw coordinate thresholds, wrong confidence source
UPGRADE 8  leg_position.py            raw coordinate thresholds, wrong confidence source
UPGRADE 9  face_group/__init__.py     face quality gate too aggressive — rejecting valid faces
UPGRADE 10 body_group/__init__.py     spine lean penalty affects all modules incorrectly
```

---

## UPGRADE 1 — `body_group/__init__.py` — group_score hardcoded None

### Problem
Line at bottom of `run_body_group()`:
```python
return {
    "_raw": results,
    "group_score": None,   # ← hardcoded — body never contributes to final score
    ...
}
```
`aggregator.py` already computes `body_score = _aggregate_group(...)` and assigns
`body_group['group_score']` after the fact. So `group_score` is actually being
set in `aggregator.py` line: `body_group['group_score'] = round(body_score, 1)...`

This means `group_score` IS being populated but only after aggregation runs.
The `None` returned from `run_body_group()` gets overwritten by aggregator.
**This is not broken — it works. But it is confusing and fragile.**

### Fix
Move the responsibility into `run_body_group()` where it belongs.
In `body_group/__init__.py`, replace the final return block:

```python
# ADD THIS before the return — compute group_score here, not in aggregator
body_weights = config.get('BODY_WEIGHTS', {
    "body_orientation": 0.30, "posture": 0.25, "shoulder_symmetry": 0.20,
    "hand_position": 0.15, "leg_position": 0.10
})
num, den = 0.0, 0.0
low_thresh = config['CONFIDENCE']['low_threshold']
low_factor = config['CONFIDENCE']['low_weight_factor']
for name, res in module_results.items():
    if res.get('skipped', True):
        continue
    w = body_weights.get(name, 0.0)
    c = res['confidence']
    ec = c if c >= low_thresh else c * low_factor
    if ec > 0:
        num += res['score'] * w * ec
        den += w * ec

group_score = round(num / (den + 1e-9), 1) if den > 0 else None

return {
    "_raw": results,
    "group_score": group_score,   # ← calculated, not None
    "group_confidence": round(group_conf, 2),
    "modules": module_results
}
```

---

## UPGRADE 2 — `face_group/__init__.py` — group_score hardcoded None

### Problem
Same pattern as body group. Final return:
```python
return {
    "_raw": results,
    "group_score": None,   # ← hardcoded
    ...
}
```
Aggregator overwrites this, so it technically works — but same fragility.

### Fix
In `face_group/__init__.py`, add calculation before the final return block
(the non-rejected path only):

```python
# ADD THIS before the final return
face_weights = config.get('FACE_WEIGHTS', {
    "head_pose": 0.25, "gaze_direction": 0.25,
    "eye_openness": 0.30, "smile": 0.20
})
num, den = 0.0, 0.0
low_thresh = config['CONFIDENCE']['low_threshold']
low_factor = config['CONFIDENCE']['low_weight_factor']
for name, res in module_results.items():
    if res.get('skipped', True):
        continue
    w = face_weights.get(name, 0.0)
    c = res['confidence']
    ec = c if c >= low_thresh else c * low_factor
    if ec > 0:
        num += res['score'] * w * ec
        den += w * ec

group_score = round(num / (den + 1e-9), 1) if den > 0 else None

return {
    "_raw": results,
    "group_score": group_score,   # ← calculated, not None
    "group_confidence": round(group_conf, 2),
    "modules": module_results
}
```

---

## UPGRADE 3 — `body_orientation.py` — Wrong Confidence Source

### Problem
```python
lms = world_landmarks if world_landmarks else landmarks

# Later:
conf = min([lms[11].presence, lms[12].presence, lms[23].presence, lms[24].presence])
```

When `world_landmarks` is available, `lms = world_landmarks`.
World landmarks **never have `.presence` populated** — always 0.0.
So `conf` = 0.0, which is below `min_to_score` (0.55), and the module always skips.

Also: the visibility check at the top uses `lms[idx].visibility < 0.5` — same problem.
World landmarks also don't populate `.visibility`.

### Fix
Split into two separate sources. Change the top of `score_body_orientation()`:

```python
def score_body_orientation(landmarks, world_landmarks, config) -> dict:
    # TWO SEPARATE SOURCES — never mix them
    lms_3d   = world_landmarks if world_landmarks else landmarks  # geometry only
    lms_conf = landmarks   # always normalised — only source with visibility+presence

    if not lms_3d:
        return {"score": 0.0, "confidence": 0.0, "skipped": True,
                "detail": "", "skip_reason": "No landmarks"}

    # Visibility from NORMALISED landmarks only
    for idx in [11, 12, 23, 24]:
        if lms_conf[idx].visibility < 0.5:
            return {"score": 0.0, "confidence": 0.0, "skipped": True,
                    "detail": "", "skip_reason": f"Landmark {idx} not visible"}

    # Confidence from NORMALISED landmarks using visibility (not presence)
    conf = min(lms_conf[idx].visibility for idx in [11, 12, 23, 24])
    min_conf = config['CONFIDENCE']['min_to_score']
    if conf < min_conf:
        return {"score": 0.0, "confidence": conf, "skipped": True,
                "detail": "", "skip_reason": f"Confidence {conf:.2f} < {min_conf}"}

    # Geometry from lms_3d (world or normalised fallback)
    p11 = np.array([lms_3d[11].x, lms_3d[11].y, lms_3d[11].z])
    p12 = np.array([lms_3d[12].x, lms_3d[12].y, lms_3d[12].z])
    p23 = np.array([lms_3d[23].x, lms_3d[23].y, lms_3d[23].z])
    p24 = np.array([lms_3d[24].x, lms_3d[24].y, lms_3d[24].z])
    # ... rest of function unchanged
```

Remove the original confidence lines at the bottom:
```python
# DELETE THESE TWO LINES:
conf = min([lms[11].presence, lms[12].presence, lms[23].presence, lms[24].presence])
min_conf = config['CONFIDENCE']['min_to_score']
if conf < min_conf:
    return {...skipped...}
```

Update the return to use the `conf` already computed at the top:
```python
    source = "world" if world_landmarks else "normalised"
    return {
        "score": score,
        "confidence": round(conf, 3),
        "detail": f"normal_z:{nz:.3f} src:{source}",
        "skipped": False,
        "skip_reason": ""
    }
```

---

## UPGRADE 4 — `body_orientation.py` — Verify Cross Product Winding

### Current Code
```python
v_shoulder = p11 - p12                     # left - right = RIGHT→LEFT
v_spine    = mid_shoulder - mid_hip        # UP direction
normal = np.cross(v_spine, v_shoulder)     # spine cross shoulder
```

### Check
The comment says "correct winding" but the vector order is `(v_spine, v_shoulder)` not
`(v_shoulder, v_spine)`. Add this debug print temporarily to verify:

```python
print(f"DEBUG nz={nz:.3f} on a known front-facing image")
```

Expected: `nz > 0.70` for a person facing camera.
If you see `nz < 0` on front-facing people → the winding is still wrong.

### Fix If nz Is Negative on Front-Facing People
Replace both vector lines AND the cross product:
```python
v_shoulder = p12 - p11              # RIGHT - LEFT = LEFT→RIGHT across chest
v_spine    = mid_hip - mid_shoulder # hip - shoulder = DOWNWARD
normal = np.cross(v_shoulder, v_spine)
```

This gives nz > 0 when chest faces the camera. Do not change if nz is already positive.

---

## UPGRADE 5 — `posture.py` — Raw Coordinates + Wrong Confidence

### Problem 1: Raw coordinate thresholds break at different distances
```python
spine_angle = abs(math.degrees(math.atan2(dx, dy)))
# dx and dy are raw normalised coords — same lean looks different at different distances
```

### Problem 2: Wrong confidence source
```python
conf = min([landmarks[i].presence for i in req_idx])
# Pose normalised landmarks DO have .presence — but .visibility is more reliable for body
```

### Fix — Normalise spine lean by shoulder width

Replace the spine angle block with:
```python
# Compute shoulder reference for distance-invariant measurement
shoulder_dist = ((landmarks[11].x - landmarks[12].x)**2 +
                 (landmarks[11].y - landmarks[12].y)**2) ** 0.5

if shoulder_dist < 0.02:
    return {"score": 0.0, "confidence": 0.0, "skipped": True,
            "detail": "", "skip_reason": "Shoulders too close — unreliable"}

shoulder_mid_x = (landmarks[11].x + landmarks[12].x) / 2.0
shoulder_mid_y = (landmarks[11].y + landmarks[12].y) / 2.0
hip_mid_x      = (landmarks[23].x + landmarks[24].x) / 2.0

# Lean normalised by shoulder width — distance invariant
dx   = hip_mid_x - shoulder_mid_x
lean = abs(dx) / (shoulder_dist + 1e-6)

if   lean < 0.10: spine_score = 100.0
elif lean < 0.25: spine_score = 80.0
elif lean < 0.50: spine_score = 55.0
else:             spine_score = 25.0
```

Replace head alignment block with normalised version:
```python
nose_x = landmarks[0].x
diff   = abs(nose_x - shoulder_mid_x) / (shoulder_dist + 1e-6)

if   diff < 0.15: head_penalty = 0.0
elif diff < 0.40: head_penalty = 10.0
else:             head_penalty = 25.0
```

Replace confidence line:
```python
# CHANGE: presence → visibility (more reliable for body landmarks)
conf = min(landmarks[i].visibility for i in [0, 11, 12, 23, 24])
```

Update detail string to use lean instead of spine_angle:
```python
"detail": f"lean:{lean:.3f} head_offset:{diff:.3f}"
```

---

## UPGRADE 6 — `shoulder_symmetry.py` — Raw Coordinates + Wrong Confidence

### Problem
```python
diff = abs(landmarks[11].y - landmarks[12].y)
# Raw Y diff — 0.03 means very different things at different distances

conf = min(landmarks[11].presence, landmarks[12].presence)
# .presence is less reliable than .visibility for body landmarks
```

### Fix — Normalise by shoulder width
```python
shoulder_dist = abs(landmarks[11].x - landmarks[12].x)  # horizontal distance
if shoulder_dist < 0.01:
    return {"score": 0.0, "confidence": 0.0, "skipped": True,
            "detail": "", "skip_reason": "Shoulders too close"}

raw_diff  = abs(landmarks[11].y - landmarks[12].y)
norm_diff = raw_diff / (shoulder_dist + 1e-6)

if   norm_diff < 0.05: score = 100.0
elif norm_diff < 0.12: score = 80.0
elif norm_diff < 0.20: score = 55.0
else:                  score = 25.0

# CHANGE: presence → visibility
conf = min(landmarks[11].visibility, landmarks[12].visibility)

side = "left_higher" if landmarks[11].y < landmarks[12].y else "right_higher"
detail = f"norm_diff:{norm_diff:.3f} {side}"
```

---

## UPGRADE 7 — `hand_position.py` — Raw Coordinates + Wrong Confidence

### Problem
```python
dx = abs(landmarks[wrist_idx].x - landmarks[hip_idx].x)
dy = landmarks[wrist_idx].y - landmarks[hip_idx].y
# Raw coords — thresholds (0.15, 0.2, 0.05) mean different things at different distances

conf = min(landmarks[15].presence, landmarks[16].presence)
# Should use .visibility for body
```

### Fix — Normalise by shoulder width

Add at the top of `score_hand_position()`, before `check_wrist()`:
```python
shoulder_dist = ((landmarks[11].x - landmarks[12].x)**2 +
                 (landmarks[11].y - landmarks[12].y)**2) ** 0.5
if shoulder_dist < 0.02:
    return {"score": 0.0, "confidence": 0.0, "skipped": True,
            "detail": "", "skip_reason": "Shoulder reference too small"}
```

Inside `check_wrist()`, normalise all offsets:
```python
def check_wrist(wrist_idx, hip_idx):
    if landmarks[wrist_idx].visibility < 0.5 or landmarks[hip_idx].visibility < 0.5:
        return None, None

    dx_norm = abs(landmarks[wrist_idx].x - landmarks[hip_idx].x) / (shoulder_dist + 1e-6)
    dy_norm = (landmarks[wrist_idx].y - landmarks[hip_idx].y) / (shoulder_dist + 1e-6)
    # Positive dy_norm = wrist below hip, Negative = wrist above hip

    if dy_norm > -0.2 and dx_norm < 0.5:
        return 100.0, "natural_sides"
    elif dy_norm > -0.4 and dx_norm >= 0.5:
        return 85.0, "slightly_away"
    elif -0.8 < dy_norm <= -0.2 and dx_norm < 0.5:
        return 75.0, "hands_on_hips"
    elif -1.2 < dy_norm <= -0.4 and dx_norm < 0.8:
        return 70.0, "arms_crossed"
    elif dy_norm <= -1.2:
        return 50.0, "hands_raised"
    else:
        return 30.0, "covering"
```

Replace confidence lines:
```python
# CHANGE: presence → visibility
if left_score is not None and right_score is not None:
    conf = min(landmarks[15].visibility, landmarks[16].visibility)
elif left_score is not None:
    conf = landmarks[15].visibility
else:
    conf = landmarks[16].visibility
```

---

## UPGRADE 8 — `leg_position.py` — Raw Coordinates + Wrong Confidence

### Problem
```python
stance_width = abs(landmarks[27].x - landmarks[28].x)
# Thresholds 0.05, 0.15, 0.25 are raw coords — break at different distances

conf = min(landmarks[27].presence, landmarks[28].presence)
# Should use .visibility
```

### Fix — Normalise by shoulder width

Add at top of `score_leg_position()`:
```python
shoulder_dist = abs(landmarks[11].x - landmarks[12].x)
if shoulder_dist < 0.01:
    return {"score": 0.0, "confidence": 0.0, "skipped": True,
            "detail": "", "skip_reason": "Shoulder reference too small"}

ankle_sep = abs(landmarks[27].x - landmarks[28].x)
stance    = ankle_sep / (shoulder_dist + 1e-6)
crossed   = landmarks[27].x > landmarks[28].x  # anatomically crossed
```

Replace scoring block:
```python
if crossed:
    score, txt = 50.0,  "crossed"
elif stance < 0.30:
    score, txt = 70.0,  "feet_together"
elif stance <= 0.60:
    score, txt = 100.0, "natural"
elif stance <= 1.00:
    score, txt = 85.0,  "relaxed_wide"
else:
    score, txt = 60.0,  "very_wide"
```

Replace confidence and detail:
```python
# CHANGE: presence → visibility
conf   = min(landmarks[27].visibility, landmarks[28].visibility)
detail = f"stance:{stance:.3f} {txt}"
```

---

## UPGRADE 9 — `face_group/__init__.py` — Face Quality Gate Too Aggressive

### Problem
Current gate rejects if gaze is skipped OR eyes are closed:
```python
gaze_skipped = gaze["skipped"]
eyes_closed  = (not eye["skipped"]) and eye["score"] == 0.0
if gaze_skipped or eyes_closed:
    return {...rejected: True...}
```

Gaze skips when `|yaw| > 30°` — a person looking slightly sideways at 31° yaw will
have gaze skipped, and this gate will reject the entire image even though the photo
may be perfectly good. This is too aggressive.

### Fix — Narrow the gate to genuine failures only
Replace the gate block with:
```python
# Face quality gate — only reject on genuinely unscoreable faces
all_skipped = all(res['skipped'] for res in module_results.values())
eyes_skipped = module_results['eye_openness']['skipped']
head_skipped = module_results['head_pose']['skipped']

if all_skipped:
    return {
        "_raw": results,
        "group_score": None,
        "group_confidence": 0.0,
        "rejected": True,
        "reject_reason": "All face modules skipped — face unscoreable",
        "modules": module_results
    }

if eyes_skipped and head_skipped:
    return {
        "_raw": results,
        "group_score": None,
        "group_confidence": 0.0,
        "rejected": True,
        "reject_reason": "Eyes and head pose both undetectable",
        "modules": module_results
    }

# Gaze skipped alone is NOT a rejection — it happens at moderate yaw
# Eyes closed alone is NOT a rejection — it scores 0 which is correct
```

---

## UPGRADE 10 — `body_group/__init__.py` — Remove Cross-Module Spine Penalty

### Problem
```python
# This block penalises ALL modules when spine angle is high:
if spine_angle >= lean_threshold:
    penalty_factor = max(0.75, 1.0 - (spine_angle - lean_threshold) * 0.02)
    for mod in module_results.values():
        if not mod.get("skipped", False):
            mod["score"] = round(mod["score"] * penalty_factor, 1)
```

This is wrong for two reasons:
1. Posture already scores the lean directly — penalising it again via all modules is double-counting
2. Applying posture penalty to body_orientation, shoulder_symmetry, hand_position,
   leg_position makes no logical sense — those scores are independent of spine angle

### Fix — Remove the entire lean penalty block
Delete these lines from `body_group/__init__.py`:
```python
# DELETE THIS ENTIRE BLOCK:
posture_detail = module_results["posture"].get("detail", "")
try:
    spine_angle = float(posture_detail.split("Spine:")[1].split("deg")[0])
except (IndexError, ValueError):
    spine_angle = 0.0

lean_threshold = config.get('BODY', {}).get('lean_penalty_threshold', 10)
if spine_angle >= lean_threshold:
    penalty_factor = max(0.75, 1.0 - (spine_angle - lean_threshold) * 0.02)
    for mod in module_results.values():
        if not mod.get("skipped", False):
            mod["score"] = round(mod["score"] * penalty_factor, 1)
```

Posture's own score already reflects the lean. Let each module speak for itself.

---

## UPGRADE 11 — `config.py` — Add Missing Keys + Calibration Note

### Add to FACE section:
```python
FACE = {
    "gaze_yaw_gate":        30.0,
    "gaze_pitch_skip":      40.0,
    "gaze_pitch_penalty":   20.0,
    "gaze_pitch_soft_low":  15.0,   # ADD — used by soft penalty range
}
```

### Add calibration warning comment to DETECTION:
```python
DETECTION = {
    ...
    "min_person_height_ratio": 0.35,   # CALIBRATE — print ratios on your camera first
    # Run: print(f"ratio={(py2-py1)/image_h:.3f}") on 20 images before enabling this gate
}
```

---

## DEBUG OUTPUT — What the Pipeline Should Print

Add this function to `scorer.py`. Call it inside `score_image()` when `--debug` is set.
It prints only what was measured — no judgements about the person.

```python
def _print_debug(image_name: str, result: dict) -> None:
    status = result.get('status', '')
    score  = result.get('final_score')
    print(f"\n{'━'*60}")
    print(f"IMAGE: {image_name}")
    if status != 'SCORED':
        print(f"STATUS: {status}")
        print(f"REASON: {result.get('reject_reason', '')}")
        print('━'*60)
        return

    print(f"STATUS: SCORED  |  FINAL: {score}  |  BAND: {result.get('score_band')}")
    print('━'*60)

    det = result.get('detection', {})
    pf  = result.get('preflight', {})
    fc  = result.get('frame_check', {})
    print(f"[CAPTURE]  blur:{pf.get('blur_score')}  res:{pf.get('resolution')}")
    print(f"[DETECT]   person_conf:{det.get('person_conf')}  face_conf:{det.get('face_conf')}")
    print(f"[FRAME]    offset:{fc.get('offset_from_centre',{}).get('x')}  score:{fc.get('offset_score')}  {fc.get('status')}")

    fg = result.get('face_group', {})
    print(f"\n[FACE GROUP — score:{fg.get('group_score')}]")
    for name, mod in fg.get('modules', {}).items():
        if mod.get('skipped'):
            print(f"  {name:<18} SKIP  ({mod.get('skip_reason','')})")
        else:
            print(f"  {name:<18} {mod['score']:>5.1f}  {mod.get('detail','')}")

    bg = result.get('body_group', {})
    print(f"\n[BODY GROUP — score:{bg.get('group_score')}]")
    for name, mod in bg.get('modules', {}).items():
        if mod.get('skipped'):
            print(f"  {name:<18} SKIP  ({mod.get('skip_reason','')})")
        else:
            print(f"  {name:<18} {mod['score']:>5.1f}  {mod.get('detail','')}")
    print('━'*60)
```

Call it in `score_image()` just before returning result:
```python
    if args.debug:   # or pass debug flag into score_image
        _print_debug(image_name, result)
    return result
```

---

## VERIFICATION CHECKLIST — Run After Each Upgrade

After every change, run on a known front-facing full-body image and check:

```
After UPGRADE 3+4 (body_orientation):
  □ body_orientation.detail shows normal_z > 0.70 on front-facing person
  □ body_orientation not skipped on images where person is clearly visible

After UPGRADE 5 (posture):
  □ posture.detail shows lean:0.XX not "Spine:XXdeg"
  □ posture not skipped when shoulders and hips visible

After UPGRADES 6–8 (symmetry, hands, legs):
  □ modules not skipped when landmarks visible
  □ scores change when person is at different distances

After UPGRADE 9 (face gate):
  □ image with person at 35° yaw is SCORED not FACE_QUALITY_REJECTED
  □ image with hand over face is still FACE_QUALITY_REJECTED

After UPGRADE 10 (remove lean penalty):
  □ body_orientation score is not reduced on a person who is standing upright but leaning

After UPGRADES 1+2 (group_score):
  □ final_score is not None on a valid image
  □ face_group.group_score is a number in results.json
  □ body_group.group_score is a number in results.json
```

---

## DO NOT CHANGE

These files are correct and must not be modified:
```
preprocessor.py          — EXIF fix and blur check correct
frame_check.py           — geometry correct, vertical disabled correctly
detection/crop.py        — coordinate chain correct
detection/yolo_detector  — detection and mapping correct
aggregator.py            — confidence weighting correct
head_pose.py             — FOV matrix, solvePnP correct
gaze_direction.py        — h+v normalised offset correct
eye_openness.py          — EAR + asymmetry correct
smile.py                 — scale-invariant ratio correct
scorer.py                — pipeline stages and gates correct
```

---

## KNOWN BUGS — Do Not Reintroduce

```
BUG A  group_score hardcoded None — fixed in UPGRADE 1+2
       Symptom: final_score always None, nothing ranked

BUG B  body_orientation using .presence on world landmarks
       Symptom: body_orientation always skipped, score never contributed
       Fixed in UPGRADE 3

BUG C  Cross product winding nz negative on front-facing person
       Symptom: body_orientation always scores 5
       Check in UPGRADE 4 — only fix if confirmed

BUG D  Raw coordinate thresholds in posture/symmetry/hands/legs
       Symptom: scores inconsistent across different distances
       Fixed in UPGRADES 5–8

BUG E  Face quality gate rejects on gaze skip alone
       Symptom: person at 31°+ yaw rejected as FACE_QUALITY_REJECTED
       Fixed in UPGRADE 9

BUG F  Cross-module lean penalty double-counts posture
       Symptom: all body scores reduced when person leans
       Fixed in UPGRADE 10
```
