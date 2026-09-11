# Human Critic Scoring — Rule Changes
> Fixes derived from re-examining the pipeline through the lens of how a human
> photo critic actually evaluates a batch of images from a shoot.

---

## The Core Principle

A human critic is **ranking photos against each other**, not checking a subject
against an ideal template. They ask:

> "Is this photo usable, and if so, how good is it compared to the others?"

This produces two distinct categories of evaluation:

**Hard reject** — technically unusable regardless of everything else:
- Eyes confirmed closed (subject looks unconscious)
- Person's back fully to the camera (no portrait possible)

**Score and rank** — everything else is a matter of degree:
- Neutral expression, serious look, no smile → valid, just not ideal
- Three-quarter profile, head slightly turned → valid portrait angle
- Slight lean, natural stance → human, not defective
- Looking slightly away → lower score, not rejection

The pipeline currently conflates these two categories in three places.
Each one causes valid, usable photos to be either wrongly rejected or
wrongly penalised.

---

## CHANGE 1 — `pose_scorer/face_group/smile.py`

### Problem
Neutral expression scores **40**. This is treated as a near-failure — the same
band as a poor pose. A composed, neutral, or serious expression is a legitimate
creative choice used in editorial, professional, and candid photography. The
absence of a smile is not a defect.

At weight 0.20 in face group, a neutral expression currently drags face_group
down by **8 points** vs a natural smile. The new value (60) still creates a
clear 8-point gap encouraging smiles, without punishing a composed neutral
expression as heavily as the current 40.

### Score impact (current vs fixed)
| Scenario | Current face_group | Fixed face_group |
|----------|--------------------|-----------------|
| Perfect pose + neutral expression | 85.5 | 89.5 |
| Good pose + neutral expression | 78.0 | 82.5 |
| Awkward pose + neutral expression | 65.5 | 69.5 |
| Perfect pose + natural smile | 97.5 | 97.5 (unchanged) |
| Perfect pose + talking/open mouth | 83.5 | 83.5 (unchanged) |
| Perfect pose + forced/wide smile | 90.5 | 88.5 (−2, forced is slightly worse than neutral, neutral=60) |

### New scoring table

```python
# BEFORE:
if mouth_ratio > 0.50:
    score = 30.0   # talking/open
elif mouth_ratio >= 0.20 and corner_lift < -0.003:
    score = 100.0  # natural smile
elif mouth_ratio >= 0.15 and corner_lift < 0:
    score = 80.0   # subtle smile
elif mouth_ratio < 0.15 and abs(corner_lift) <= 0.003:
    score = 40.0   # neutral  ← WRONG
elif mouth_ratio >= 0.15 and corner_lift >= 0:
    score = 65.0   # forced/wide
else:
    score = 40.0   # neutral fallback  ← WRONG

# AFTER:
if mouth_ratio > 0.50:
    score = 30.0   # talking/open — caught mid-word, expression unusable
elif mouth_ratio >= 0.20 and corner_lift < -0.003:
    score = 100.0  # natural smile — clear positive signal
elif mouth_ratio >= 0.15 and corner_lift < 0:
    score = 85.0   # subtle smile
elif mouth_ratio >= 0.15 and corner_lift >= 0:
    score = 55.0   # forced/wide — looks unnatural, slight penalty
elif mouth_ratio < 0.15 and abs(corner_lift) <= 0.003:
    score = 60.0   # neutral — mild disappointment, smile is expected
else:
    score = 60.0   # neutral fallback
```

### Why forced/wide (55) sits below neutral (60)
A strained wide smile with no genuine lift looks worse than composed neutrality.
The subject looks uncomfortable. Neutral (60) scores above forced (55) because
at least the neutral expression is genuine. Natural smile (100) is the target.

### File
`pose_scorer/face_group/smile.py` — replace the score assignment block only.
The mouth_ratio and corner_lift computation above it is unchanged.

---

## CHANGE 2 — `pose_scorer/face_group/__init__.py`

### Problem
The quality gate rejects the **entire image** when gaze is skipped:

```python
# Current — WRONG:
if gaze_skipped or eyes_closed:
    return { "rejected": True, ... }
```

Gaze skips when `|yaw| > 30°` or `|pitch| > 40°`. Both cases mean the iris
measurement cannot be taken reliably. But gaze scoring **0** (from FIX_GUIDE
skip=0 rule) already penalises these images — the 0 score pulls face_group
down. Rejecting on top of that penalty throws away an otherwise usable photo.

A person at 35° yaw with open eyes and good expression should score ~61 in
face_group (gaze=0 pulling it down) and be ranked low, not discarded entirely.

### New gate logic

```python
# AFTER — reject only on confirmed hard defects:

eye  = module_results["eye_openness"]
head = module_results["head_pose"]

# Hard reject 1: eyes confirmed closed (EAR < 0.10, not a confidence failure)
# eye_openness scores 0 only when EAR < 0.10 (fully closed).
# Low confidence returns skipped=True, not score=0, so those are not caught here.
eyes_confirmed_closed = (
    not eye.get("skipped", True) and eye.get("score", 100) == 0.0
)

# Hard reject 2: face completely undetectable — all modules failed
all_failed = all(res.get("skipped", True) for res in module_results.values())

if eyes_confirmed_closed or all_failed:
    return {
        "_raw": results,
        "group_score": None,
        "group_confidence": 0.0,
        "rejected": True,
        "reject_reason": (
            "Eyes confirmed closed (EAR < 0.10)"
            if eyes_confirmed_closed
            else "All face modules failed — face undetectable"
        ),
        "modules": module_results,
    }

# Gaze skipped → NOT a rejection. The skip=0 score already penalises.
# Score the image and let the final score reflect the penalty.
```

### What this changes
| Scenario | Current | Fixed |
|----------|---------|-------|
| Yaw=35°, gaze skipped, eyes open | REJECTED | SCORED (face_group ~61) |
| Pitch=45°, gaze skipped, eyes open | REJECTED | SCORED (face_group ~47) |
| Eyes closed (EAR < 0.10) | REJECTED | REJECTED (unchanged) |
| All modules failed | REJECTED | REJECTED (unchanged) |

### File
`pose_scorer/face_group/__init__.py` — replace the quality gate block only.
Everything above (roll gate) and below (penalty calculation, group scoring) is
unchanged.

---

## CHANGE 3 — `pose_scorer/config.py` + `pose_scorer/scorer.py`

### Problem
The orientation gate rejects images where `orientation_score < 40`:

```python
# scorer.py — current:
min_orient = config_dict.get('BODY', {}).get('min_orientation_score', 40)
if orientation_mod['score'] < min_orient:
    return _create_fail_result(..., "BODY_ORIENTATION_REJECTED", ...)
```

Looking at the actual score bands in `body_orientation.py`:

| normal_z | Score | What it means |
|----------|-------|---------------|
| > 0.85 | 100 | Facing camera directly |
| ≥ 0.70 | 85 | Slight turn (~15-30°) |
| ≥ 0.50 | 60 | Noticeable turn (~30-45°) |
| ≥ 0.20 | 30 | Significant turn (~45-70°) |
| ≥ 0.00 | 10 | Mostly side-on (~70-85°) |
| < 0.00 | 5 | Back to camera |

**Score 30 (normal_z 0.20-0.50) is currently rejected.** This corresponds
to a 45-70° body turn — the classic three-quarter portrait stance. A human
critic sees this as one of the most natural and flattering angles for a
standing person. The pipeline is discarding these photos.

**Score 10 (normal_z 0.00-0.20) is 70-85° side-on.** At this angle most body
landmarks are still visible but the chest normal barely faces the camera.
Posture and hand position scoring is degraded but not broken. Score it low,
do not reject.

**Score 5 (normal_z < 0.00) means back to camera.** Genuinely unusable —
correct to reject.

### New threshold

```python
# config.py — AFTER:
BODY = {
    "min_orientation_score": 10,   # was 40
    # Only reject score 5 (back to camera) and score 10 (near full side-on)
    # Score 30 (three-quarter, 45-70 deg) now scores instead of rejects
    "lean_penalty_threshold": 0.30,
}
```

`scorer.py` reads `min_orientation_score` from config, so no change needed
there — the config change propagates automatically.

### What this changes
| Orientation score | Current | Fixed |
|------------------|---------|-------|
| 100, 85, 60 | SCORED | SCORED (unchanged) |
| 30 (3/4 turn, 45-70°) | REJECTED | SCORED (body_group pulled low) |
| 10 (side-on, 70-85°) | REJECTED | SCORED (body_group very low) |
| 5 (back to camera) | REJECTED | REJECTED (unchanged) |

The body_group score for a score-30 orientation image will be pulled down
significantly (orientation is 25% of body_group after FIX_GUIDE rebalancing).
The image will rank low — which is correct. It will not be discarded.

---

## Complete Change Summary

| # | File | Change | Lines |
|---|------|--------|-------|
| 1 | `face_group/smile.py` | neutral 40→60, forced 65→55, subtle 80→85 | Score assignment block |
| 2 | `face_group/__init__.py` | Gate: remove gaze_skipped condition, keep eyes_closed only | Quality gate block |
| 3 | `config.py` | `min_orientation_score`: 40→10 | BODY dict |

---

## Rejection Rules — Final Definitive List

After all changes, the pipeline rejects an image in **exactly these cases**:

| Condition | Gate | Reason |
|-----------|------|--------|
| Image too blurry | PREFLIGHT_FAIL | Technically unusable |
| Image too small | PREFLIGHT_FAIL | Technically unusable |
| No person detected | NO_PERSON_DETECTED | Nothing to score |
| Multiple persons | MULTIPLE_PERSONS_DETECTED | Ambiguous subject |
| Person too small in frame | PERSON_TOO_SMALL | Subject too far |
| No face detected | NO_FACE_DETECTED | Cannot score face |
| Face cut off at edge | FACE_FRAME_REJECTED | Face not fully in frame |
| Head roll > 45° | Roll gate in face_group | Face sideways, unreadable |
| Eyes confirmed closed (EAR < 0.10) | Quality gate in face_group | Subject looks unconscious |
| All face modules failed | Quality gate in face_group | Face undetectable |
| Back to camera (orientation score ≤ 5) | Orientation gate in scorer | No portrait possible |

Everything else is **scored and ranked**. Low scores are not rejections.

---

## What Is Never a Rejection After These Changes

| Condition | Old behaviour | New behaviour |
|-----------|---------------|---------------|
| Neutral / serious expression | Scored low (40) | Scored (60) — mild disappointment, smile expected |
| Gaze skipped (yaw > 30°) | **REJECTED** | Scored (gaze=0, face_group ~60) |
| Gaze skipped (pitch > 40°) | **REJECTED** | Scored (gaze=0, face_group low) |
| Three-quarter body turn (45-70°) | **REJECTED** | Scored low (orientation=30) |
| Side-ish body (70-85°) | **REJECTED** | Scored very low (orientation=10) |
| Ankles out of frame | Scored (redistributed) | Scored (redistributed, unchanged) |
| Slight head tilt | Scored (roll penalty) | Scored (roll penalty, unchanged) |
| Person looking slightly away | Scored (gaze penalty) | Scored (gaze penalty, unchanged) |
