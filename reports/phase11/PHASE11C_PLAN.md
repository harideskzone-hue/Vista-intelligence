# Phase 11C - ANPR Optimization Plan

This phase systematically optimizes the ANPR pipeline while preserving rigorous ground-truth evaluation, avoiding unvalidated direct E2E tuning.

## Objectives
Optimize the overall ANPR Exact Match rate (E2E) by isolating and improving individual pipeline stages.

## Stage 11C-0: Baseline Reproduction (Current Phase)
Reproduce the frozen Phase 10E results without changing production.
*   **Path A:** 52 GT plates | 28 TP | 8 FP | 24 FN | Precision 77.8% | Recall 53.8% | F1 63.6%
*   **Path B:** 25 text-bearing plates | 20 exact matches | CER 10.8%
*   **Path C:** 25 text-bearing plates | 5 exact matches | Vehicle-gate survival 6/25 (24.0%)

## Future Stages (Upon 11C-0 Clearance)

### 11C-1: Vehicle-Gate Diagnostic
Test the vehicle-gating component independently.
*   **Measure:** GT plate -> detected vehicle overlap, vehicle-gate survival, plate localization after vehicle crop, crop acceptance, OCR availability.

### 11C-2: Plate Detector Optimization
Test plate detector variables (e.g., confidence, imgsz).
*   **Freeze:** Vehicle detector, OCR, crop rules, validator.
*   **Measure:** Plate TP, FP, FN, precision, recall, F1, IoU distribution.

### 11C-3: Crop-Quality Optimization
Investigate padding, min area, aspect ratio.
*   **Measure:** Accepted crops, OCR output, exact match, CER, false/invalid outputs.

### 11C-4: OCR Optimization
Vary OCR processing on the 25 text-bearing GT plate crops.
*   **Measure:** Exact match, CER, character accuracy, invalid-format rate.

### 11C-5: E2E Candidate Validation
Combine top candidates and evaluate against the frozen 20% E2E exact baseline.

### 11C-6: Temporal Voting [BLOCKED]
Blocked until authorized sequence-level ground truth exists.
