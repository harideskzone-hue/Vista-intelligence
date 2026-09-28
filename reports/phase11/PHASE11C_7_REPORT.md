# Phase 11C-7: Plate Detector Optimization Report

## 1. Objective
Investigate whether the current plate detector (`best.pt`) is the bottleneck responsible for the 6/20 localization failures observed in Phase 11C-3. The experiment evaluates plate confidence and inference size (imgsz) sweeping over the 20 successfully contained vehicle crops, utilizing the isolated downstream OCR (with CLAHE) configuration.

## 2. Experimental Data
Evaluations were strictly run against the **20 vehicle crops** (with 10% padding) successfully acquired by the raw YOLO vehicle detector. Native inference size for `best.pt` was determined to be `640`.

| Configuration | Plate Loc (TP) | False Positives (FP) | False Negatives (FN) | Accepted Crops | OCR Exact | CER (Loc. Only) | FPS |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **conf=0.25, imgsz=640 (Baseline)** | 14 / 20 | 4 | 6 | 14 | **14 / 25 E2E Ceiling** | 0.0000 | 9.8 |
| conf=0.2, imgsz=640 | 14 / 20 | 6 | 6 | 14 | **14 / 25 E2E Ceiling** | 0.0000 | 39.6 |
| conf=0.15, imgsz=640 | 14 / 20 | 9 | 6 | 14 | **14 / 25 E2E Ceiling** | 0.0000 | 46.0 |
| conf=0.1, imgsz=640 | 15 / 20 | 16 | 5 | 15 | **15 / 25 E2E Ceiling** | 0.0000 | 45.1 |
| conf=0.25, imgsz=320 | 9 / 20 | 3 | 11 | 9 | **8 / 25 E2E Ceiling** | 0.0111 | 9.4 |

## 3. Key Findings
- **Baseline Confirmation:** The frozen baseline (`conf=0.25, imgsz=640`) successfully localized 14 plates, generating 4 false positives, and matching 14 E2E exact (thanks to CLAHE OCR).
- **Confidence Sweeping Causes FP Spikes:** Lowering confidence to `0.20` and `0.15` yielded **zero** new true positives, while strictly increasing False Positives. Dropping confidence all the way to `0.10` finally recovered 1 additional plate (15/20 Loc), but caused an unacceptable regression by injecting **16 False Positives** (a 300% increase over the baseline).
- **Inference Size (imgsz):** Reducing the inference size from the native `640` to `320` caused a massive regression in recall, dropping True Positives from 14 to 9. The plate detector fundamentally requires the higher resolution to find the plates within the vehicle crops.

## 4. Conclusion & Promotion Verdict
**Under the tested confidence and inference-size configurations, the six localization misses were not recovered without a substantial false-positive increase. The tested parameter sweep therefore provides insufficient evidence for a plate-detector configuration change, and the frozen `conf=0.25, imgsz=640` configuration should be retained.**

**Verdict:** Do NOT promote any parameter changes from Phase 11C-7. The `conf=0.25, imgsz=640` configuration is retained as the frozen baseline under the tested promotion criteria.
