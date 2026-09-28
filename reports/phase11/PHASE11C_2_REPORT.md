# Phase 11C-2: Vehicle Detector & Tracker Diagnostic

## Objective
Distinguish raw YOLO vehicle detection failures from tracking/suppression failures to determine the root cause of the 19 vehicle-gate misses observed in Phase 11C-1.

## 11C-2A & 2B: Detector vs Tracker Suppression
The 25 text-bearing GT plates were evaluated against the raw YOLOv8n detector (`model.predict()`, stateless) versus the frozen Phase 10E production pipeline (`VehicleANPRPipeline` featuring ByteTrack with `persist=True`).

**Key Finding:** The underlying raw vehicle detector is highly capable, but the tracker is aggressively suppressing valid detections because the static evaluation dataset does not provide the continuous frames required for tracks to mature and confirm.

### Failure Distribution (Conf = 0.40)
Out of 25 text-bearing GT plates:
*   **Raw Detector Failure (YOLO missed vehicle):** 5
*   **Tracker Suppression (YOLO detected, pipeline dropped):** 14
*   **Success at Vehicle Gate (Pipeline retained):** 6

**Conclusion:** 14 out of the 19 failures (74%) in the vehicle gate are caused purely by tracker state suppression (ByteTrack initializing new tracks as unconfirmed and dropping them), not by the detector's inability to see the vehicle. 

## 11C-2C: Confidence Sensitivity
Evaluated the raw detector across varying confidence thresholds `[0.20, 0.30, 0.40, 0.50]`.

| Confidence | Raw Vehicles Detected | Downstream Localization | E2E Exact Match |
| :--- | :--- | :--- | :--- |
| **0.20** | 21 / 25 | 14 | 12 |
| **0.30** | 20 / 25 | 13 | 12 |
| **0.40 (Frozen)** | 20 / 25 | 13 | 12 |
| **0.50** | 19 / 25 | 13 | 12 |

**Conclusion:** 
Lowering the vehicle detector confidence yields negligible improvements. Dropping the threshold all the way to 0.20 recovers only 1 additional vehicle, but does not improve the final E2E exact match rate. The frozen threshold of `0.40` is optimal. 

## 11C-2D: Tracker Isolation (Bypass)
When the tracker is completely bypassed—simulating an environment where tracking does not drop valid first-frame detections—the pipeline's theoretical E2E performance dramatically increases:

*   **Frozen 10E Pipeline Exact Matches:** 5 / 25 (20%)
*   **Tracker-Bypassed Exact Matches:** 12 / 25 (48%)

Of the 20 vehicles successfully detected by raw YOLO:
*   7 failed at crop/localization (plate detector missed them)
*   1 failed at OCR/Validation
*   12 achieved perfect E2E Exact matches.

## Summary
The vehicle detector itself is performing well (20/25 containment). The massive performance degradation (dropping from 12 potential exact matches down to 5) is an artifact of applying a continuous stream tracker (`persist=True`) to an isolated, static image dataset.

**Actionable Insight:** Do not alter the vehicle detector model or confidence threshold (`0.40`). The evaluation bottleneck is the tracker suppression artifact, which confirms we must safely bypass or mock the tracker for dataset-level diagnostic optimization, as tracking logic only properly applies to real video streams.

## Per-Image Diagnostic Table (Conf = 0.40)

| Image | GT Plate | Raw Containment | Tracker Retained | Loc Pass | E2E Exact |
| :--- | :--- | :--- | :--- | :--- | :--- |
| image_0028.jpg | KL34A465 | True | True | True | True |
| image_0028.jpg | KL34A465 | True | True | False | False |
| image_0028.jpg | KL34F | True | True | True | True |
| image_0028.jpg | KL35F4337 | True | True | True | True |
| image_0028.jpg | KL35H5834 | True | True | True | True |
| image_0028.jpg | KL03S6894 | True | True | False | False |
| image_0029.jpg | UP84AE9889 | True | True | True | True |
| image_0030.jpg | GJ01DY6855 | True | True | True | False |
| image_0031.jpg | KL498262 | True | True | True | True |
| image_0032.jpg | WB42AX7446 | True | True | True | True |
| image_0033.jpg | MP07L7524 | True | True | False | False |
| image_0034.jpg | MP04PA0434 | False | False | False | False |
| image_0035.jpg | RJ11GB1829 | True | True | False | False |
| image_0036.jpg | KL41L7001 | True | True | True | True |
| image_0037.jpg | TN58D5353 | True | True | False | False |
| image_0038.jpg | KL07BX7197 | False | False | False | False |
| image_0039.jpg | UP84AE6664 | False | False | False | False |
| image_0040.jpg | KL10AG7249 | False | False | False | False |
| image_0041.jpg | TN58AP5280 | True | True | False | False |
| image_0042.jpg | DL3CD1210 | True | True | True | True |
| image_0043.jpg | RJ11GB8850 | False | False | False | False |
| image_0044.jpg | MP13GA9462 | True | True | True | True |
| image_0045.jpg | KA09C2763 | True | True | True | True |
| image_0046.jpg | MH18AA1002 | True | True | True | True |
| image_0047.jpg | KA01AJ7533 | True | True | False | False |
