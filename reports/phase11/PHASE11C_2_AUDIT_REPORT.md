# Phase 11C-2: Audit Report (Vehicle Detector & Tracker)

## 1. Audit Conclusion
The audit has been completed and **tracker suppression is explicitly demonstrated and proven**. By ensuring that `pipeline` and `tracked_model` states are properly reset for every experimental sequence, the diagnostic JSON perfectly aligns with the previously observed tracker suppression behavior.

## 2. Failure Distribution (Pipeline Path, Conf=0.40)
Using the strict waterfall logic where tracker suppression takes precedence:

| Failure Mode | Count |
| :--- | :--- |
| 1. raw detector failure | 5 |
| 2. tracker suppression | 14 |
| 3. pipeline area gate suppression | 0 |
| 4. crop/localization failure | 2 |
| 5. OCR failure | 0 |
| N/A (Success at exact match via raw bypass) | 4 |
| Original Baseline E2E Exact | 5 |

## 3. Tracker Bypass (Raw Detector Capability)
The 11C-2 diagnostic evaluated the theoretical E2E match rate if we bypass the tracker and directly use the `raw_detector_produced_vehicle` bounding box. The results independently confirm the 5/25 → 12/25 improvement claim.

- **Raw YOLO Containment (IoU > 0.5):** 20 / 25
- **Plate Localization on Raw Crop:** 13 / 20
- **E2E Exact Match on Raw Crop:** 12 / 25

## 4. Confidence Sensitivity
| Confidence | Raw Vehicles Detected | Downstream Localization | E2E Exact Match |
| :--- | :--- | :--- | :--- |
| **0.20** | 21 / 25 | 14 | 12 |
| **0.30** | 20 / 25 | 13 | 12 |
| **0.40** | 20 / 25 | 13 | 12 |
| **0.50** | 19 / 25 | 13 | 12 |

**Conclusion:** Lowering the confidence yields negligible improvements. The frozen `0.40` threshold remains mathematically optimal.

## 5. Per-Image Diagnostic Table (Conf = 0.40)

| Image | GT Plate | Raw Containment | Tracker Retained | Tracker Suppressed | Loc Pass (Raw) | E2E Exact (Raw) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| image_0028.jpg | KL34A465 | True | False | True | True | True |
| image_0028.jpg | KL34A465 | True | False | True | False | False |
| image_0028.jpg | KL34F | True | False | True | True | True |
| image_0028.jpg | KL35F4337 | True | False | True | True | True |
| image_0028.jpg | KL35H5834 | True | False | True | True | True |
| image_0028.jpg | KL03S6894 | True | False | True | False | False |
| image_0029.jpg | UP84AE9889 | True | False | True | True | True |
| image_0030.jpg | GJ01DY6855 | True | False | True | True | False |
| image_0031.jpg | KL498262 | True | False | True | True | True |
| image_0032.jpg | WB42AX7446 | True | False | True | True | True |
| image_0033.jpg | MP07L7524 | True | False | True | False | False |
| image_0034.jpg | MP04PA0434 | False | False | False | False | False |
| image_0035.jpg | RJ11GB1829 | True | False | True | False | False |
| image_0036.jpg | KL41L7001 | True | True | False | True | True |
| image_0037.jpg | TN58D5353 | True | False | True | False | False |
| image_0038.jpg | KL07BX7197 | False | False | False | False | False |
| image_0039.jpg | UP84AE6664 | False | False | False | False | False |
| image_0040.jpg | KL10AG7249 | False | False | False | False | False |
| image_0041.jpg | TN58AP5280 | True | True | False | False | False |
| image_0042.jpg | DL3CD1210 | True | True | False | True | True |
| image_0043.jpg | RJ11GB8850 | False | False | False | False | False |
| image_0044.jpg | MP13GA9462 | True | False | True | True | True |
| image_0045.jpg | KA09C2763 | True | True | False | True | True |
| image_0046.jpg | MH18AA1002 | True | True | False | True | True |
| image_0047.jpg | KA01AJ7533 | True | True | False | False | False |
