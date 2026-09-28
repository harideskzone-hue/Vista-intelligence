# Phase 11C-3: Crop-Quality Optimization Report

## 1. Objective
Evaluate whether vehicle-crop geometry (specifically, padding) and downstream plate-crop acceptance rules are limiting ANPR performance after a valid vehicle bounding box is secured. The experiment builds directly upon the 12/25 raw-detector-bypass E2E baseline established in 11C-2.

## 2. Plate-Crop Acceptance Rules Diagnostic
For all padding configurations tested, **every single localized plate crop successfully passed the quality acceptance rules** (minimum dimension, minimum area, aspect ratio checks). There were **0 rejections** due to crop-quality sanity bounds. 

## 3. Crop Geometry Optimization (Padding Variants)
The 20 successfully detected vehicles were evaluated across four padding levels (expanding the crop by P% of vehicle width/height in all directions).

| Padding | Loc Count | Acc Crops | OCR | E2E Exact | Invalid Fmt | CER |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **0%** | 13 / 20 | 13 | 13 | **12 / 25** | 1 | 0.0077 |
| **5%** | 13 / 20 | 13 | 13 | **11 / 25** | 2 | 0.0182 |
| **10%** | 14 / 20 | 14 | 14 | **13 / 25** | 1 | 0.0071 |
| **15%** | 13 / 20 | 13 | 13 | **12 / 25** | 1 | 0.0077 |

**Key Findings:**
- The baseline (0% padding) perfectly reproduces the 11C-2 bypass performance of 13 localizations and 12 exact matches.
- **10% padding** recovers 1 additional plate during the localization phase, increasing the E2E exact match rate from 12/25 to **13/25**.
- Excessive padding (15%) reintroduces background noise, causing localization regressions.

*Note: The reported Average CER (Character Error Rate) is calculated **exclusively over the localized and accepted plate crops** (where an OCR string is produced). It does not include `CER=1.0` penalties for detector misses, as the goal is to evaluate OCR-level readability on successful crops.*

## 4. Dominant Loss Identification
The primary ANPR breakdown at this stage is exclusively **vehicle crop geometry → plate detector**.

- 20 vehicles successfully enter this phase.
- Among the tested padding values (0%, 5%, 10%, 15%), **10% padding produced the highest observed plate-localization count (14) and raw-bypass E2E exact count (13)** on this 25-sample diagnostic subset.
- Therefore, the plate detector (`best.pt` at conf=0.25) is natively missing 6 plates even when provided the highest-performing padded vehicle crop. 

## 5. Per-Image Detailed Results (10% Padding Configuration)

| Image | GT Plate | Raw Containment | Loc Success | Rejection Reason | OCR Output | Exact Match |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| image_0028.jpg | KL34A465 | True | True | None | KL34A465 | True |
| image_0028.jpg | KL34A465 | True | False | plate_detector_miss | None | False |
| image_0028.jpg | KL34F | True | True | None | KL34F | True |
| image_0028.jpg | KL35F4337 | True | True | None | KL35F4337 | True |
| image_0028.jpg | KL35H5834 | True | True | None | KL35H5834 | True |
| image_0028.jpg | KL03S6894 | True | False | plate_detector_miss | None | False |
| image_0029.jpg | UP84AE9889 | True | True | None | UP84AE9889 | True |
| image_0030.jpg | GJ01DY6855 | True | True | None | GJ01DY6855 | True |
| image_0031.jpg | KL498262 | True | True | None | KL498262 | True |
| image_0032.jpg | WB42AX7446 | True | True | None | WB42AX7446 | True |
| image_0033.jpg | MP07L7524 | True | False | plate_detector_miss | None | False |
| image_0034.jpg | MP04PA0434 | False | False | raw_detector_miss | None | False |
| image_0035.jpg | RJ11GB1829 | True | False | plate_detector_miss | None | False |
| image_0036.jpg | KL41L7001 | True | True | None | KL41L7001 | True |
| image_0037.jpg | TN58D5353 | True | False | plate_detector_miss | None | False |
| image_0038.jpg | KL07BX7197 | False | False | raw_detector_miss | None | False |
| image_0039.jpg | UP84AE6664 | False | False | raw_detector_miss | None | False |
| image_0040.jpg | KL10AG7249 | False | False | raw_detector_miss | None | False |
| image_0041.jpg | TN58AP5280 | True | True | None | TN58AP5280 | True |
| image_0042.jpg | DL3CD1210 | True | True | None | DL3CD1210 | True |
| image_0043.jpg | RJ11GB8850 | False | False | raw_detector_miss | None | False |
| image_0044.jpg | MP13GA9462 | True | True | None | MP13GA9462 | True |
| image_0045.jpg | KA09C2763 | True | True | None | KA09C2763 | True |
| image_0046.jpg | MH18AA1002 | True | True | None | MH48AA1002 | False |
| image_0047.jpg | KA01AJ7533 | True | False | plate_detector_miss | None | False |
