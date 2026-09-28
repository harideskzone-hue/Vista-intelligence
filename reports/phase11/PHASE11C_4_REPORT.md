# Phase 11C-4: OCR Optimization Report

## 1. Objective
Evaluate OCR-level preprocessing interventions to determine if text readability can be improved on the 14 successfully localized plate crops obtained from the 10% vehicle-padding configuration. The upstream models and configurations (YOLO11n, vehicle conf=0.40, best.pt, plate conf=0.25) remain strictly frozen.

## 2. OCR Preprocessing Variants
The 14 localized BGR plate crops were evaluated against 5 preprocessing configurations before being passed to the `FastPlateOCR` engine:

| Preprocessing | Evaluated Crops | Exact Matches | CER | Char Accuracy | Invalid Fmt |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **original** | 14 | **13 / 14** | 0.0071 | 0.9929 | 1 |
| **grayscale** | 14 | **13 / 14** | 0.0071 | 0.9929 | 1 |
| **contrast** | 14 | **14 / 14** | 0.0000 | 1.0000 | 1 |
| **upscale** | 14 | **13 / 14** | 0.0071 | 0.9929 | 1 |
| **sharpen** | 14 | **10 / 14** | 0.0383 | 0.9617 | 3 |

## 3. Key Findings
- **Baseline (original)**: 13/14 exact matches.
- **Contrast Normalization (CLAHE)**: **14/14 exact matches (100% accuracy)**. By applying Contrast Limited Adaptive Histogram Equalization on the Lightness (L) channel, the OCR engine successfully recovered the single character missed in the original crop. CER dropped to an optimal `0.0000`.
- **Sharpening**: Aggressive unsharp masking degraded performance (10/14), likely amplifying compression artifacts or noise.
- **Grayscale & Upscaling**: Provided no measurable improvement over the baseline.

*Note: The 1 `invalid_format` flag observed across the highest-performing configurations is due to the ground truth text itself (`KL34F`) not adhering to the strict Indian format regex, despite being perfectly decoded by the OCR engine.*

## 4. Current E2E Diagnostic Trajectory (Static Dataset)
By sequentially optimizing isolated pipeline components (bypassing the tracker suppression), the diagnostic E2E Exact Match ceiling has been raised:

1. **11C-2**: 12/25 (Raw YOLO detector bypass)
2. **11C-3**: 13/25 (+1 via 10% vehicle crop padding)
3. **11C-4**: **14/25** (+1 via CLAHE contrast normalization)

## 5. Next Steps
The OCR engine is highly capable (100% accuracy on properly localized plates with contrast normalization). The absolute bottleneck defining the 14/25 ceiling is the **Plate Detector** (`best.pt`), which natively missed 6 well-contained plates, and the **Raw Vehicle Detector** (`yolo11n.pt`), which natively missed 5 vehicles. Both of these are upstream of the OCR.
