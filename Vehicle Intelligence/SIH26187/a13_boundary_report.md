# A.13 Camera/Optical Boundary Analysis Report

> [!IMPORTANT]
> **Ground Truth Available**: NO (Empirical observational distributions only)
> **Accuracy Boundary**: NOT ESTABLISHED (Cannot prove impossibility without manual plate annotations)
> **Operational Evidence Boundary**: PRELIMINARY (Multi-dimensional evidence rather than hard single-variable thresholds)

## 1. Benchmark Condition Coverage Status

| Condition | Video File | Status | Observations | Notes |
| :--- | :--- | :---: | ---: | :--- |
| 1. Day / Urban 4K | `pexels-george-morina-6719160 (2160p)` | **COVERED** | 580 | High-resolution daylight urban traffic |
| 2. Night / Low-Light | `night_driving` | **UNAVAILABLE / EXCLUDED** | 0 | No decodable low-light traffic video present on system |
| 3. Highway / Fast Motion | `highway_fast` | **COVERED** | 207 | Motion blur, high-speed vehicle approaches |
| 4. Far / CCTV / Small | `far_vehicles` | **COVERED** | 1,938 | Distant traffic, wide-angle CCTV surveillance |

## 2. Per-Condition Observational Summary

| Condition | Crop Count | Median WxH | Median Area | Median Sharpness | Valid Format % |
| :--- | ---: | :--- | ---: | ---: | ---: |
| pexels-george-morina-6719160 (2160p) | 580 | 202x65 | 13267 px | 100 | 18.6% |
| highway_fast | 207 | 113x54 | 5992 px | 2350 | 8.7% |
| far_vehicles | 1938 | 135x43 | 5544 px | 189 | 7.5% |

## 3. Pixel Area Distribution & Readability

Empirical analysis of crop bounding box area vs. valid Indian registration format detection:

| Pixel Area | Crop Count | Median Conf | Median Sharpness | Valid Indian Format % |
| :--- | ---: | ---: | ---: | ---: |
| < 200 px | 9 | 0.15 | 732 | 0.0% |
| 200-400 px | 21 | 0.14 | 483 | 9.5% |
| 400-800 px | 111 | 0.15 | 456 | 9.0% |
| 800-1500 px | 203 | 0.17 | 263 | 6.4% |
| > 1500 px | 2381 | 0.46 | 158 | 10.3% |

> [!NOTE]
> **Non-Monotonicity Finding**: Valid format percentage does NOT increase monotonically with area (200-400 px is 9.5%, 400-800 px is 9.0%, 800-1500 px is 6.4%, >1500 px is 10.3%).
> Therefore, a simple cutoff like `min_area = 1000 px` is **scientifically indefensible** as a guarantee of readability. Only `< 200 px` shows 0.0% validity, indicating extreme small crops yield very weak optical information.

## 4. Sharpness (Laplacian Variance) Distribution

Empirical analysis of image sharpness (Laplacian variance) vs. valid format detection:

| Sharpness (Var) | Crop Count | Median Area | Valid Indian Format % |
| :--- | ---: | ---: | ---: |
| < 100 (Blurry) | 704 | 7811 px | 12.8% |
| 100-300 | 1320 | 7444 px | 8.6% |
| 300-600 | 398 | 4277 px | 10.1% |
| 600-1000 | 91 | 1708 px | 13.2% |
| > 1000 (Sharp) | 212 | 4843 px | 7.5% |

> [!WARNING]
> **Sharpness Pitfall**: The highest sharpness bucket (`> 1000`) exhibits the *lowest* valid format percentage (7.5%), while blurry crops (`< 100`) have 12.8%.
> Laplacian variance is heavily affected by high-frequency background noise, bumper textures, and sensor artifacts. Sharpness must NOT be used as a hard single-variable filter.

## 5. Architectural Recommendations for A.14 Production Quality Gate

Based on the empirical evidence, the A.14 production pipeline must adhere to the following principles:

1. **No Artificial Single-Variable Cutoffs**: `plate_min_area`, `plate_min_sharpness`, and `plate_min_contrast` are marked as **NOT ESTABLISHED**.
2. **Fail-Closed Multi-Dimensional Evidence Assessment**: The quality gate collects geometry (`width`, `height`, `area`, `aspect_ratio`), sharpness (`Laplacian`), and contrast (`RMS std`), applying physical sanity checks (e.g. rejection of microscopic `< 100 px` crops or degenerate aspect ratios) without fabricating certainty.
3. **Separation of Concerns**: Valid Indian regex format is NOT synonymous with plate correctness. The system explicitly separates:
   - `plate_detected`: presence of bounding box
   - `optical_quality`: raw metrics (area, sharpness, contrast)
   - `ocr_text`: raw string prediction from FastPlateOCR
   - `ocr_confidence`: model probability distribution
   - `format_valid`: regex conformance
   - `temporal_support`: corroborating observation count across track history
   - `identity_status`: `NO_PLATE_DETECTED`, `INSUFFICIENT_RESOLUTION`, `LOW_QUALITY`, `OCR_UNCERTAIN`, `PROBABLE`, `CONFIRMED`
4. **Frozen A.10 Scheduling**: Keep `plate_stride = 6`, `min_vehicle_area = 10000`, and `verification_stride = 60` frozen as vehicle-crop scheduling optimizations (not plate optical limits).