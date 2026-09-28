# Phase 8G: Sampling Degradation Study

## 1. Methodology
- **Dataset**: 7 unique MOT17 underlying sequences. (Sanity check verified that DPM/FRCNN/SDP subsets are fully redundant as our pipeline ignores MOTChallenge `det.txt` boxes).
- **Tracker**: `yolo11n.pt` + `bytetrack.yaml` (Production Configuration Frozen)
- **Temporal Sparsity**: Tracker state is NOT reset between sampled frames. Interstitial frames are entirely skipped. Time-based fractional sampling is used (e.g. 1.2 target FPS).
- **Evaluation Focus**: Measuring operational degradation caused strictly by temporal sparsity to identify where continuity breaks down.

## 2. Overall Degradation Results

### Tracking Continuity (Relative to Native)
| Sampling | MOTA | Δ MOTA | IDF1 | Δ IDF1 | IDSW / 1K GT | Δ IDSW / 1K GT | Mostly Tracked | Mostly Lost |
|---|---|---|---|---|---|---|---|---|
| Native | 0.2382 | +0.0000 | 0.3335 | +0.0000 | 3.58 | +0.00 | 65 | 307 |
| 15 | 0.2410 | +0.0028 | 0.3343 | +0.0008 | 6.65 | +3.07 | 63 | 306 |
| 10 | 0.2308 | -0.0075 | 0.3286 | -0.0049 | 9.43 | +5.85 | 57 | 317 |
| 5 | 0.1974 | -0.0408 | 0.2967 | -0.0368 | 17.72 | +14.14 | 40 | 347 |
| 2 | 0.1338 | -0.1045 | 0.2012 | -0.1323 | 39.94 | +36.36 | 33 | 358 |
| 1.2 | 0.1101 | -0.1281 | 0.1694 | -0.1642 | 38.47 | +34.89 | 27 | 369 |
| 1 | 0.0985 | -0.1397 | 0.1674 | -0.1661 | 36.19 | +32.61 | 25 | 384 |

### Normalized Detection Metrics
| Sampling | GT Evaluated | TP | FP | FN | Precision | Recall | FP Rate | FN Rate |
|---|---|---|---|---|---|---|---|---|
| Native | 112297 | 30077 | 2921 | 82220 | 0.9115 | 0.2678 | 0.0260 | 0.7322 |
| 15 | 60768 | 16746 | 1696 | 44022 | 0.9080 | 0.2756 | 0.0279 | 0.7244 |
| 10 | 40836 | 10931 | 1123 | 29905 | 0.9068 | 0.2677 | 0.0275 | 0.7323 |
| 5 | 20426 | 4936 | 542 | 15490 | 0.9011 | 0.2417 | 0.0265 | 0.7583 |
| 2 | 8163 | 1613 | 195 | 6550 | 0.8921 | 0.1976 | 0.0239 | 0.8024 |
| 1.2 | 4939 | 838 | 104 | 4101 | 0.8896 | 0.1697 | 0.0211 | 0.8303 |
| 1 | 4090 | 632 | 81 | 3458 | 0.8864 | 0.1545 | 0.0198 | 0.8455 |

### Sampling Performance & Runtime
| Target FPS | Total Evaluated Frames | Effective Processing FPS | Wall-Clock Runtime (s) |
|---|---|---|---|
| Native | 5316 | 35.8 | 148.4 |
| 15 | 3152 | 36.9 | 85.4 |
| 10 | 2141 | 37.1 | 57.7 |
| 5 | 1071 | 37.4 | 28.6 |
| 2 | 429 | 36.3 | 11.8 |
| 1.2 | 258 | 37.0 | 7.0 |
| 1 | 215 | 37.7 | 5.7 |
