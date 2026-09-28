# Phase 8I-B: `track_buffer` Sweep Summary (match_thresh=0.9)

### 1.2 FPS Candidate Comparison
| Config (buffer) | Raw Recall | Tracked Recall | Suppression Gap | IDF1 | IDSW/1K | Precision | MT/ML |
|---|---|---|---|---|---|---|---|
| 3 frames | 0.2944 | 0.1707 | **+0.1237** | 0.1766 | 33.00 | 0.8673 | 26/375 |
| 5 frames | 0.2944 | 0.1731 | **+0.1213** | 0.1725 | 33.81 | 0.8689 | 28/373 |
| 10 frames | 0.2944 | 0.1814 | **+0.1130** | 0.1782 | 36.65 | 0.8784 | 30/370 |
| 30 frames | 0.2944 | 0.1838 | **+0.1105** | 0.1730 | 39.48 | 0.8756 | 32/366 |

### Native Regression Guard Comparison
| Config (buffer) | Tracked Recall | IDF1 | IDSW/1K | Precision | MOTA | MT/ML |
|---|---|---|---|---|---|---|
| 3 frames | 0.2669 | 0.2904 | 4.43 | 0.9116 | 0.2366 | 69/306 |
| 5 frames | 0.2675 | 0.3056 | 3.95 | 0.9103 | 0.2372 | 69/306 |
| 10 frames | 0.2685 | 0.3214 | 3.37 | 0.9091 | 0.2383 | 69/304 |
| 30 frames | 0.2703 | 0.3421 | 3.19 | 0.9088 | 0.2399 | 71/304 |
