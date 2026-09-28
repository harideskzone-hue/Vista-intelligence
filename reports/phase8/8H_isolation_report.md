# Phase 8H: Detection vs Association Isolation

## 1. Tracker Suppression Gap (Recall & Precision)
| Sampling | GT Objects | Raw Recall | Tracked Recall | **Suppression Gap** | Raw Precision | Tracked Precision | **Precision Gap** |
|---|---|---|---|---|---|---|---|
| Native | 112297 | 0.2800 | 0.2678 | **+0.0122** | 0.9040 | 0.9115 | **-0.0074** |
| 5 | 20426 | 0.2990 | 0.2417 | **+0.0573** | 0.8944 | 0.9011 | **-0.0067** |
| 2 | 8163 | 0.2978 | 0.1976 | **+0.1002** | 0.8957 | 0.8921 | **+0.0036** |
| 1.2 | 4939 | 0.2944 | 0.1697 | **+0.1247** | 0.9076 | 0.8896 | **+0.0180** |
| 1 | 4090 | 0.2966 | 0.1545 | **+0.1421** | 0.8906 | 0.8864 | **+0.0042** |

## 2. Tracking Continuity Degradation
| Sampling | MOTA | IDF1 | IDSW / 1K GT |
|---|---|---|---|
| Native | 0.2382 | 0.3335 | 3.58 |
| 5 | 0.1974 | 0.2967 | 17.72 |
| 2 | 0.1338 | 0.2012 | 39.94 |
| 1.2 | 0.1101 | 0.1694 | 38.47 |
| 1 | 0.0985 | 0.1674 | 36.19 |

## 3. Resource Utilization
| Sampling | Evaluated Frames | Processing FPS | Wall-Clock Runtime (s) | Average CPU (%) | Peak CPU (%) | Peak RSS (MB) |
|---|---|---|---|---|---|---|
| Native | 5316 | 21.6 | 246.5 | 63.7% | 93.4% | 678.6 |
| 5 | 1071 | 22.3 | 48.1 | 63.5% | 77.7% | 860.8 |
| 2 | 429 | 22.3 | 19.3 | 63.5% | 78.9% | 840.6 |
| 1.2 | 258 | 20.7 | 12.5 | 62.1% | 76.4% | 581.7 |
| 1 | 215 | 20.4 | 10.5 | 60.1% | 77.0% | 557.5 |
