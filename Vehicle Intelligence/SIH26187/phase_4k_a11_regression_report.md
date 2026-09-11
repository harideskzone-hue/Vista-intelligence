# Phase 4K-A.11 Multi-Condition Regression Benchmark

## Video: `VIDEO-2026-09-09-15-33-40.mp4`

**Overall Improvements:**
- Compute Reduction (Plate YOLO calls): **94.2%**
- FPS Improvement: **370.5%**

| Metric | Original (s1/a0/v1) | Optimized (s6/a10k/v60) |
| :--- | :--- | :--- |
| FPS | 3.69 | **17.36** |
| Plate YOLO calls | 1422 | **83** |
| Plate calls/frame | 3.23 | **0.19** |
| LPR calls | 349 | **23** |
| LPR calls/frame | 0.79 | **0.05** |
| Plate Recall | 100.0% | 100.0% |
| Missed Recognitions | 0 | 0 |
| Correct Confirmations | 1 | 1 |
| False Confirmations | 0 | 0 |
| Avg Confirmation Latency | 10.90s | 11.00s |
| UNKNOWN Count | 20 | 20 |
| Tracks w/ No Evidence | 19 | 19 |
| Short Tracks (<15 frames) | 12 | 12 |
| Long Tracks (>=15 frames) | 9 | 9 |

---
