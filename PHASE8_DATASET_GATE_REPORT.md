# Phase 8.0 Dataset Acquisition & Protocol Gate Report

This report definitively proves whether the evaluation inputs are real, valid, licensed/authorized, and frozen.

| Requirement | Status | Evidence |
|---|---|---|
| Phase 7 baseline frozen | PASS | git_tag: phase7-final-validated |
| Production config frozen | PASS | phase8_protocol.json generated |
| Person detector weights frozen | PASS | SHA256: 0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1 |
| Face detector weights frozen | PASS | SHA256: c6a5405127a2e351292315a6a8084ea3e790dbec25b9d16a8e80d1e3f866efe1 |
| Face recognizer weights frozen | BLOCKED | Decoupled External API |
| ANPR weights frozen | BLOCKED | No ANPR in production code |
| YOLO threshold frozen | PASS | yolo_confidence: 0.5 |
| Tracker config frozen | PASS | tracker_correlation: 0.65 |
| Input resolution frozen | PASS | 480 (from edge/boundary/adapter.py imgsz=480) |
| Inference device frozen | PASS | MPS (CentralInferenceWorker) |
| ChokePoint access authorized | BLOCKED | 403 Forbidden. Authorization required via Zenodo API. |
| MOT17 sequences downloaded | PASS |  |
| MOT17 GT validated | PASS |  |
| ANPR dataset source confirmed | BLOCKED | Alternative source (HuggingFace zenitsu09) lacks explicit license |
| ANPR bounding boxes & text validated | BLOCKED | Dataset acquisition blocked by licensing constraints |
| ANPR properly licensed | BLOCKED | Kaggle requires auth; HF alternatives are unlicensed |
| Boundary scenarios recorded | BLOCKED | Manual physical recording required |
| Acceptance Criteria defined | BLOCKED | User must define targets before 8A-8G |

> [!IMPORTANT]
> **Conclusion:** Execution of evaluators 8A–8G is strictly **PROHIBITED** until all BLOCKED items are legally acquired, validated, and transitioned to PASS.