# SIH26187 — Core Pipeline

AI-powered photo booth system: camera → detection → scoring → database → dashboard.

## Quick Start

```bash
bash dev_start.sh
```

That's it. The launcher handles everything automatically:
1. ✅ Python 3.10–3.12 check
2. ✅ Disk space check (≥ 2GB)
3. ✅ Port conflict detection (5001)
4. ✅ Virtual environment creation
5. ✅ Dependency installation
6. ✅ AI model validation + auto-download
7. ✅ YOLO + InsightFace warmup
8. ✅ FastAPI backend startup (port 5001)
9. ✅ Camera pipeline launch
10. ✅ Dashboard opens at http://localhost:5001/
11. ✅ Clean shutdown on Ctrl+C

### Optional flags

```bash
bash dev_start.sh --debug       # Show bounding boxes + confidence overlays
bash dev_start.sh --safe-mode   # Skip InsightFace (lightweight mode)
bash dev_start.sh --skip-warmup # Faster startup, slower first inference
```

---

## Repository Structure

```
SIH26187/
├── dev_start.sh              # Single-command launcher
├── cameras.json              # Camera configuration
├── pipeline_test.py          # End-to-end test suite
├── pyrightconfig.json        # VS Code type checking config
│
├── face_api/              # FastAPI backend
│   ├── run.py                # Server entry point
│   ├── requirements.txt
│   └── app/
│       ├── api/routes.py     # All endpoints (incl. /api/debug/last_upload)
│       ├── services/face_service.py
│       ├── core/             # encoder, vector_db, storage, image_utils
│       └── config.py
│
└── face_engine/                # AI camera pipeline
    ├── live_scorer.py        # Camera stream + YOLO detection
    ├── model_manager.py      # Model registry + SHA256 download
    ├── requirements.txt
    └── models/               # AI model weights (auto-downloaded)
```

---

## API Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/api/health` | Public | Health check + person count |
| GET | `/api/debug/last_upload` | Public | Last upload metadata |
| POST | `/api/add` | Token | Add image to database |
| POST | `/api/search` | Session | Search by face |
| GET | `/api/search` | Session | Search by ID or name |
| POST | `/api/batch_add` | Token | Bulk add images |
| GET | `/` | Session | Dashboard |

---

## Dependencies

**Backend** (`face_api/requirements.txt`):
- `fastapi`, `uvicorn`, `insightface`, `onnxruntime`, `faiss-cpu`

**Pipeline** (`face_engine/requirements.txt`):
- `ultralytics`, `mediapipe==0.10.14`, `opencv-python`, `numpy`, `scipy`

---

## AI Models

Auto-downloaded on first run to `face_engine/models/`:

| Model | Size | Purpose |
|-------|------|---------|
| `yolo26n-face.pt` | 5.6 MB | Face detection |

---

## Observability

- **Logs**: `~/Library/Application Support/MagicClick/logs/` (`api.log`, `scorer.log`, `worker.log`)
- **Failed uploads**: `failed_uploads.json` in the output directory (auto-retried)
- **Debug endpoint**: `GET http://localhost:5001/api/debug/last_upload`
- **Preflight table**: Printed on every startup

---

## Running Tests

```bash
~/Library/Application\ Support/MagicClick/.venv/bin/python pipeline_test.py
```
