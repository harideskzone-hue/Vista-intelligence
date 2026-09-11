# Vista Intelligence — AI Border & Perimeter Surveillance System

An end-to-end AI surveillance platform for SIH 2026 (Problem Statement 26187):
- 🧠 **Face Recognition** — enrollment, real-time matching, temporal confirmation
- 🚧 **Border / Perimeter Detection** — virtual line crossing, zone entry/exit, intrusion alerts
- 🚗 **Vehicle Intelligence** — ANPR (plate detection + OCR), vehicle classification & tracking
- 📊 **Live Dashboard** — real-time feeds, event log, evidence management

---

## ⚡ Quick Start (after cloning)

### 1. System prerequisite
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg
```

### 2. Python environment
```bash
cd face_api
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Start the server
```bash
# From the face_api/ directory (with venv active)
python run.py
```

Dashboard opens at → **http://localhost:5001/**

### Or use the full auto-launcher (macOS/Linux)
```bash
bash dev_start.sh
```
The launcher handles venv creation, dependency install, model warmup, and startup automatically.

---

## 🗂 Repository Structure

```
Vista-intelligence/
│
├── requirements.txt              # Combined requirements for all modules
├── dev_start.sh                  # Auto-launcher (macOS/Linux)
│
├── face_api/                     # FastAPI backend (main web server)
│   ├── run.py                    # Server entry point → http://localhost:5001
│   ├── requirements.txt          # Backend-only requirements
│   └── app/
│       ├── api/
│       │   ├── routes.py         # Face enrollment, search, camera management
│       │   ├── boundary_routes.py# Border detection API endpoints
│       │   ├── vehicle_routes.py # ANPR & vehicle intelligence API
│       │   └── event_routes.py   # Event & clip retrieval
│       ├── services/face_service.py
│       └── core/                 # encoder, vector_db, storage
│   └── templates/
│       ├── dashboard.html        # Live feed + event dashboard
│       ├── boundary.html         # 🚧 Border detection UI (canvas line drawing)
│       ├── database.html         # Identity enrollment & management
│       ├── vehicle.html          # Vehicle intelligence dashboard
│       ├── preview.html          # Live camera preview
│       └── wanted.html           # Watchlist / wanted persons
│
├── edge/                         # Edge AI intelligence modules
│   ├── boundary/
│   │   ├── adapter.py            # YOLO + ByteTrack → TrackObservation adapter
│   │   ├── crossing_engine.py    # Virtual line crossing state machine
│   │   ├── geometry.py           # Point-segment distance, side-of-line math
│   │   ├── anomaly_engine.py     # Rule-based anomaly detection
│   │   └── manager.py            # BoundaryManager orchestrator (singleton)
│   ├── events/
│   │   ├── event_hub.py          # Central event logging
│   │   └── clip_manager.py       # H.264 clip extraction via ffmpeg
│   ├── face/                     # Face recognition pipeline
│   └── vehicle/                  # Vehicle detection & ANPR
│
├── face_engine/                  # Live camera scorer daemon
│   ├── live_scorer.py
│   ├── model_manager.py
│   └── models/                   # Pre-trained model weights (included)
│       ├── buffalo_sc/           # InsightFace face recognition models
│       ├── yolo26n-face.pt       # YOLO face detector
│       └── face_landmarker.task  # MediaPipe face landmarks
│
├── Vehicle Intelligence/SIH26187/
│   ├── anpr/                     # Automatic Number Plate Recognition
│   │   ├── detector.py           # Plate detection (YOLO)
│   │   ├── ocr.py                # Plate OCR (fast-plate-ocr)
│   │   ├── temporal_voter.py     # Multi-frame plate voting
│   │   └── models/best.pt        # ANPR YOLO model weights
│   ├── vehicle_ai/               # Vehicle classifier + tracker
│   ├── integration/              # Combined ANPR + tracking pipeline
│   └── data/fast_plate_ocr/models/ # Fine-tuned OCR model weights
│
├── yolo11n.pt                    # YOLO11n vehicle detection model
├── chokepoint-bbs/               # Benchmark ground-truth annotations
└── tests/                        # Full test suite (19 test files)
```

---

## 🌐 Web Pages

| URL | Page | Description |
|-----|------|-------------|
| `http://localhost:5001/` | Dashboard | Live feed, recognition events |
| `http://localhost:5001/boundary` | Border Detection | Draw virtual border, upload test video |
| `http://localhost:5001/database` | Database | Enroll & manage identities |
| `http://localhost:5001/vehicle` | Vehicle | ANPR live feed & plate logs |
| `http://localhost:5001/preview` | Live Preview | Multi-camera live view |
| `http://localhost:5001/wanted` | Watchlist | Wanted persons management |

---

## 🚧 Border Detection Module

The border/perimeter detection system works as follows:

1. **Draw a virtual border line** directly on the video preview using the canvas UI
2. **Upload a surveillance video** for offline testing, or connect a live camera
3. The system runs **YOLO + ByteTrack** multi-object tracking on every frame
4. **Line crossings** are detected by tracking each person's foot-point (bottom-center of bbox)
5. Events are classified as **INTRUDING** (safe→restricted) or **RETREATING** (restricted→safe)
6. **Clips** are automatically cut around each crossing event (pre + event + post) and encoded as H.264

### Border Detection API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/boundary/line` | Set virtual border line (normalized coords) |
| POST | `/api/boundary/test-video` | Upload video for offline detection |
| GET | `/api/boundary/events` | Get all boundary crossing events |
| GET | `/api/events/{id}/clip` | Download a specific event clip |

---

## 🔑 AI Models Included

| Model | Location | Purpose |
|-------|----------|---------|
| `yolo11n.pt` | root + `Vehicle Intelligence/` | Vehicle detection |
| `yolo26n-face.pt` | `face_engine/models/` | Face detection |
| `det_500m.onnx` | `face_engine/models/buffalo_sc/` | Face landmark detection |
| `w600k_mbf.onnx` | `face_engine/models/buffalo_sc/` | Face recognition embeddings |
| `face_landmarker.task` | `face_engine/models/` | MediaPipe face landmarks |
| `anpr/models/best.pt` | `Vehicle Intelligence/SIH26187/anpr/models/` | Plate detection |
| `cct_xs_v2_global.onnx` | `Vehicle Intelligence/.../fast_plate_ocr/models/` | Plate OCR |
| Fine-tuned OCR models | `Vehicle Intelligence/.../fine_tuned/` | Custom OCR weights |

---

## 📦 Dependencies

```bash
# Python packages
pip install -r requirements.txt

# System packages (required for clip encoding)
brew install ffmpeg     # macOS
apt install ffmpeg      # Ubuntu
```

---

## 🧪 Running Tests

```bash
cd face_api && source .venv/bin/activate
python -m pytest ../tests/ -v
```
