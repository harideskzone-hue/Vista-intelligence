# Face Embedding Database - Documentation

A CPU-based anonymous face clustering database with REST API (FastAPI).

## Features

- **Face Detection & Encoding** - InsightFace (buffalo_l model)
- **Vector Search** - FAISS for similarity matching
- **Image Storage** - MozJPEG compression (quality 85)
- **Person Naming** - Assign names, search by name
- **Anonymous Clustering** - Auto-match faces to existing persons
- **FastAPI Powered** - High performance, Type safety, Automatic Docs

---

## API Endpoints

**Interactive Documentation:**
Go to `http://localhost:5000/docs` or `http://localhost:5000/redoc` for interactive Swagger UI.

### 1. `/api/search` - Unified Search

Search by face, ID, or name.

```
GET /api/search?id=<person_id>     # By ID
GET /api/search?name=<name>        # By name
POST /api/search
{"img": "base64", "cropped": false} # By face
```

**Response:**
```json
{
  "success": true,
  "person_id": "abc123",
  "name": "John",
  "similarity": 0.95,
  "images": ["base64...", "base64..."]
}
```

---

### 2. `/api/add` - Add Image

Add a new image. Auto-matches to existing person or creates new.

```
POST /api/add
Content-Type: application/json

{
  "img": "<base64>",
  "score": 8.5
}
```

**Or form-data:**
- `image`: File
- `score`: Number

**Response:**
```json
{
  "success": true,
  "action": "created_new" | "added_to_existing",
  "person_id": "abc123",
  "image_count": 1
}
```

---

### 3. `/api/name` - Assign Name

Assign a name to a person.

```
POST /api/name
Content-Type: application/json

{
  "person_id": "abc123",
  "name": "John"
}
```

**Or form-data:**
- `person_id`: Text
- `name`: Text

---

### 4. `/api/topn` - Top 3 Images

Get top 3 images globally by score.

```
GET /api/topn
```

**Response:**
```json
{
  "success": true,
  "images": [
    {"id": "...", "person_id": "...", "score": 9.5, "image": "base64..."}
  ]
}
```

---

### 5. `/api/health` - Health Check

```
GET /api/health
```

**Response:**
```json
{
  "status": "healthy",
  "person_count": 5,
  "vector_count": 5
}
```

---

### 6. `/api/person/{id}` - Get/Delete Person

```
GET /api/person/{id}      # Get info
DELETE /api/person/{id}   # Delete person
```

---

## Project Structure

```
face_api/
├── app/
│   ├── api/
│   │   └── routes.py      # FastAPI endpoints
│   ├── core/
│   │   ├── encoder.py     # InsightFace wrapper
│   │   ├── vector_db.py   # FAISS operations
│   │   └── storage.py     # Image storage + SQLite
│   ├── templates/
│   │   ├── dashboard.html # Main dashboard
│   │   ├── camera.html    # Camera & search view
│   │   └── upload.html    # Upload & search view
│   └── config.py          # Configuration
├── data/
│   ├── images/            # Stored JPEG images
│   └── db/                # SQLite + FAISS index
├── run.py                 # Application entry point
└── requirements.txt       # Dependencies
```

---

## UI Access

- **Dashboard**: `http://localhost:5000/`
- **Camera Mode**: `http://localhost:5000/camera`
- **Upload Mode**: `http://localhost:5000/upload`

---

## Configuration

Edit `app/config.py`:

```python
SIMILARITY_THRESHOLD = 0.6   # Match threshold
PORT = 5000                  # API port
```

---

## Quick Start

```bash
pip install -r requirements.txt
python run.py
```

Server runs at http://localhost:5000
API Docs at http://localhost:5000/docs
Dashboard at http://localhost:5000/

---

## Database Schema

### persons
| Column | Type | Description |
|--------|------|-------------|
| id | TEXT | UUID primary key |
| name | TEXT | Optional assigned name |
| faiss_index | INTEGER | FAISS index position |
| image_count | INTEGER | Number of images |
| created_at | TIMESTAMP | Creation time |

### images
| Column | Type | Description |
|--------|------|-------------|
| id | TEXT | UUID primary key |
| person_id | TEXT | Foreign key to persons |
| image_path | TEXT | Path to JPEG file |
| score | REAL | Quality score |
| file_size | INTEGER | File size in bytes |

---

## Technology Stack

| Component | Technology |
|-----------|------------|
| Face Encoder | InsightFace (ONNX, CPU) |
| Vector DB | FAISS-CPU |
| Metadata | SQLite |
| Image Compression | MozJPEG (Pillow) |
| API Framework | FastAPI (Python) |
| Server | Uvicorn |

---

## Performance

| Operation | Time |
|-----------|------|
| Add image | ~300ms |
| Search | ~250ms |
| Top N | <50ms |

Tested with up to 300,000 images.
