# Face Embedding Database

CPU-based face clustering with MozJPEG compression.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/search` | GET | Search by face, ID, or name |
| `/api/add` | POST | Add image |
| `/api/name` | POST | Assign name to person |
| `/api/topn` | GET | Top 3 images globally |

## Usage

### Search by Face (Base64)
```
GET /api/search?img=<base64_image>
```

### Search by ID or Name
```
GET /api/search?id=<person_id>
GET /api/search?name=John
```

### Add Image
```
POST /api/add
{
  "img": "<base64>",
  "score": 8.5
}
```

### Assign Name
```
POST /api/name
{
  "person_id": "abc123",
  "name": "John"
}
```

### Top 3 Images
```
GET /api/topn
```

## Quick Start

```bash
pip install -r requirements.txt
python run.py
```

Server runs at http://localhost:5000
