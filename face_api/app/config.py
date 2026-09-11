"""\nFace Embedding Database Configuration\n"""
import os

import sys

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Unified User Data Directory (Standalone/Portable mode)
# Priority: 1. SIH26187_DATA env var, 2. repository-local data/ folder
USER_DATA_DIR = os.environ.get("SIH26187_DATA")
if not USER_DATA_DIR:
    # Default to data/ directory at the project root (one level above face_api)
    USER_DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "data")

USER_DATA_DIR = os.path.abspath(USER_DATA_DIR)


DATA_DIR = os.path.join(USER_DATA_DIR, 'face_api_data')
IMAGES_DIR = os.path.join(DATA_DIR, 'images')
DB_DIR = os.path.join(DATA_DIR, 'db')
RECORDINGS_DIR = os.path.join(DATA_DIR, 'recordings')

# Create directories if they don't exist
for dir_path in [USER_DATA_DIR, DATA_DIR, IMAGES_DIR, DB_DIR, RECORDINGS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Database paths
SQLITE_PATH = os.path.join(DB_DIR, 'metadata.db')
FAISS_INDEX_PATH = os.path.join(DB_DIR, 'faiss.index')

# Face matching settings
# buffalo_sc cosine similarity: same-person typically 0.25-0.55, strangers ~0.05-0.25
# Use 0.30 so a face captured from different angles still merges into one person.
# Override with env var: MC_SIMILARITY_THRESHOLD=0.35
SIMILARITY_THRESHOLD: float = float(os.environ.get('MC_SIMILARITY_THRESHOLD', '0.30'))

# Minimum face detection confidence accepted by encoder
MIN_DET_SCORE: float = float(os.environ.get('MC_MIN_DET_SCORE', '0.50'))

EMBEDDING_DIM = 512  # InsightFace embedding dimension

# Server settings
HOST = '0.0.0.0'
PORT = 5001
DEBUG = True
