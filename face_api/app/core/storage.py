"""
Storage Module
Handles image storage (MozJPEG) and SQLite metadata with person naming.
"""
import os
import uuid
import sqlite3
import base64
import io
from datetime import datetime
from contextlib import contextmanager

import numpy as np
from PIL import Image

from app.config import SQLITE_PATH, IMAGES_DIR, RECORDINGS_DIR


class Storage:
    """Manages image files and SQLite metadata with person naming."""
    
    def __init__(self):
        """Initialize storage and create tables if needed."""
        self.db_path = SQLITE_PATH
        self.images_dir = IMAGES_DIR
        self.recordings_dir = RECORDINGS_DIR
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.recordings_dir, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Create database tables if they don't exist."""
        with self._get_connection() as conn:
            conn.executescript("""
                -- Person table
                CREATE TABLE IF NOT EXISTS persons (
                    id TEXT PRIMARY KEY,
                    faiss_index INTEGER,
                    image_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP
                );
                
                -- Images table
                CREATE TABLE IF NOT EXISTS images (
                    id TEXT PRIMARY KEY,
                    person_id TEXT REFERENCES persons(id),
                    image_path TEXT,
                    score REAL,
                    file_size INTEGER,
                    created_at TIMESTAMP
                );
                
                CREATE INDEX IF NOT EXISTS idx_images_person 
                ON images(person_id);
                
                CREATE INDEX IF NOT EXISTS idx_images_score 
                ON images(score DESC);
                
                -- Recordings table
                CREATE TABLE IF NOT EXISTS recordings (
                    id TEXT PRIMARY KEY,
                    camera_id TEXT,
                    label TEXT,
                    file_path TEXT,
                    start_time TIMESTAMP,
                    end_time TIMESTAMP,
                    duration REAL,
                    size INTEGER,
                    status TEXT
                );
            """)

            # -- API Integration tables (State Government) --------------------
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS api_keys (
                    id          TEXT PRIMARY KEY,
                    name        TEXT NOT NULL,
                    key_hash    TEXT NOT NULL UNIQUE,
                    key_prefix  TEXT NOT NULL,
                    state       TEXT,
                    created_at  TIMESTAMP NOT NULL,
                    last_used   TIMESTAMP,
                    is_active   INTEGER DEFAULT 1
                );
                CREATE TABLE IF NOT EXISTS api_request_logs (
                    id            TEXT PRIMARY KEY,
                    api_key_id    TEXT REFERENCES api_keys(id),
                    endpoint      TEXT,
                    method        TEXT,
                    request_id    TEXT,
                    status_code   INTEGER,
                    timestamp     TIMESTAMP NOT NULL,
                    source_ip     TEXT,
                    error_message TEXT
                );
                CREATE TABLE IF NOT EXISTS idempotency_keys (
                    idempotency_key TEXT PRIMARY KEY,
                    api_key_id      TEXT,
                    result_json     TEXT,
                    created_at      TIMESTAMP NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash);
                CREATE INDEX IF NOT EXISTS idx_request_logs_key ON api_request_logs(api_key_id);
                CREATE INDEX IF NOT EXISTS idx_idempotency ON idempotency_keys(idempotency_key);
            """)

            # Migration: Add source provenance columns to persons table
            cursor = conn.execute("PRAGMA table_info(persons)")
            columns = [row[1] for row in cursor.fetchall()]
            if "source_type" not in columns:
                conn.execute("ALTER TABLE persons ADD COLUMN source_type TEXT DEFAULT 'MANUAL'")
            if "source_state" not in columns:
                conn.execute("ALTER TABLE persons ADD COLUMN source_state TEXT")
            if "source_key_id" not in columns:
                conn.execute("ALTER TABLE persons ADD COLUMN source_key_id TEXT")
            if "crime" not in columns:
                conn.execute("ALTER TABLE persons ADD COLUMN crime TEXT")
            if "case_number" not in columns:
                conn.execute("ALTER TABLE persons ADD COLUMN case_number TEXT")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_persons_source ON persons(source_type)")

            # Migration: Add name column if it doesn't exist
            cursor = conn.execute("PRAGMA table_info(persons)")
            columns = [row[1] for row in cursor.fetchall()]
            if 'name' not in columns:
                conn.execute("ALTER TABLE persons ADD COLUMN name TEXT")
            
            if 'classification' not in columns:
                conn.execute("ALTER TABLE persons ADD COLUMN classification TEXT DEFAULT 'normal'")
            
            # Migration: Add file_size column to images if it doesn't exist
            cursor = conn.execute("PRAGMA table_info(images)")
            img_columns = [row[1] for row in cursor.fetchall()]
            if 'file_size' not in img_columns:
                conn.execute("ALTER TABLE images ADD COLUMN file_size INTEGER")
            
            # Migration: Add score column to images if it doesn't exist
            cursor = conn.execute("PRAGMA table_info(images)")
            img_columns = [row[1] for row in cursor.fetchall()]
            if 'score' not in img_columns:
                conn.execute("ALTER TABLE images ADD COLUMN score REAL DEFAULT 0.0")
            
            # Create unique index for name (enforces uniqueness)
            conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_person_name ON persons(name)")
    
    @contextmanager
    def _get_connection(self):
        """Get a database connection with context management."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()
    
    def create_person(self, faiss_index: int, name: str = None, classification: str = 'normal') -> str:
        """
        Create a new person entry.
        
        Args:
            faiss_index: Index position in FAISS
            name: Optional name for the person
            
        Returns:
            Generated person ID (UUID)
        """
        person_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO persons (id, name, classification, faiss_index, image_count, created_at, updated_at)
                VALUES (?, ?, ?, ?, 0, ?, ?)
            """, (person_id, name, classification, faiss_index, now, now))
        
        return person_id
    
    def get_person(self, person_id: str) -> dict | None:
        """Get person info by ID."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM persons WHERE id = ?", (person_id,)
            ).fetchone()
            
            if row:
                return dict(row)
        return None
    
    def get_person_by_name(self, name: str) -> dict | None:
        """Get person info by name."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM persons WHERE name = ?", (name,)
            ).fetchone()
            
            if row:
                return dict(row)
        return None
    
    def get_person_by_id_or_name(self, identifier: str) -> dict | None:
        """Get person by ID or name."""
        # Try by ID first
        person = self.get_person(identifier)
        if person:
            return person
        # Try by name
        return self.get_person_by_name(identifier)
    
    def assign_name(self, person_id: str, name: str) -> bool:
        """
        Assign a name to a person.
        
        Args:
            person_id: Person ID
            name: Name to assign
            
        Returns:
            True if successful, False if person not found or name taken
        """
        now = datetime.now().isoformat()
        try:
            with self._get_connection() as conn:
                result = conn.execute("""
                    UPDATE persons SET name = ?, updated_at = ? WHERE id = ?
                """, (name, now, person_id))
                return result.rowcount > 0
        except sqlite3.IntegrityError:
            # Name already exists
            return False
    
    def update_person_faiss_index(self, person_id: str, faiss_index: int):
        """Update the FAISS index for a person."""
        now = datetime.now().isoformat()
        with self._get_connection() as conn:
            conn.execute("""
                UPDATE persons SET faiss_index = ?, updated_at = ? WHERE id = ?
            """, (faiss_index, now, person_id))
    
    def increment_image_count(self, person_id: str):
        """Increment the image count for a person."""
        now = datetime.now().isoformat()
        with self._get_connection() as conn:
            conn.execute("""
                UPDATE persons SET image_count = image_count + 1, updated_at = ? WHERE id = ?
            """, (now, person_id))
    
    def save_image(self, person_id: str, image_bytes: bytes, score: float) -> str:
        """
        Save an image to disk with MozJPEG compression.
        
        Args:
            person_id: Person this image belongs to
            image_bytes: Raw image bytes
            score: Quality score from external model
            
        Returns:
            Generated image ID
        """
        if not person_id or str(person_id).upper() == "UNKNOWN":
            import logging
            logging.getLogger(__name__).warning("save_image called with UNKNOWN. Rejecting storage.")
            return ""

        image_id = str(uuid.uuid4())

        
        # Create person subdirectory
        person_dir = os.path.join(self.images_dir, person_id)
        os.makedirs(person_dir, exist_ok=True)
        
        # Compress with Pillow (MozJPEG-compatible optimization)
        img = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if img.mode in ('RGBA', 'P'):
            img = img.convert('RGB')
        
        # Save as optimized JPEG
        image_path = os.path.join(person_dir, f"{image_id}.jpg")
        output = io.BytesIO()
        img.save(output, 'JPEG', quality=85, optimize=True)
        compressed_bytes = output.getvalue()
        
        # Write to disk
        with open(image_path, 'wb') as f:
            f.write(compressed_bytes)
        
        file_size = len(compressed_bytes)
        
        # Record in database
        now = datetime.now().isoformat()
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO images (id, person_id, image_path, score, file_size, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (image_id, person_id, image_path, score, file_size, now))
        
        # Increment image count
        self.increment_image_count(person_id)
        
        return image_id
    
    def get_all_images(self, person_id: str) -> list[dict]:
        """Get all images for a person."""
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT id, image_path, score, file_size, created_at
                FROM images
                WHERE person_id = ?
                ORDER BY score DESC
            """, (person_id,)).fetchall()
            
            return [dict(row) for row in rows]
    
    def get_image_bytes(self, image_id: str) -> bytes | None:
        """Get raw JPEG bytes for an image."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT image_path FROM images WHERE id = ?", (image_id,)
            ).fetchone()
            
            if row and os.path.exists(row['image_path']):
                with open(row['image_path'], 'rb') as f:
                    return f.read()
        return None
    
    def get_image_as_base64(self, image_id: str) -> str | None:
        """Get image as base64 string."""
        img_bytes = self.get_image_bytes(image_id)
        if img_bytes:
            return base64.b64encode(img_bytes).decode('utf-8')
        return None
    
    def get_image_path(self, image_id: str) -> str | None:
        """Get the file path for an image by ID."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT image_path FROM images WHERE id = ?", (image_id,)
            ).fetchone()
            
            if row:
                return row['image_path']
        return None
    
    def delete_person(self, person_id: str) -> bool:
        """Delete a person and all their images."""
        person = self.get_person(person_id)
        if not person:
            return False
        
        # Get all image paths first
        images = self.get_all_images(person_id)
        
        # Delete image files
        for img in images:
            if os.path.exists(img['image_path']):
                os.remove(img['image_path'])
        
        # Delete person directory if empty
        person_dir = os.path.join(self.images_dir, person_id)
        if os.path.exists(person_dir) and not os.listdir(person_dir):
            os.rmdir(person_dir)
        
        # Delete from database
        with self._get_connection() as conn:
            conn.execute("DELETE FROM images WHERE person_id = ?", (person_id,))
            conn.execute("DELETE FROM persons WHERE id = ?", (person_id,))
        
        return True
    
    def get_all_persons(self) -> list[dict]:
        """Get all persons in the database."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM persons ORDER BY created_at DESC"
            ).fetchall()
            return [dict(row) for row in rows]
    
    def get_person_count(self) -> int:
        """Get total number of persons."""
        with self._get_connection() as conn:
            row = conn.execute("SELECT COUNT(*) as count FROM persons").fetchone()
            return row['count']
    
    def get_top_images_global(self, n: int = 3) -> list[dict]:
        """Get top N images across ALL persons by score."""
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT i.id, i.person_id, i.image_path, i.score, i.created_at,
                       p.name as person_name
                FROM images i
                LEFT JOIN persons p ON i.person_id = p.id
                ORDER BY i.score DESC
                LIMIT ?
            """, (n,)).fetchall()
            
            return [dict(row) for row in rows]

    def get_recent_images_global(self, n: int = 12) -> list[dict]:
        """Get N most recent images across ALL persons."""
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT i.id, i.person_id, i.image_path, i.score, i.created_at,
                       p.name as person_name
                FROM images i
                LEFT JOIN persons p ON i.person_id = p.id
                ORDER BY i.created_at DESC
                LIMIT ?
            """, (n,)).fetchall()
            
            return [dict(row) for row in rows]

    def reset(self):
        """Delete all data (database and images)."""
        import shutil
        
        # remove sqlite db
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except OSError:
                pass
                
        # clear images directory
        if os.path.exists(self.images_dir):
            for filename in os.listdir(self.images_dir):
                file_path = os.path.join(self.images_dir, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")
                    
        # Re-init DB
        self._init_db()


# Global instance
    # ── Recordings ─────────────────────────────────────────────────────────────
    
    def start_recording(self, camera_id: str, label: str, file_path: str) -> str:
        rec_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO recordings (id, camera_id, label, file_path, start_time, duration, size, status)
                VALUES (?, ?, ?, ?, ?, 0.0, 0, 'RECORDING')
            """, (rec_id, camera_id, label, file_path, now))
        return rec_id
        
    def finish_recording(self, rec_id: str, end_time: str, duration: float, size: int, status: str = 'COMPLETED'):
        with self._get_connection() as conn:
            conn.execute("""
                UPDATE recordings
                SET end_time = ?, duration = ?, size = ?, status = ?
                WHERE id = ?
            """, (end_time, duration, size, status, rec_id))
            
    def get_recordings(self, limit: int = 100):
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT * FROM recordings
                ORDER BY start_time DESC
                LIMIT ?
            """, (limit,)).fetchall()
            return [dict(row) for row in rows]
            
    def get_recording(self, rec_id: str) -> dict | None:
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM recordings WHERE id = ?", (rec_id,)).fetchone()
            if row:
                return dict(row)
        return None
        
    def cleanup_recordings(self, days: int) -> int:
        """Delete recordings older than N days. Returns number of deleted recordings."""
        from datetime import datetime, timedelta
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        
        deleted_count = 0
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT id, file_path FROM recordings WHERE start_time < ?", (cutoff,)
            ).fetchall()
            
            for row in rows:
                rec_id = row['id']
                file_path = row['file_path']
                
                conn.execute("DELETE FROM recordings WHERE id = ?", (rec_id,))
                if file_path and os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        print(f"Failed to delete recording file {file_path}: {e}")
                        
                deleted_count += 1
            conn.commit()
            
        return deleted_count

    def delete_recording(self, rec_id: str) -> bool:
        with self._get_connection() as conn:
            conn.execute("DELETE FROM recordings WHERE id = ?", (rec_id,))
        return True


    # ── API Key Management ────────────────────────────────────────────────────

    def create_api_key(self, name: str, state: str = None) -> tuple:
        """
        Generate a new API key, store only its SHA-256 hash.
        Returns (full_key_string, record_dict).
        The full key is returned ONCE and never retrievable again.
        """
        import secrets, hashlib, uuid, datetime
        raw = "vsk_" + secrets.token_hex(24)          # vsk_<48 hex chars> = 52 chars total
        key_hash   = hashlib.sha256(raw.encode()).hexdigest()
        key_prefix = raw[:12]                          # "vsk_XXXXXXXX" — 12 chars for display
        key_id     = str(uuid.uuid4())
        now        = datetime.datetime.utcnow().isoformat()

        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO api_keys (id, name, key_hash, key_prefix, state, created_at, is_active)
                VALUES (?, ?, ?, ?, ?, ?, 1)
            """, (key_id, name, key_hash, key_prefix, state, now))

        record = {
            "id": key_id, "name": name, "key_prefix": key_prefix,
            "state": state, "created_at": now, "is_active": True,
        }
        return raw, record

    def validate_api_key(self, raw_key: str) -> dict | None:
        """
        Validate an API key. Returns the key record if valid and active, else None.
        Also updates last_used timestamp.
        """
        import hashlib, datetime
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        now      = datetime.datetime.utcnow().isoformat()
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM api_keys WHERE key_hash = ? AND is_active = 1", (key_hash,)
            ).fetchone()
            if row is None:
                return None
            conn.execute(
                "UPDATE api_keys SET last_used = ? WHERE id = ?", (now, row["id"])
            )
            return dict(row)

    def list_api_keys(self) -> list:
        """List all API keys (never returns key_hash or full key)."""
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT id, name, key_prefix, state, created_at, last_used, is_active
                FROM api_keys ORDER BY created_at DESC
            """).fetchall()
            return [dict(r) for r in rows]

    def revoke_api_key(self, key_id: str) -> bool:
        """Disable (soft-delete) an API key. Returns True if found."""
        with self._get_connection() as conn:
            result = conn.execute(
                "UPDATE api_keys SET is_active = 0 WHERE id = ?", (key_id,)
            )
            return result.rowcount > 0

    def enable_api_key(self, key_id: str) -> bool:
        """Re-enable a previously revoked API key."""
        with self._get_connection() as conn:
            result = conn.execute(
                "UPDATE api_keys SET is_active = 1 WHERE id = ?", (key_id,)
            )
            return result.rowcount > 0

    def delete_api_key(self, key_id: str) -> bool:
        """Hard delete an API key and its audit logs."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM api_request_logs WHERE api_key_id = ?", (key_id,))
            result = conn.execute("DELETE FROM api_keys WHERE id = ?", (key_id,))
            return result.rowcount > 0

    # ── Audit Logging ─────────────────────────────────────────────────────────

    def log_api_request(
        self, api_key_id: str, endpoint: str, method: str,
        status_code: int, source_ip: str = None,
        request_id: str = None, error_message: str = None
    ) -> str:
        """Insert one row into api_request_logs. Returns log entry id."""
        import uuid, datetime
        log_id = str(uuid.uuid4())
        now    = datetime.datetime.utcnow().isoformat()
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO api_request_logs
                    (id, api_key_id, endpoint, method, request_id, status_code, timestamp, source_ip, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (log_id, api_key_id, endpoint, method, request_id, status_code, now, source_ip, error_message))
        return log_id

    def get_api_key_logs(self, api_key_id: str, limit: int = 50) -> list:
        """Retrieve recent audit log entries for a given API key."""
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT * FROM api_request_logs
                WHERE api_key_id = ? ORDER BY timestamp DESC LIMIT ?
            """, (api_key_id, limit)).fetchall()
            return [dict(r) for r in rows]

    # ── Idempotency ───────────────────────────────────────────────────────────

    def check_idempotency(self, idempotency_key: str) -> dict | None:
        """Return the stored result JSON if this key was already processed."""
        import json
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT result_json FROM idempotency_keys WHERE idempotency_key = ?",
                (idempotency_key,)
            ).fetchone()
            if row:
                return json.loads(row["result_json"])
            return None

    def store_idempotency(self, idempotency_key: str, api_key_id: str, result: dict):
        """Store result for an idempotency key (24-hour window)."""
        import json, datetime
        now = datetime.datetime.utcnow().isoformat()
        with self._get_connection() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO idempotency_keys (idempotency_key, api_key_id, result_json, created_at) VALUES (?,?,?,?)",
                (idempotency_key, api_key_id, json.dumps(result), now)
            )

    # ── Person with source provenance ─────────────────────────────────────────

    def update_person_source(
        self, person_id: str, source_type: str, source_state: str = None,
        source_key_id: str = None, crime: str = None, case_number: str = None
    ):
        """Tag a person record with its origin (STATE_API / MANUAL)."""
        import datetime
        now = datetime.datetime.utcnow().isoformat()
        with self._get_connection() as conn:
            conn.execute("""
                UPDATE persons
                SET source_type=?, source_state=?, source_key_id=?, crime=?, case_number=?, updated_at=?
                WHERE id=?
            """, (source_type, source_state, source_key_id, crime, case_number, now, person_id))

    def get_recent_api_uploads(self, limit: int = 10) -> list:
        """Last N persons uploaded via STATE_API."""
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT id, name, classification, source_type, source_state, crime,
                       case_number, created_at, updated_at
                FROM persons WHERE source_type = 'STATE_API'
                ORDER BY created_at DESC LIMIT ?
            """, (limit,)).fetchall()
            return [dict(r) for r in rows]


_storage: Storage | None = None


def get_storage() -> Storage:
    """Get or create the global Storage instance."""
    global _storage
    if _storage is None:
        _storage = Storage()
    return _storage
