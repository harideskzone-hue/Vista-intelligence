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


_storage: Storage | None = None


def get_storage() -> Storage:
    """Get or create the global Storage instance."""
    global _storage
    if _storage is None:
        _storage = Storage()
    return _storage

