"""
Identity Manager: SQLite-backed source of truth for identities and metadata.
Enforces the invariant that new identities can only be created via explicit enrollment.
"""
from __future__ import annotations

import datetime
import os
import sqlite3
import uuid
import logging
from contextlib import contextmanager
from typing import Generator

from edge.face.types import (
    Person, FaceEmbedding, FaceImage, CameraConfig, RecognitionEvent
)

log = logging.getLogger("identity_manager")

class IdentityManager:
    """
    Manages persistence for the Face module. 
    SQLite is the source of truth for identities.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    @contextmanager
    def _get_conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Initialize SQLite tables if they do not exist."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with self._get_conn() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS persons (
                    person_id TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL,
                    category TEXT NOT NULL,
                    classification TEXT NOT NULL DEFAULT 'normal',
                    created_at TIMESTAMP NOT NULL,
                    status TEXT NOT NULL
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS face_embeddings (
                    embedding_id TEXT PRIMARY KEY,
                    person_id TEXT NOT NULL,
                    embedding_vector BLOB NOT NULL,
                    model_version TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    quality_score REAL NOT NULL,
                    source TEXT NOT NULL,
                    FOREIGN KEY (person_id) REFERENCES persons (person_id)
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS face_images (
                    image_id TEXT PRIMARY KEY,
                    person_id TEXT NOT NULL,
                    path TEXT NOT NULL,
                    camera_id TEXT NOT NULL,
                    frame_id INTEGER NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    sharpness REAL NOT NULL,
                    pose_score REAL,
                    lighting_score REAL,
                    occlusion_score REAL,
                    recognition_similarity REAL NOT NULL,
                    quality_score REAL NOT NULL,
                    hash_sha256 TEXT NOT NULL UNIQUE,
                    source TEXT NOT NULL,
                    FOREIGN KEY (person_id) REFERENCES persons (person_id)
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS recognition_events (
                    event_id TEXT PRIMARY KEY,
                    camera_id TEXT NOT NULL,
                    person_id TEXT,
                    frame_id INTEGER NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    similarity REAL,
                    quality_score REAL NOT NULL,
                    decision TEXT NOT NULL,
                    image_id TEXT,
                    FOREIGN KEY (person_id) REFERENCES persons (person_id),
                    FOREIGN KEY (image_id) REFERENCES face_images (image_id)
                )
            ''')
            conn.commit()

            try:
                conn.execute('ALTER TABLE face_images ADD COLUMN embedding_id TEXT')
                conn.commit()
            except sqlite3.OperationalError:
                pass # Column already exists
    
    # --- PERSON MANAGEMENT ---

    def enroll_person(self, person_id: str, display_name: str, category: str, classification: str = "normal") -> Person:
        """
        Explicitly enroll a new person.
        This is the ONLY way an identity is created.
        """
        now = datetime.datetime.now(datetime.timezone.utc)
        person = Person(
            person_id=person_id,
            display_name=display_name,
            category=category,
            classification=classification,
            created_at=now,
            status="ACTIVE"
        )
        
        with self._get_conn() as conn:
            conn.execute('''
                INSERT INTO persons (person_id, display_name, category, classification, created_at, status)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (person.person_id, person.display_name, person.category, person.classification, 
                  person.created_at.isoformat(), person.status))
            conn.commit()
            
        log.info(f"Enrolled new person: {person_id} ({display_name})")
        return person

    def get_person(self, person_id: str) -> Person | None:
        """Retrieve a person by ID."""
        with self._get_conn() as conn:
            row = conn.execute('SELECT * FROM persons WHERE person_id = ?', (person_id,)).fetchone()
            if not row:
                return None
            return Person(
                person_id=row['person_id'],
                display_name=row['display_name'],
                category=row['category'],
                classification=row['classification'],
                created_at=datetime.datetime.fromisoformat(row['created_at']),
                status=row['status']
            )

    def execute_enrollment_transaction(self, person: Person, embedding: FaceEmbedding, image: FaceImage) -> None:
        """Atomically insert a person, their embedding, and their image."""
        with self._get_conn() as conn:
            # SQLite transaction implicitly started
            try:
                conn.execute('''
                    INSERT INTO persons (person_id, display_name, category, classification, created_at, status)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (person.person_id, person.display_name, person.category, person.classification, 
                      person.created_at.isoformat(), person.status))
                      
                conn.execute('''
                    INSERT INTO face_embeddings 
                    (embedding_id, person_id, embedding_vector, model_version, created_at, quality_score, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (embedding.embedding_id, embedding.person_id, embedding.embedding_vector, embedding.model_version, 
                      embedding.created_at.isoformat(), embedding.quality_score, embedding.source))
                      
                conn.execute('''
                    INSERT INTO face_images 
                    (image_id, person_id, path, camera_id, frame_id, timestamp, sharpness, 
                     pose_score, lighting_score, occlusion_score, recognition_similarity, 
                     quality_score, hash_sha256, source, embedding_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    image.image_id, image.person_id, image.path, image.camera_id, 
                    image.frame_id, image.timestamp.isoformat(), image.sharpness, 
                    image.pose_score, image.lighting_score, image.occlusion_score, 
                    image.recognition_similarity, image.quality_score, image.hash_sha256, image.source, image.embedding_id
                ))
                conn.commit()
            except sqlite3.Error as e:
                conn.rollback()
                raise RuntimeError(f"Enrollment transaction failed: {e}")

    def execute_bulk_enrollment_transaction(self, person: Person, embeddings: list[FaceEmbedding], images: list[FaceImage]) -> None:
        """Atomically insert a person, their multiple embeddings, and their images."""
        if not embeddings or not images:
            raise ValueError("Must provide at least one embedding and image for bulk enrollment.")
            
        with self._get_conn() as conn:
            # SQLite transaction implicitly started
            try:
                conn.execute('''
                    INSERT INTO persons (person_id, display_name, category, classification, created_at, status)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (person.person_id, person.display_name, person.category, person.classification, 
                      person.created_at.isoformat(), person.status))
                      
                for embedding in embeddings:
                    conn.execute('''
                        INSERT INTO face_embeddings 
                        (embedding_id, person_id, embedding_vector, model_version, created_at, quality_score, source)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (embedding.embedding_id, embedding.person_id, embedding.embedding_vector, embedding.model_version, 
                          embedding.created_at.isoformat(), embedding.quality_score, embedding.source))
                          
                for image in images:
                    conn.execute('''
                        INSERT INTO face_images 
                        (image_id, person_id, path, camera_id, frame_id, timestamp, sharpness, 
                         pose_score, lighting_score, occlusion_score, recognition_similarity, 
                         quality_score, hash_sha256, source)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        image.image_id, image.person_id, image.path, image.camera_id, 
                        image.frame_id, image.timestamp.isoformat(), image.sharpness, 
                        image.pose_score, image.lighting_score, image.occlusion_score, 
                        image.recognition_similarity, image.quality_score, image.hash_sha256, image.source
                    ))
                conn.commit()
            except sqlite3.Error as e:
                conn.rollback()
                raise RuntimeError(f"Bulk enrollment transaction failed: {e}")


    def delete_person(self, person_id: str) -> None:
        """Completely delete a person and all their data (embeddings, images, events)."""
        import os
        import shutil
        
        with self._get_conn() as conn:
            try:
                # SQLite doesn't have cascade on our tables, so we delete manually in order
                conn.execute('DELETE FROM recognition_events WHERE person_id = ?', (person_id,))
                conn.execute('DELETE FROM face_images WHERE person_id = ?', (person_id,))
                conn.execute('DELETE FROM face_embeddings WHERE person_id = ?', (person_id,))
                conn.execute('DELETE FROM persons WHERE person_id = ?', (person_id,))
                conn.commit()
            except sqlite3.Error as e:
                conn.rollback()
                raise RuntimeError(f"Failed to delete person {person_id}: {e}")
                
        # Also clean up evidence directory
        data_dir = os.path.dirname(self.db_path)
        evidence_dir = os.path.join(data_dir, "evidence", person_id)
        if os.path.exists(evidence_dir):
            try:
                shutil.rmtree(evidence_dir)
            except Exception as e:
                log.error(f"Failed to delete evidence dir for {person_id}: {e}")
                
        # And enrollment directory
        enrollment_dir = os.path.join(data_dir, "persons", person_id)
        if os.path.exists(enrollment_dir):
            try:
                shutil.rmtree(enrollment_dir)
            except Exception as e:
                log.error(f"Failed to delete enrollment dir for {person_id}: {e}")

    def delete_reference_image(self, image_id: str) -> str:
        """Delete a reference image. If it has a corresponding embedding, delete that too. Returns person_id."""
        import os
        
        with self._get_conn() as conn:
            row = conn.execute('SELECT person_id, path, source, embedding_id FROM face_images WHERE image_id = ?', (image_id,)).fetchone()
            if not row:
                raise ValueError(f"Image {image_id} not found.")
                
            person_id = row['person_id']
            path = row['path']
            source = row['source']
            embedding_id = row['embedding_id']
            
            # Ensure it is a reference image
            if not source.startswith('ENROLLMENT'):
                raise ValueError(f"Image {image_id} is recognition evidence, not a reference image.")
                
            try:
                # Delete the image record
                conn.execute('DELETE FROM face_images WHERE image_id = ?', (image_id,))
                
                # Delete the corresponding embedding if present
                if embedding_id:
                    conn.execute('DELETE FROM face_embeddings WHERE embedding_id = ?', (embedding_id,))
                    
                conn.commit()
            except sqlite3.Error as e:
                conn.rollback()
                raise RuntimeError(f"Failed to delete reference image {image_id}: {e}")
                
        # Delete file
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except Exception as e:
                log.error(f"Failed to delete image file {path}: {e}")
                
        return person_id

    def get_all_persons(self) -> list[Person]:
        """Retrieve all enrolled persons."""
        persons = []
        with self._get_conn() as conn:
            rows = conn.execute('SELECT * FROM persons').fetchall()
            for row in rows:
                persons.append(Person(
                    person_id=row['person_id'],
                    display_name=row['display_name'],
                    category=row['category'],
                    classification=row['classification'],
                    created_at=datetime.datetime.fromisoformat(row['created_at']),
                    status=row['status']
                ))
        return persons

    # --- EMBEDDINGS ---

    def add_embedding(self, person_id: str, embedding_vector: bytes, 
                      model_version: str, quality_score: float, source: str) -> FaceEmbedding:
        """Add a new embedding for an existing person."""
        # Ensure person exists
        if not self.get_person(person_id):
            raise ValueError(f"Cannot add embedding: Person {person_id} does not exist.")
            
        embedding_id = str(uuid.uuid4())
        now = datetime.datetime.now(datetime.timezone.utc)
        
        emb = FaceEmbedding(
            embedding_id=embedding_id,
            person_id=person_id,
            embedding_vector=embedding_vector,
            model_version=model_version,
            created_at=now,
            quality_score=quality_score,
            source=source
        )
        
        with self._get_conn() as conn:
            conn.execute('''
                INSERT INTO face_embeddings 
                (embedding_id, person_id, embedding_vector, model_version, created_at, quality_score, source)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (emb.embedding_id, emb.person_id, emb.embedding_vector, emb.model_version, 
                  emb.created_at.isoformat(), emb.quality_score, emb.source))
            conn.commit()
            
        return emb

    def get_all_embeddings(self) -> list[FaceEmbedding]:
        """Retrieve all embeddings (for building the FAISS index)."""
        embeddings = []
        with self._get_conn() as conn:
            rows = conn.execute('SELECT * FROM face_embeddings').fetchall()
            for row in rows:
                embeddings.append(FaceEmbedding(
                    embedding_id=row['embedding_id'],
                    person_id=row['person_id'],
                    embedding_vector=row['embedding_vector'],
                    model_version=row['model_version'],
                    created_at=datetime.datetime.fromisoformat(row['created_at']),
                    quality_score=row['quality_score'],
                    source=row['source']
                ))
        return embeddings

    def get_image(self, image_id: str) -> FaceImage | None:
        """Retrieve a specific image record by ID."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM face_images WHERE image_id = ?",
                (image_id,)
            ).fetchone()
            if row:
                return FaceImage(**dict(row))
            return None

    def get_embedding(self, embedding_id: str) -> FaceEmbedding | None:
        """Retrieve a specific embedding by ID."""
        with self._get_conn() as conn:
            row = conn.execute('SELECT * FROM face_embeddings WHERE embedding_id = ?', (embedding_id,)).fetchone()
            if not row:
                return None
            return FaceEmbedding(
                embedding_id=row['embedding_id'],
                person_id=row['person_id'],
                embedding_vector=row['embedding_vector'],
                model_version=row['model_version'],
                created_at=datetime.datetime.fromisoformat(row['created_at']),
                quality_score=row['quality_score'],
                source=row['source']
            )

    # --- IMAGES & EVENTS ---

    def cleanup_old_evidence(self, days: int) -> int:
        """Delete recognition evidence older than N days. Returns number of deleted images."""
        import os
        from datetime import datetime, timedelta
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        
        deleted_count = 0
        with self._get_conn() as conn:
            # Find evidence images older than cutoff (source != 'ENROLLMENT')
            rows = conn.execute(
                "SELECT image_id, path FROM face_images WHERE source != 'ENROLLMENT' AND timestamp < ?",
                (cutoff,)
            ).fetchall()
            
            for row in rows:
                image_id = row['image_id']
                path = row['path']
                
                # Delete from DB
                conn.execute("DELETE FROM face_images WHERE image_id = ?", (image_id,))
                
                # Delete from disk
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except Exception as e:
                        log.error(f"Failed to delete evidence file {path}: {e}")
                
                deleted_count += 1
                
            # Also clean up old recognition events
            conn.execute("DELETE FROM recognition_events WHERE timestamp < ?", (cutoff,))
            
            conn.commit()
            
        return deleted_count

    def get_reference_images(self, person_id: str) -> list[str]:
        """Get all reference image IDs for a person."""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT image_id FROM face_images WHERE person_id = ? AND source LIKE 'ENROLLMENT%'", 
                (person_id,)
            ).fetchall()
            return [row['image_id'] for row in rows]

    def check_duplicate_image(self, hash_sha256: str) -> bool:
        """Check if an image hash already exists."""
        with self._get_conn() as conn:
            row = conn.execute('SELECT 1 FROM face_images WHERE hash_sha256 = ?', (hash_sha256,)).fetchone()
            return row is not None

    def add_face_image(self, image: FaceImage) -> None:
        """Save face image metadata."""
        if not self.get_person(image.person_id):
            raise ValueError(f"Cannot add image: Person {image.person_id} does not exist.")
            
        with self._get_conn() as conn:
            conn.execute('''
                INSERT INTO face_images 
                (image_id, person_id, path, camera_id, frame_id, timestamp, sharpness, 
                 pose_score, lighting_score, occlusion_score, recognition_similarity, 
                 quality_score, hash_sha256, source, embedding_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                image.image_id, image.person_id, image.path, image.camera_id, 
                image.frame_id, image.timestamp.isoformat(), image.sharpness, 
                image.pose_score, image.lighting_score, image.occlusion_score, 
                image.recognition_similarity, image.quality_score, image.hash_sha256, image.source, image.embedding_id
            ))
            conn.commit()

    def log_event(self, event: RecognitionEvent) -> None:
        """Log a recognition decision event."""
        with self._get_conn() as conn:
            conn.execute('''
                INSERT INTO recognition_events 
                (event_id, camera_id, person_id, frame_id, timestamp, 
                 similarity, quality_score, decision, image_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                event.event_id, event.camera_id, event.person_id, event.frame_id, 
                event.timestamp.isoformat(), event.similarity, event.quality_score, 
                event.decision, event.image_id
            ))
            conn.commit()
