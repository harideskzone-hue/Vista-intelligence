"""
Enrollment Flow: Explicit operator-driven identity creation.
Prevents auto-creation, performs duplicate checking, and ensures transactional persistence.
"""
from __future__ import annotations

import datetime
import os
import uuid
import numpy as np
import logging

from typing import Protocol

from edge.face.types import FaceDetection, QualityDecision, FaceImage, Person, FaceEmbedding
from edge.face.identity_manager import IdentityManager
from edge.face.watchlist import Watchlist

log = logging.getLogger("enrollment")

class FaceDetector(Protocol):
    def detect(self, image: np.ndarray, frame_index: int = 0) -> list[FaceDetection]: ...

class FaceQualityGate(Protocol):
    def evaluate(self, detection: FaceDetection, image: np.ndarray) -> QualityDecision: ...

class FaceEmbedder(Protocol):
    def embed(self, image: np.ndarray, detection: FaceDetection) -> np.ndarray | None: ...


class EnrollmentError(Exception):
    """Base exception for enrollment failures."""
    pass

class EnrollmentDuplicateError(EnrollmentError):
    """Raised when the face is already known."""
    pass


class EnrollmentManager:
    """Coordinates the enrollment pipeline securely."""

    def __init__(
        self,
        detector: FaceDetector,
        quality_gate: FaceQualityGate,
        embedder: FaceEmbedder,
        identity_manager: IdentityManager,
        watchlist: Watchlist,
        duplicate_threshold: float = 0.85
    ):
        self.detector = detector
        self.quality_gate = quality_gate
        self.embedder = embedder
        self.identity_manager = identity_manager
        self.watchlist = watchlist
        self.duplicate_threshold = duplicate_threshold

    def enroll_bulk(self, person_id: str, display_name: str, category: str, images: list[np.ndarray], classification: str = "normal") -> dict:
        """
        Explicitly enroll a new person from multiple images.
        Skips invalid images, returns a summary.
        If zero images are valid, raises EnrollmentError without creating the identity.
        """
        valid_embeddings = []
        valid_face_images = []
        rejected = []
        
        now = datetime.datetime.now(datetime.timezone.utc)
        
        for i, image in enumerate(images):
            try:
                # 1. Detection
                detections = self.detector.detect(image)
                if not detections:
                    rejected.append({"index": i, "reason": "NO_FACE_DETECTED"})
                    continue
                    
                # Select largest face
                detection = max(detections, key=lambda d: (d.x2 - d.x1) * (d.y2 - d.y1))
                
                # 2. Quality Gate
                quality = self.quality_gate.evaluate(detection, image)
                if quality.status != "PASS":
                    rejected.append({"index": i, "reason": f"QUALITY_FAIL: {quality.reason}"})
                    continue
                    
                # 3. Embedding
                embedding = self.embedder.embed(image, detection)
                if embedding is None:
                    rejected.append({"index": i, "reason": "EMBEDDING_FAILED"})
                    continue
                    
                # 4. Duplicate Check
                candidates = self.watchlist.search(embedding, top_k=1)
                if candidates and candidates[0][2] >= self.duplicate_threshold:
                    # In bulk enrollment, we might skip duplicate images of the SAME person or flag if it matches another person.
                    # Since we are creating a NEW person, any match is a red flag.
                    rejected.append({"index": i, "reason": f"POSSIBLE_EXISTING_IDENTITY: {candidates[0][0]}"})
                    continue
                    
                emb_id = str(uuid.uuid4())
                emb = FaceEmbedding(
                    embedding_id=emb_id,
                    person_id=person_id,
                    embedding_vector=embedding.tobytes(),
                    model_version="v1",
                    created_at=now,
                    quality_score=quality.sharpness,
                    source="ENROLLMENT_BULK"
                )
                
                img_id = str(uuid.uuid4())
                _img_path = os.path.join(os.environ.get("SIH26187_DATA", "data"), "persons", person_id, "enrollment", f"{img_id}.jpg")
                os.makedirs(os.path.dirname(_img_path), exist_ok=True)
                import cv2 as _cv2
                _cv2.imwrite(_img_path, image)
                face_img = FaceImage(
                    image_id=img_id,
                    person_id=person_id,
                    path=_img_path,
                    camera_id="ENROLLMENT_BULK",
                    frame_id=0,
                    timestamp=now,
                    sharpness=quality.sharpness,
                    pose_score=None,
                    lighting_score=None,
                    occlusion_score=None,
                    recognition_similarity=1.0,
                    quality_score=quality.sharpness,
                    hash_sha256=str(uuid.uuid4()), # Placeholder
                    source="ENROLLMENT_BULK",
                    embedding_id=emb_id
                )
                
                valid_embeddings.append(emb)
                valid_face_images.append(face_img)
                
            except Exception as e:
                rejected.append({"index": i, "reason": f"ERROR: {str(e)}"})
                
        if not valid_embeddings:
            raise EnrollmentError("ZERO_VALID_IMAGES")
            
        person = Person(
            person_id=person_id,
            display_name=display_name,
            category=category,
            classification=classification,
            created_at=now,
            status="ACTIVE"
        )
        
        # 5. Transactional Persistence
        try:
            self.identity_manager.execute_bulk_enrollment_transaction(person, valid_embeddings, valid_face_images)
        except Exception as e:
            raise EnrollmentError(f"DATABASE_PERSISTENCE_FAILED: {e}")
            
        # 6. Update Watchlist safely
        try:
            self.watchlist.rebuild_index()
        except Exception as e:
            raise EnrollmentError(f"WATCHLIST_SYNC_FAILED: {e}")
            
        log.info(f"Successfully bulk enrolled {person_id} with {len(valid_embeddings)} images. Rejected: {len(rejected)}")
        return {
            "status": "SUCCESS", 
            "person_id": person_id, 
            "accepted": len(valid_embeddings),
            "rejected": len(rejected),
            "rejected_details": rejected
        }

    def enroll(self, person_id: str, display_name: str, category: str, image: np.ndarray, classification: str = "normal") -> dict:
        """
        Explicitly enroll a new person from an image.
        """
        # 1. Detection
        detections = self.detector.detect(image)
        if not detections:
            raise EnrollmentError("NO_FACE_DETECTED")
            
        # Select largest face
        detection = max(detections, key=lambda d: (d.x2 - d.x1) * (d.y2 - d.y1))
        
        # 2. Quality Gate
        quality = self.quality_gate.evaluate(detection, image)
        if quality.status != "PASS":
            raise EnrollmentError(f"QUALITY_FAIL: {quality.reason}")
            
        # 3. Embedding
        embedding = self.embedder.embed(image, detection)
        if embedding is None:
            raise EnrollmentError("EMBEDDING_FAILED")
            
        # 4. Duplicate Check
        candidates = self.watchlist.search(embedding, top_k=1)
        if candidates and candidates[0][2] >= self.duplicate_threshold:
            raise EnrollmentDuplicateError(f"POSSIBLE_EXISTING_IDENTITY: {candidates[0][0]}")

        # 5. Transactional Persistence
        now = datetime.datetime.now(datetime.timezone.utc)
        
        person = Person(
            person_id=person_id,
            display_name=display_name,
            category=category,
            classification=classification,
            created_at=now,
            status="ACTIVE"
        )
        
        emb_id = str(uuid.uuid4())
        emb = FaceEmbedding(
            embedding_id=emb_id,
            person_id=person_id,
            embedding_vector=embedding.tobytes(),
            model_version="v1",
            created_at=now,
            quality_score=quality.sharpness, # Example metric mapping
            source="ENROLLMENT"
        )
        
        img_id = str(uuid.uuid4())
        data_dir = os.environ.get("SIH26187_DATA", "data")
        abs_img_path = os.path.join(data_dir, "persons", person_id, "enrollment", f"{img_id}.jpg")
        os.makedirs(os.path.dirname(abs_img_path), exist_ok=True)
        import cv2
        cv2.imwrite(abs_img_path, image)
        
        face_img = FaceImage(
            image_id=img_id,
            person_id=person_id,
            path=abs_img_path,
            camera_id="ENROLLMENT",
            frame_id=0,
            timestamp=now,
            sharpness=quality.sharpness,
            pose_score=None,
            lighting_score=None,
            occlusion_score=None,
            recognition_similarity=1.0,
            quality_score=quality.sharpness,
            hash_sha256=str(uuid.uuid4()), # Placeholder for actual SHA256 of image bytes
            source="ENROLLMENT",
            embedding_id=emb_id
        )
        
        # We need a transactional execute to prevent partial identity creation
        try:
            self.identity_manager.execute_enrollment_transaction(person, emb, face_img)
        except Exception as e:
            raise EnrollmentError(f"DATABASE_PERSISTENCE_FAILED: {e}")
            
        # 6. Update Watchlist safely
        try:
            # We explicitly rebuild or add the vector
            # Rebuilding ensures consistency
            self.watchlist.rebuild_index()
        except Exception as e:
            # The SQLite commit succeeded but FAISS failed. 
            # We must fail safely. In a real system, we might flag the index for async rebuild.
            raise EnrollmentError(f"WATCHLIST_SYNC_FAILED: {e}")
            
        log.info(f"Successfully enrolled {person_id}")
        return {"status": "SUCCESS", "person_id": person_id, "embedding_id": emb.embedding_id}

    def add_reference_images(self, person_id: str, images: list[np.ndarray]) -> dict:
        """Add new reference images to an existing person."""
        person = self.identity_manager.get_person(person_id)
        if not person:
            raise EnrollmentError(f"Person {person_id} not found")
            
        valid_embeddings = []
        valid_face_images = []
        rejected = []
        now = datetime.datetime.now(datetime.timezone.utc)
        
        for i, image in enumerate(images):
            try:
                detections = self.detector.detect(image)
                if not detections:
                    rejected.append({"index": i, "reason": "NO_FACE_DETECTED"})
                    continue
                    
                detection = max(detections, key=lambda d: (d.x2 - d.x1) * (d.y2 - d.y1))
                quality = self.quality_gate.evaluate(detection, image)
                if quality.status != "PASS":
                    rejected.append({"index": i, "reason": f"QUALITY_FAIL: {quality.reason}"})
                    continue
                    
                embedding = self.embedder.embed(image, detection)
                if embedding is None:
                    rejected.append({"index": i, "reason": "EMBEDDING_FAILED"})
                    continue
                    
                emb_id = str(uuid.uuid4())
                emb = FaceEmbedding(
                    embedding_id=emb_id,
                    person_id=person_id,
                    embedding_vector=embedding.tobytes(),
                    model_version="v1",
                    created_at=now,
                    quality_score=quality.sharpness,
                    source="ENROLLMENT_ADD"
                )
                
                img_id = str(uuid.uuid4())
                face_img = FaceImage(
                    image_id=img_id,
                    person_id=person_id,
                    path=os.path.join(os.environ.get("SIH26187_DATA", "data"), "persons", person_id, "enrollment", f"{img_id}.jpg"),
                    camera_id="ENROLLMENT_ADD",
                    frame_id=0,
                    timestamp=now,
                    sharpness=quality.sharpness,
                    pose_score=None,
                    lighting_score=None,
                    occlusion_score=None,
                    recognition_similarity=1.0,
                    quality_score=quality.sharpness,
                    hash_sha256=str(uuid.uuid4()), 
                    source="ENROLLMENT_ADD",
                    embedding_id=emb_id
                )
                
                valid_embeddings.append(emb)
                valid_face_images.append(face_img)
            except Exception as e:
                rejected.append({"index": i, "reason": f"ERROR: {str(e)}"})
                
        if not valid_embeddings:
            raise EnrollmentError("ZERO_VALID_IMAGES")
            
        # Add to DB
        with self.identity_manager._get_conn() as conn:
            try:
                for emb in valid_embeddings:
                    conn.execute('''
                        INSERT INTO face_embeddings 
                        (embedding_id, person_id, embedding_vector, model_version, created_at, quality_score, source)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (emb.embedding_id, emb.person_id, emb.embedding_vector, emb.model_version, 
                          emb.created_at.isoformat(), emb.quality_score, emb.source))
                          
                for face_img in valid_face_images:
                    conn.execute('''
                        INSERT INTO face_images 
                        (image_id, person_id, path, camera_id, frame_id, timestamp, sharpness, 
                         pose_score, lighting_score, occlusion_score, recognition_similarity, 
                         quality_score, hash_sha256, source, embedding_id)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        face_img.image_id, face_img.person_id, face_img.path, face_img.camera_id, 
                        face_img.frame_id, face_img.timestamp.isoformat(), face_img.sharpness, 
                        face_img.pose_score, face_img.lighting_score, face_img.occlusion_score, 
                        face_img.recognition_similarity, face_img.quality_score, face_img.hash_sha256, 
                        face_img.source, face_img.embedding_id
                    ))
                conn.commit()
            except Exception as e:
                conn.rollback()
                raise EnrollmentError(f"DB Error: {e}")
                
        self.watchlist.rebuild_index()
        
        return {
            "success": True,
            "person_id": person_id,
            "added_count": len(valid_embeddings),
            "rejected_count": len(rejected),
            "rejected_details": rejected,
            "valid_images": valid_face_images
        }
