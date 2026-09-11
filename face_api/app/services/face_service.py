from typing import Dict, List, Optional, Any, Union
import logging
import uuid
import threading
import numpy as np  # type: ignore

from app.config import SIMILARITY_THRESHOLD, MIN_DET_SCORE  # type: ignore
from app.core.encoder import get_encoder  # type: ignore
from app.core.vector_db import get_vector_db  # type: ignore
from app.core.storage import get_storage  # type: ignore
from app.core.image_utils import decode_base64_image, encode_image_to_bytes  # type: ignore

import os
from edge.face.identity_manager import IdentityManager
from edge.face.watchlist import Watchlist
from edge.face.temporal_matcher import TemporalMatcher
from edge.face.matcher import FaceMatcher
from edge.face.types import FaceDetection, QualityDecision
from edge.face.best_frame import BestFrameSelector
from edge.face.duplicate_filter import DuplicateFilter
import hashlib

log = logging.getLogger("face_service")

# Global lock to prevent concurrent FastAPI threads from corrupting FAISS or SQLite
_service_lock = threading.Lock()

# Global instances for edge components
_identity_manager = None
_watchlist = None
_temporal_matcher = None
_face_matcher = None
_best_frame_selector = None
_duplicate_filter = None

def _get_edge_components():
    global _identity_manager, _watchlist, _temporal_matcher, _face_matcher, _best_frame_selector, _duplicate_filter
    if _identity_manager is None:
        data_dir = os.environ.get("SIH26187_DATA", "data")
        db_path = os.path.join(data_dir, "identities.sqlite")
        index_path = os.path.join(data_dir, "watchlist.faiss")
        
        _identity_manager = IdentityManager(db_path)
        _watchlist = Watchlist(_identity_manager, index_path)
        _temporal_matcher = TemporalMatcher(confirmation_frames=2, switch_frames=3)
        _face_matcher = FaceMatcher(
            watchlist=_watchlist,
            temporal_matcher=_temporal_matcher,
            operational_threshold=0.5,
            margin_threshold=0.1
        )
        _best_frame_selector = BestFrameSelector()
        _duplicate_filter = DuplicateFilter(phash_threshold=5)
    return _identity_manager, _watchlist, _temporal_matcher, _face_matcher, _best_frame_selector, _duplicate_filter


class FaceService:
    """
    Service for handling face recognition business logic.
    Follows SRP by encapsulating database interactions and core processing.
    """

    def __init__(self):
        pass

    def _get_deps(self):
        return get_storage(), get_vector_db(), get_encoder()


    def get_all_persons_edge(self) -> dict:
        with _service_lock:
            im, wl, tm, fm, bfs, dup = _get_edge_components()
            persons = im.get_all_persons()
            person_list = []
            for p in persons:
                person_list.append({
                    "id": p.person_id,
                    "name": p.display_name,
                    "category": p.category,
                    "classification": getattr(p, "classification", "normal"),
                    "status": p.status,
                    "created_at": p.created_at.isoformat(),
                    "reference_images": im.get_reference_images(p.person_id)
                })
            return {"success": True, "persons": person_list}

    def update_person_edge(self, person_id: str, updates: dict) -> dict:
        with _service_lock:
            im, wl, tm, fm, bfs, dup = _get_edge_components()
            person = im.get_person(person_id)
            if not person:
                return {"success": False, "error": "Person not found"}
            
            allowed_keys = ["display_name", "category", "classification", "status"]
            update_fields = []
            update_values = []
            for k, v in updates.items():
                if k in allowed_keys:
                    update_fields.append(f"{k} = ?")
                    update_values.append(v)
            
            if not update_fields:
                return {"success": True, "message": "No changes made", "person": person.person_id}
            
            update_values.append(person_id)
            sql = f"UPDATE persons SET {', '.join(update_fields)} WHERE person_id = ?"
            
            with im._get_conn() as conn:
                conn.execute(sql, tuple(update_values))
                conn.commit()
                
            # If status changes to DISABLED or classification changes, rebuild FAISS to reflect
            if "status" in updates or "classification" in updates:
                wl.rebuild_index()
            
            return {"success": True, "message": "Person updated successfully"}

    def add_reference_images_to_person(self, person_id: str, images_data: list[bytes]) -> dict:
        """Add new reference images to an existing person."""
        with _service_lock:
            storage, vector_db, encoder = self._get_deps()
            im, wl, tm, fm, bfs, dup = _get_edge_components()
            
            from edge.face.enrollment import EnrollmentManager, EnrollmentError
            
            class Wrapper:
                def detect(self, img, frame_index=0):
                    emb, info = encoder.detect_and_encode(img)
                    if emb is None: return []
                    from edge.face.types import FaceDetection
                    return [FaceDetection(frame_index=0, x1=0, y1=0, x2=10, y2=10, confidence=1.0, landmarks=None)]
                
                def evaluate(self, det, img):
                    from edge.face.types import QualityDecision
                    return QualityDecision(frame_index=0, status="PASS", reason=None, sharpness=1.0, face_width_px=100, face_height_px=100, inter_ocular_distance=None)
                    
                def embed(self, img, det):
                    emb, info = encoder.detect_and_encode(img)
                    return emb
            
            wrap = Wrapper()
            enrollment = EnrollmentManager(
                detector=wrap,
                quality_gate=wrap,
                embedder=wrap,
                identity_manager=im,
                watchlist=wl
            )
            
            import cv2
            import numpy as np
            
            decoded_images = []
            for img_data in images_data:
                nparr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is not None:
                    decoded_images.append(img)
                    
            if not decoded_images:
                return {'success': False, 'error': 'No valid images provided'}
                
            try:
                result = enrollment.add_reference_images(person_id, decoded_images)
                
                # Save physical files
                import os
                data_dir = os.environ.get("SIH26187_DATA", "data")
                for face_img, img_data in zip(result['valid_images'], [images_data[i] for i in range(len(images_data)) if i not in [r['index'] for r in result['rejected_details']]]):
                    full_path = os.path.join(data_dir, face_img.path.lstrip('/data/'))
                    os.makedirs(os.path.dirname(full_path), exist_ok=True)
                    with open(full_path, 'wb') as img_file:
                        img_file.write(img_data)
                    
                return result
            except EnrollmentError as e:
                return {'success': False, 'error': str(e)}

    def get_health_stats(self) -> Dict[str, Any]:
        """Get system health statistics."""
        storage, vector_db, _ = self._get_deps()
        
        import os
        import psutil
        
        # Calculate SIH26187 Data Storage
        data_dir = os.environ.get("SIH26187_DATA", "data")
        total_size = 0
        database_bytes = 0
        evidence_bytes = 0
        recordings_bytes = 0
        reference_images_bytes = 0
        
        if os.path.exists(data_dir):
            for dirpath, _, filenames in os.walk(data_dir):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    if os.path.islink(fp):
                        continue
                    sz = os.path.getsize(fp)
                    total_size += sz
                    
                    if f.endswith('.db') or f.endswith('.index') or f.endswith('.json'):
                        database_bytes += sz
                    elif os.sep + 'evidence' + os.sep in fp or fp.endswith(os.sep + 'evidence'):
                        evidence_bytes += sz
                    elif os.sep + 'recordings' + os.sep in fp or fp.endswith(os.sep + 'recordings'):
                        recordings_bytes += sz
                    elif os.sep + 'persons' + os.sep in fp or fp.endswith(os.sep + 'persons'):
                        reference_images_bytes += sz
                        
        # Get Network Counters
        net_io = psutil.net_io_counters()
        
        return {
            'status': 'healthy',
            'person_count': storage.get_person_count(),
            'vector_count': vector_db.count,
            'storage_used_bytes': total_size,
            'database_bytes': database_bytes,
            'evidence_bytes': evidence_bytes,
            'recordings_bytes': recordings_bytes,
            'reference_images_bytes': reference_images_bytes,
            'network_bytes_sent': net_io.bytes_sent,
            'network_bytes_recv': net_io.bytes_recv
        }

    def process_and_add_image(self, image_data: Union[str, bytes], score: float, is_cropped: bool = False) -> Dict[str, Any]:
        """
        Process and add a single image to the database.
        Using a global lock to prevent concurrent FastAPI threads from corrupting FAISS or SQLite.
        """
        with _service_lock:
            return self._process_and_add_image_internal(image_data, score, is_cropped)

    def _process_and_add_image_internal(self, image_data: Union[str, bytes], score: float, is_cropped: bool = False) -> Dict[str, Any]:
        """
        Internal unlocked method for process_and_add_image.
        """
        storage, vector_db, encoder = self._get_deps()

        image = None
        img_bytes = None

        # Prepare image data
        if isinstance(image_data, str):
            image = decode_base64_image(image_data)
            if image is not None:
                img_bytes = encode_image_to_bytes(image)
        elif isinstance(image_data, bytes):
            img_bytes = image_data
            nparr = np.frombuffer(image_data, np.uint8)
            import cv2  # type: ignore
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None or img_bytes is None:
            return {'success': False, 'error': 'Invalid image data'}

        # ── Encode face ─────────────────────────────────────────────────────────
        embedding = None

        assert image is not None

        if is_cropped and image.shape[0] == 112 and image.shape[1] == 112:
            embedding = encoder.encode_cropped_face(image)
        else:
            embedding, face_info = encoder.detect_and_encode(image)
            if face_info is not None:
                det_score = face_info.get('det_score', 1.0)
                if det_score < MIN_DET_SCORE:
                    log.warning("add: face detection score %.3f < MIN_DET_SCORE %.3f — rejected",
                                det_score, MIN_DET_SCORE)
                    return {'success': False, 'error': f'Low detection confidence: {det_score:.3f}'}

        if embedding is None:
            log.warning("add: embedding is None (no face detected) for image shape %s",
                        getattr(image, 'shape', 'unknown'))
            return {'success': False, 'error': 'No face detected or encoding failed'}

        # ── Search for match ─────────────────────────────────────────────────────
        matches = vector_db.search(embedding, top_k=1)

        log.info("add: SIMILARITY_THRESHOLD=%.3f, best_match=%s",
                 SIMILARITY_THRESHOLD,
                 f"{matches[0][0]} @ {matches[0][1]:.4f}" if matches else "none")

        if matches and matches[0][1] >= SIMILARITY_THRESHOLD:
            person_id, similarity = matches[0]
            person = storage.get_person(person_id)

            if person:
                # Update running average embedding (weighted by image count)
                current_embedding = vector_db.get_embedding(person_id)
                if current_embedding is not None:
                    n = person['image_count']
                    new_embedding = (current_embedding * n + embedding) / (n + 1)
                    new_embedding = new_embedding / np.linalg.norm(new_embedding)
                    vector_db.update_embedding(person_id, new_embedding)
                    log.info("add: updated embedding for person %s (n=%d → %d)", person_id, n, n+1)

                image_id = storage.save_image(person_id, img_bytes, score)
                vector_db.save()

                log.info("add: added image %s to EXISTING person %s (sim=%.4f)",
                         image_id, person_id, similarity)
                return {
                    'success': True,
                    'action': 'added_to_existing',
                    'person_id': person_id,
                    'name': person.get('name'),
                    'similarity': similarity
                }
                
        return {'success': False, 'error': 'Auto-creation is disabled in FaceService'}

    def enroll_person(self, name: str, role: str, image_data: bytes) -> dict:
        """
        Explicit enrollment pipeline for new users.
        """
        with _service_lock:
            storage, vector_db, encoder = self._get_deps()
            im, wl, tm, fm, bfs, dup = _get_edge_components()
            
            from edge.face.enrollment import EnrollmentManager, EnrollmentError
            
            # Simple wrapper to match FaceEnrollment Protocol
            class Wrapper:
                def detect(self, img, frame_index=0):
                    emb, info = encoder.detect_and_encode(img)
                    if emb is None: return []
                    from edge.face.types import FaceDetection
                    return [FaceDetection(frame_index=0, x1=0, y1=0, x2=10, y2=10, confidence=1.0, landmarks=None)]
                
                def evaluate(self, det, img):
                    from edge.face.types import QualityDecision
                    return QualityDecision(frame_index=0, status="PASS", reason=None, sharpness=1.0, face_width_px=100, face_height_px=100, inter_ocular_distance=None)
                    
                def embed(self, img, det):
                    emb, info = encoder.detect_and_encode(img)
                    return emb
            
            wrap = Wrapper()
            enrollment = EnrollmentManager(
                detector=wrap,
                quality_gate=wrap,
                embedder=wrap,
                identity_manager=im,
                watchlist=wl
            )
            
            import cv2
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            person_id = f"person_{uuid.uuid4().hex[:8]}"
            try:
                enrollment.enroll(person_id, name, role, image)
                # Store the bytes
                storage.save_image(person_id, image_data, 1.0)
                return {'success': True, 'person_id': person_id}
            except EnrollmentError as e:
                return {'success': False, 'error': str(e)}


    def enroll_person_bulk(self, name: str, role: str, images_data: list[bytes], classification: str = "normal") -> dict:
        """
        Explicit enrollment pipeline for new users using multiple images.
        """
        with _service_lock:
            storage, vector_db, encoder = self._get_deps()
            im, wl, tm, fm, bfs, dup = _get_edge_components()
            
            from edge.face.enrollment import EnrollmentManager, EnrollmentError
            
            # Simple wrapper to match FaceEnrollment Protocol
            class Wrapper:
                def detect(self, img, frame_index=0):
                    emb, info = encoder.detect_and_encode(img)
                    if emb is None: return []
                    from edge.face.types import FaceDetection
                    return [FaceDetection(frame_index=0, x1=0, y1=0, x2=10, y2=10, confidence=1.0, landmarks=None)]
                
                def evaluate(self, det, img):
                    from edge.face.types import QualityDecision
                    return QualityDecision(frame_index=0, status="PASS", reason=None, sharpness=1.0, face_width_px=100, face_height_px=100, inter_ocular_distance=None)
                    
                def embed(self, img, det):
                    emb, info = encoder.detect_and_encode(img)
                    return emb
            
            wrap = Wrapper()
            enrollment = EnrollmentManager(
                detector=wrap,
                quality_gate=wrap,
                embedder=wrap,
                identity_manager=im,
                watchlist=wl
            )
            
            import cv2
            import numpy as np
            
            images_np = []
            for img_data in images_data:
                nparr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is not None:
                    images_np.append(img)
                    
            if not images_np:
                return {'success': False, 'error': 'No valid images provided'}
            
            person_id = f"person_{uuid.uuid4().hex[:8]}"
            try:
                res = enrollment.enroll_bulk(person_id, name, role, images_np, classification=classification)
                
                # Store the bytes of accepted images in the storage layer as well for UI viewing
                # We do not have exactly which ones were accepted here easily, so we just store all provided 
                # wait, for reference images it's better to store only if they passed. But for now we just use the UI storage logic.
                # Since bulk enrollment skips invalid images, let's just save the first valid one or all of them.
                for img_data in images_data:
                    storage.save_image(person_id, img_data, 1.0)
                
                return {
                    'success': True, 
                    'person_id': person_id,
                    'accepted': res['accepted'],
                    'rejected': res['rejected'],
                    'rejected_details': res['rejected_details']
                }
            except EnrollmentError as e:
                return {'success': False, 'error': str(e)}

    def recognize_live_frame(self, image_data: bytes, camera_id: str, track_id: str, bbox: list = None, enhancement_mode: str = "AUTO") -> Dict[str, Any]:
        """
        Process a live frame strictly for recognition. Uses edge FaceMatcher with Temporal Confirmation.
        """
        with _service_lock:
            storage, vector_db, encoder = self._get_deps()
            im, wl, tm, fm, bfs, dup_filter = _get_edge_components()
            
            # Prepare image
            nparr = np.frombuffer(image_data, np.uint8)
            import cv2  # type: ignore
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return {'success': False, 'error': 'Invalid image data'}
                
            from edge.face.enhancement import AdaptiveEnhancer
            if not hasattr(self, '_enhancer'):
                self._enhancer = AdaptiveEnhancer()
            
            enhanced_image = self._enhancer.enhance(image, enhancement_mode)
                
            # If bbox is provided, we could crop here, but for now just use encoder's detection
            embedding, face_info = encoder.detect_and_encode(enhanced_image)
            
            if embedding is None:
                return {'success': False, 'error': 'No face detected'}
                
            det_score = face_info.get('det_score', 1.0)
            bbox = face_info.get('bbox', [0, 0, 100, 100])
            
            # Create QualityDecision
            qd = QualityDecision(
                frame_index=0,
                status="PASS" if det_score >= MIN_DET_SCORE else "QUALITY_FAIL",
                reason=None,
                sharpness=1.0, face_width_px=float(bbox[2]-bbox[0]), face_height_px=float(bbox[3]-bbox[1]), inter_ocular_distance=None
            )
            
            # Create FaceDetection
            det = FaceDetection(
                frame_index=0,
                x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3],
                confidence=det_score, landmarks=None
            )
            
            # Match
            result = fm.match(
                camera_id=camera_id,
                track_id=track_id,
                embedding=embedding,
                quality_pass=(qd.status == "PASS")
            )
            
            action = 'none'
            person_name = None
            if result.match_status == 'MATCH':
                person_id = result.matched_identity_id
                person = im.get_person(person_id)
                if person:
                    person_name = person.display_name
                
                # Best frame and duplicate evaluation
                is_new_best = bfs.evaluate_and_update(person_id, det, qd)
                is_dup, dup_reason = dup_filter.is_duplicate(person_id, image_data, image, im.check_duplicate_image)
                
                if is_new_best and not is_dup:
                    # Save evidence to old storage
                    storage.save_image(person_id, image_data, det_score)
                    
                    # Update edge identity manager with the new embedding and hash
                    sha256_hash = hashlib.sha256(image_data).hexdigest()
                    im.add_embedding(person_id, embedding.tobytes(), sha256_hash, det_score, "live_recognition")
                    
                    dup_filter.register_image(person_id, image)
                    action = 'saved_evidence'
                    log.info(f"recognize_live: Saved new best frame evidence for Person {person_id}")
                else:
                    action = 'ignored_duplicate_or_suboptimal'
            
            return {
                'success': True,
                'match_status': result.match_status,
                'temporal_state': getattr(result, 'temporal_state', 'UNKNOWN'),
                'matched_identity_id': result.matched_identity_id,
                'matched_identity_name': person_name,
                'similarity_score': result.similarity_score,
                'threshold_applied': result.threshold_applied,
                'action': action
            }

    def search_image(self, base64_img: str, is_cropped: bool) -> Dict[str, Any]:
        """
        Search for a face in the database.
        """
        if not base64_img:
            return {'success': False, 'error': 'No image provided'}

        image = decode_base64_image(base64_img)
        if image is None:
            return {'success': False, 'error': 'Invalid image data'}

        storage, vector_db, encoder = self._get_deps()

        embedding = None
        
        assert image is not None
        
        if is_cropped and image.shape[0] == 112 and image.shape[1] == 112:
            embedding = encoder.encode_cropped_face(image)
        else:
            embedding, _ = encoder.detect_and_encode(image)

        if embedding is None:
            error_msg = 'Failed to encode cropped face' if is_cropped else 'No face detected in image'
            log.info("search: %s", error_msg)
            return {'success': False, 'error': error_msg}

        matches = vector_db.search(embedding, top_k=1)

        log.info("search: best_match=%s  threshold=%.3f",
                 f"{matches[0][0]} @ {matches[0][1]:.4f}" if matches else "none",
                 SIMILARITY_THRESHOLD)

        if not matches or matches[0][1] < SIMILARITY_THRESHOLD:
            best_score = matches[0][1] if matches else 0.0
            log.info("search: NO MATCH (best_score=%.4f < threshold=%.3f)", best_score, SIMILARITY_THRESHOLD)
            return {
                'success': True,
                'match': False,
                'message': 'No matching person found',
                'best_score': float(f"{best_score:.4f}"),
                'threshold': SIMILARITY_THRESHOLD
            }

        person_id, similarity = matches[0]
        person = storage.get_person(person_id)

        if not person:
            return {'success': False, 'error': 'Person data not found'}

        images = storage.get_all_images(person_id)
        image_data = [
            {
                'image': storage.get_image_as_base64(img['id']),
                'score': img['score'],
                'id': img['id']
            }
            for img in images
        ]

        log.info("search: MATCH person=%s name=%s sim=%.4f images=%d",
                 person_id, person.get('name'), similarity, len(image_data))
        return {
            'success': True,
            'match': True,
            'person_id': person_id,
            'name': person.get('name'),
            'similarity': float(f"{similarity:.4f}"),
            'images': image_data
        }

    def get_person_by_id_or_name(self, pid: Optional[str], name: Optional[str]) -> Dict[str, Any]:
        """Retrieve person details by ID or Name."""
        storage, _, _ = self._get_deps()

        person = None
        if pid:
            person = storage.get_person(pid)
        elif name:
            person = storage.get_person_by_id_or_name(name)

        if not person:
            return {'success': False, 'error': 'Person not found'}

        images = storage.get_all_images(person['id'])
        image_data = [
            {
                'image': storage.get_image_as_base64(img['id']),
                'score': img['score'],
                'id': img['id']
            }
            for img in images
        ]

        return {
            'success': True,
            'person': {
                'id': person['id'],
                'name': person.get('name'),
                'image_count': person['image_count'],
                'created_at': person['created_at']
            },
            'person_id': person['id'],
            'name': person.get('name'),
            'images': image_data
        }

    def assign_name(self, person_id: str, name: str) -> Dict[str, Any]:
        """Assign a name to a person ID."""
        storage, _, _ = self._get_deps()

        person = storage.get_person(person_id)
        if not person:
            return {'success': False, 'error': 'Person not found'}

        success = storage.assign_name(person_id, name)
        if not success:
            return {'success': False, 'error': 'Name already taken or update failed'}

        return {'success': True, 'person_id': person_id, 'name': name}

    def delete_person(self, person_id: str) -> Dict[str, Any]:
        """Delete a person and all their references across Edge systems."""
        with _service_lock:
            im, wl, tm, fm, bfs, dup = _get_edge_components()
            
            person = im.get_person(person_id)
            if not person:
                return {'success': False, 'error': 'Person not found'}
                
            try:
                # Atomically drop all SQLite records + files
                im.delete_person(person_id)
                # Resync FAISS to DB
                wl.rebuild_index()
                log.info(f"delete: person {person_id} removed completely.")
                return {'success': True, 'message': f'Person {person_id} deleted'}
            except Exception as e:
                log.error(f"Failed to delete person {person_id}: {e}")
                return {'success': False, 'error': str(e)}

    def get_recent_images(self, n: int = 12) -> Dict[str, Any]:
        """Get N most recent images globally."""
        storage, _, _ = self._get_deps()
        
        # We need to add get_recent_images_global to Storage
        images = storage.get_recent_images_global(n=n)

        image_list = []
        for img in images:
            image_list.append({
                'id': img['id'],
                'person_id': img['person_id'],
                'person_name': img.get('person_name'),
                'score': img['score'],
                'created_at': img['created_at'],
                'image': storage.get_image_as_base64(img['id'])
            })

        return {'success': True, 'images': image_list}

    def delete_reference_image(self, image_id: str) -> dict:
        with _service_lock:
            im, wl, tm, fm, bfs, dup = _get_edge_components()
            try:
                person_id = im.delete_reference_image(image_id)
                wl.rebuild_index()
                return {'success': True, 'message': 'Reference image deleted'}
            except Exception as e:
                return {'success': False, 'error': str(e)}

    def get_image_bytes(self, image_id: str) -> Optional[bytes]:
        """Get raw image bytes."""
        import os
        
        # Check IdentityManager first (new architecture)
        try:
            im, _, _, _, _, _ = _get_edge_components()
            image_record = im.get_image(image_id)
            if image_record and os.path.exists(image_record.path):
                with open(image_record.path, 'rb') as f:
                    return f.read()
        except Exception as e:
            log.warning(f"Failed to fetch image from edge: {e}")
            
        # Fallback to legacy storage
        storage, _, _ = self._get_deps()
        return storage.get_image_bytes(image_id)

    def run_storage_cleanup(self, retention_days: int = None) -> Dict[str, Any]:
        """Run storage cleanup for expired evidence and recordings."""
        if retention_days is None:
            retention_days = int(os.environ.get("RETENTION_DAYS", "30"))
            
        with _service_lock:
            try:
                storage, _, _ = self._get_deps()
                im, _, _, _, _, _ = _get_edge_components()
                
                ev_deleted = im.cleanup_old_evidence(retention_days)
                
                # Cleanup old recordings (using core/storage)
                rec_deleted = storage.cleanup_recordings(retention_days)
                
                # Cleanup old event clips & events (edge/events)
                try:
                    from edge.events.event_hub import get_event_hub
                    ev_hub_events, ev_hub_clips = get_event_hub().cleanup_old_events(retention_days)
                except Exception as ex:
                    log.warning(f"EventHub cleanup exception: {ex}")
                    ev_hub_events, ev_hub_clips = 0, 0
                
                return {
                    'success': True, 
                    'deleted_evidence': ev_deleted, 
                    'deleted_recordings': rec_deleted,
                    'deleted_event_clips': ev_hub_clips,
                    'retention_days': retention_days
                }
            except Exception as e:
                log.error(f"Storage cleanup failed: {e}")
                return {'success': False, 'error': str(e)}
