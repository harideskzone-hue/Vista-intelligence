import os
import pytest
import numpy as np
import cv2
from datetime import datetime, timedelta

import face_api.app.services.face_service as fs_module
from face_api.app.services.face_service import FaceService, _get_edge_components
from face_api.app.core.storage import Storage

@pytest.fixture(autouse=True)
def mocked_encoder(monkeypatch):
    class MockEncoder:
        def detect_and_encode(self, image):
            emb = np.random.rand(512).astype(np.float32)
            emb = emb / np.linalg.norm(emb)
            info = {'det_score': 0.99, 'bbox': [0,0,100,100]}
            return emb, info

    monkeypatch.setattr(fs_module, "get_encoder", lambda: MockEncoder())

@pytest.fixture
def clean_service(tmp_path):
    old_val = os.environ.get("SIH26187_DATA")
    os.environ["SIH26187_DATA"] = str(tmp_path)
    
    fs_module._identity_manager = None
    fs_module._watchlist = None
    fs_module._temporal_matcher = None
    fs_module._face_matcher = None
    fs_module._best_frame_selector = None
    fs_module._duplicate_filter = None
    
    yield FaceService()
    
    if old_val:
        os.environ["SIH26187_DATA"] = old_val
    else:
        del os.environ["SIH26187_DATA"]

def create_mock_jpeg():
    img = np.zeros((112, 112, 3), dtype=np.uint8)
    _, buffer = cv2.imencode('.jpg', img)
    return buffer.tobytes()

def test_storage_metrics_and_cleanup(clean_service):
    """
    Task 19 Safety boundary test:
    OLD evidence       -> deleted
    OLD recording      -> deleted
    OLD ref image      -> PRESERVED
    person record      -> PRESERVED
    embedding          -> PRESERVED
    FAISS identity     -> PRESERVED
    Fresh evidence     -> PRESERVED (not expired)
    Fresh recording    -> PRESERVED (not expired)
    """
    # 1. Enroll a person (creates reference image via ENROLLMENT source)
    res = clean_service.enroll_person("Cleanup Subject", "STAFF", create_mock_jpeg())
    assert res['success'], f"Enrollment failed: {res}"
    p_id = res['person_id']
    
    im, wl, _, _, _, _ = _get_edge_components()
    
    # Verify reference image is in DB and on disk
    ref_image_ids = im.get_reference_images(p_id)
    assert len(ref_image_ids) == 1
    ref_image_record = im.get_image(ref_image_ids[0])
    assert ref_image_record is not None, "Reference image record must exist"
    ref_image_path = ref_image_record.path
    assert os.path.exists(ref_image_path), "Reference image file must exist on disk"
    
    # 2. Add an EXPIRED (40 days old) evidence image (source='MATCH')
    expired_time = (datetime.now() - timedelta(days=40)).isoformat()
    old_evidence_path = os.path.join(os.path.dirname(im.db_path), "evidence", p_id, "old_evidence.jpg")
    os.makedirs(os.path.dirname(old_evidence_path), exist_ok=True)
    with open(old_evidence_path, 'wb') as f:
        f.write(b"mock_evidence_old")
        
    old_ev_id = "img_old_evidence_bd"
    with im._get_conn() as conn:
        conn.execute('''
            INSERT INTO face_images 
            (image_id, person_id, path, camera_id, frame_id, timestamp, sharpness, 
             pose_score, lighting_score, occlusion_score, recognition_similarity, 
             quality_score, hash_sha256, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (old_ev_id, p_id, old_evidence_path, "cam1", "f1", expired_time, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, "hash1bd", "MATCH"))
        conn.commit()
        
    # 3. Add a FRESH (5 days old) evidence image (not expired)
    fresh_time = (datetime.now() - timedelta(days=5)).isoformat()
    fresh_evidence_path = os.path.join(os.path.dirname(im.db_path), "evidence", p_id, "fresh_evidence.jpg")
    with open(fresh_evidence_path, 'wb') as f:
        f.write(b"mock_evidence_fresh")
        
    fresh_ev_id = "img_fresh_evidence_bd"
    with im._get_conn() as conn:
        conn.execute('''
            INSERT INTO face_images 
            (image_id, person_id, path, camera_id, frame_id, timestamp, sharpness, 
             pose_score, lighting_score, occlusion_score, recognition_similarity, 
             quality_score, hash_sha256, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (fresh_ev_id, p_id, fresh_evidence_path, "cam1", "f2", fresh_time, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, "hash2bd", "MATCH"))
        conn.commit()

    # 4. Add EXPIRED and FRESH recordings via storage
    storage, _, _ = clean_service._get_deps()
    
    old_rec_path = os.path.join(storage.recordings_dir, "old_rec_bd.mp4")
    os.makedirs(os.path.dirname(old_rec_path), exist_ok=True)
    with open(old_rec_path, 'wb') as f:
        f.write(b"mock_video_old")
        
    with storage._get_connection() as conn:
        conn.execute("DELETE FROM recordings WHERE id IN ('rec_old_bd', 'rec_fresh_bd')")
        conn.execute(
            "INSERT INTO recordings (id, camera_id, label, file_path, start_time, duration, size, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("rec_old_bd", "cam1", "cam1", old_rec_path, expired_time, 10, 14, "COMPLETED")
        )
        conn.commit()
        
    fresh_rec_path = os.path.join(storage.recordings_dir, "fresh_rec_bd.mp4")
    with open(fresh_rec_path, 'wb') as f:
        f.write(b"mock_video_fresh")
        
    with storage._get_connection() as conn:
        conn.execute(
            "INSERT INTO recordings (id, camera_id, label, file_path, start_time, duration, size, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("rec_fresh_bd", "cam1", "cam1", fresh_rec_path, fresh_time, 10, 15, "COMPLETED")
        )
        conn.commit()
        
    # 5. Verify metrics are non-zero (Task 18)
    stats = clean_service.get_health_stats()
    assert stats['storage_used_bytes'] > 0, "Total storage should be > 0"
    assert 'database_bytes' in stats
    assert 'evidence_bytes' in stats
    assert 'recordings_bytes' in stats
    assert 'reference_images_bytes' in stats
    print(f"[METRICS] total={stats['storage_used_bytes']} db={stats['database_bytes']} evidence={stats['evidence_bytes']} recordings={stats['recordings_bytes']} ref={stats['reference_images_bytes']}")
        
    # 6. Run Cleanup with 30-day retention (Task 17)
    cleanup_res = clean_service.run_storage_cleanup(retention_days=30)
    assert cleanup_res['success'], f"Cleanup failed: {cleanup_res}"
    assert cleanup_res['deleted_evidence'] == 1, f"Expected 1 old evidence deleted, got {cleanup_res['deleted_evidence']}"
    assert cleanup_res['deleted_recordings'] == 1, f"Expected 1 old recording deleted, got {cleanup_res['deleted_recordings']}"
    print(f"[CLEANUP] Deleted evidence={cleanup_res['deleted_evidence']}, recordings={cleanup_res['deleted_recordings']}")
    
    # 7. SAFETY BOUNDARY: Verify DELETIONS of expired items
    assert not os.path.exists(old_evidence_path), "Old evidence file must be deleted"
    assert not os.path.exists(old_rec_path), "Old recording file must be deleted"
    
    with im._get_conn() as conn:
        row = conn.execute("SELECT * FROM face_images WHERE image_id = ?", (old_ev_id,)).fetchone()
        assert row is None, "Old evidence DB record must be deleted"
        
    with storage._get_connection() as conn:
        row = conn.execute("SELECT * FROM recordings WHERE id = ?", ("rec_old_bd",)).fetchone()
        assert row is None, "Old recording DB record must be deleted"
    
    print("[PASS] Expired evidence and recordings deleted")

    # 8. SAFETY BOUNDARY: Verify PRESERVED items
    # Fresh evidence NOT deleted
    assert os.path.exists(fresh_evidence_path), "Fresh evidence must be preserved"
    with im._get_conn() as conn:
        row = conn.execute("SELECT * FROM face_images WHERE image_id = ?", (fresh_ev_id,)).fetchone()
        assert row is not None, "Fresh evidence DB record must be preserved"
    
    # Fresh recording NOT deleted
    assert os.path.exists(fresh_rec_path), "Fresh recording must be preserved"
    with storage._get_connection() as conn:
        row = conn.execute("SELECT * FROM recordings WHERE id = ?", ("rec_fresh_bd",)).fetchone()
        assert row is not None, "Fresh recording DB record must be preserved"
        
    # Reference image NEVER deleted (permanent)
    assert os.path.exists(ref_image_path), "Reference image file must be preserved (PERMANENT)"
    ref_images_after = im.get_reference_images(p_id)
    assert len(ref_images_after) == 1, "Reference image DB record must be preserved"
    
    # Person record preserved
    person = im.get_person(p_id)
    assert person is not None, "Person record must be preserved"
    
    # Embeddings preserved
    embeddings = im.get_all_embeddings()
    assert any(e.person_id == p_id for e in embeddings), "Embedding must be preserved"
    
    # FAISS index preserved
    assert wl.index.ntotal == 1, "FAISS index must still contain the embedding"
    
    print("[PASS] Reference images, person, embeddings, FAISS all intact after cleanup")
