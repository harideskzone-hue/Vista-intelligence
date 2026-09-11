import os
import uuid
import pytest
import numpy as np
from unittest.mock import Mock

from edge.face.types import FaceDetection, QualityDecision
from edge.face.identity_manager import IdentityManager
from edge.face.watchlist import Watchlist
from edge.face.enrollment import EnrollmentManager, EnrollmentError, EnrollmentDuplicateError

@pytest.fixture
def identity_manager(tmp_path):
    db_path = str(tmp_path / "face.db")
    return IdentityManager(db_path)

@pytest.fixture
def watchlist(identity_manager, tmp_path):
    index_dir = str(tmp_path / "faiss_index")
    return Watchlist(identity_manager, index_dir, dimension=512)

@pytest.fixture
def mock_detector():
    detector = Mock()
    # default returns one face
    d = FaceDetection(
        frame_index=0, x1=10, y1=10, x2=100, y2=100, confidence=0.99, landmarks=None
    )
    detector.detect.return_value = [d]
    return detector

@pytest.fixture
def mock_quality_gate():
    qg = Mock()
    qg.evaluate.return_value = QualityDecision(
        frame_index=0, status="PASS", reason=None, sharpness=100.0,
        face_width_px=90, face_height_px=90, inter_ocular_distance=40.0
    )
    return qg

@pytest.fixture
def mock_embedder():
    embedder = Mock()
    emb = np.zeros(512, dtype=np.float32)
    emb[0] = 1.0
    embedder.embed.return_value = emb
    return embedder

@pytest.fixture
def enrollment_manager(mock_detector, mock_quality_gate, mock_embedder, identity_manager, watchlist):
    return EnrollmentManager(
        detector=mock_detector,
        quality_gate=mock_quality_gate,
        embedder=mock_embedder,
        identity_manager=identity_manager,
        watchlist=watchlist,
        duplicate_threshold=0.85
    )

def test_explicit_enrollment_success(enrollment_manager, identity_manager, watchlist):
    # 1. Explicit enrollment creates exactly one Person
    # 2. Stores expected embedding
    # 3. Stores image metadata
    # 10. Searchable through watchlist
    
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    res = enrollment_manager.enroll("person_001", "Bob", "STAFF", img)
    
    assert res["status"] == "SUCCESS"
    assert res["person_id"] == "person_001"
    
    # DB checks
    persons = identity_manager.get_all_persons()
    assert len(persons) == 1
    assert persons[0].person_id == "person_001"
    
    embeddings = identity_manager.get_all_embeddings()
    assert len(embeddings) == 1
    assert embeddings[0].person_id == "person_001"
    assert embeddings[0].embedding_id == res["embedding_id"]
    
    # We don't have a get_all_images in IdentityManager, but we can query it directly
    with identity_manager._get_conn() as conn:
        imgs = conn.execute("SELECT * FROM face_images").fetchall()
        assert len(imgs) == 1
        assert imgs[0]["person_id"] == "person_001"
        assert imgs[0]["source"] == "ENROLLMENT"
        
    # Check watchlist is updated
    assert watchlist.index.ntotal == 1
    
    # Searchable
    emb_vec = np.frombuffer(embeddings[0].embedding_vector, dtype=np.float32)
    results = watchlist.search(emb_vec, top_k=1)
    assert len(results) == 1
    assert results[0][0] == "person_001"

def test_duplicate_face_rejected(enrollment_manager, mock_embedder):
    # 5. Duplicate face does NOT silently create another Person
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    enrollment_manager.enroll("person_001", "Bob", "STAFF", img)
    
    # Try enrolling again with the same face (embedder returns same vector)
    with pytest.raises(EnrollmentDuplicateError) as exc:
        enrollment_manager.enroll("person_002", "Alice", "VISITOR", img)
    
    assert "POSSIBLE_EXISTING_IDENTITY" in str(exc.value)
    
    # DB remains unchanged
    assert len(enrollment_manager.identity_manager.get_all_persons()) == 1

def test_invalid_or_no_face_rejected(enrollment_manager, mock_detector):
    # 7. Invalid/no face is rejected
    mock_detector.detect.return_value = []
    
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    with pytest.raises(EnrollmentError) as exc:
        enrollment_manager.enroll("person_002", "NoFace", "STAFF", img)
        
    assert "NO_FACE_DETECTED" in str(exc.value)
    assert len(enrollment_manager.identity_manager.get_all_persons()) == 0

def test_poor_quality_face_rejected(enrollment_manager, mock_quality_gate):
    # 6. Poor-quality face is rejected
    mock_quality_gate.evaluate.return_value = QualityDecision(
        frame_index=0, status="QUALITY_FAIL", reason="BLUR", sharpness=5.0,
        face_width_px=90, face_height_px=90, inter_ocular_distance=40.0
    )
    
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    with pytest.raises(EnrollmentError) as exc:
        enrollment_manager.enroll("person_002", "Blurry", "STAFF", img)
        
    assert "QUALITY_FAIL: BLUR" in str(exc.value)
    assert len(enrollment_manager.identity_manager.get_all_persons()) == 0

def test_transaction_rollback_on_failure(enrollment_manager, identity_manager):
    # 8. Failed persistence rolls back
    # Let's force a failure in the DB.
    # Person with same ID already exists, but different embedding
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    enrollment_manager.enroll("person_001", "Bob", "STAFF", img)
    
    # Change embedder to return new face so it passes duplicate check
    v2 = np.zeros(512, dtype=np.float32)
    v2[1] = 1.0
    enrollment_manager.embedder.embed.return_value = v2
    
    # Try enrolling same person_id (causes UNIQUE constraint failure in persons table)
    with pytest.raises(EnrollmentError) as exc:
        enrollment_manager.enroll("person_001", "Bob Duplicate", "STAFF", img)
        
    assert "DATABASE_PERSISTENCE_FAILED" in str(exc.value)
    
    # Assert roll back happened. We should still only have 1 embedding and 1 image.
    assert len(identity_manager.get_all_embeddings()) == 1
    with identity_manager._get_conn() as conn:
        assert len(conn.execute("SELECT * FROM face_images").fetchall()) == 1

def test_faiss_failure_surfaced(enrollment_manager, watchlist):
    # 9. FAISS failure is surfaced safely
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    
    # Break the watchlist intentionally by setting dimension wrong
    watchlist.dimension = 128
    
    with pytest.raises(EnrollmentError) as exc:
        enrollment_manager.enroll("person_002", "Fail FAISS", "STAFF", img)
        
    assert "WATCHLIST_SYNC_FAILED" in str(exc.value)

def test_unknown_face_does_not_auto_create(watchlist, identity_manager):
    # 4. Unknown face does NOT auto-create a Person
    # "Unknown face arrives -> matcher/search -> UNKNOWN -> DB still contains exactly 1 Person"
    
    # Setup initial state
    identity_manager.enroll_person("person_000001", "Existing", "STAFF")
    
    v1 = np.zeros(512, dtype=np.float32)
    v1[0] = 1.0
    identity_manager.add_embedding("person_000001", v1.tobytes(), "v1", 0.9, "enr")
    watchlist.rebuild_index()
    
    assert len(identity_manager.get_all_persons()) == 1
    
    # Simulate an unknown face arriving
    v_unknown = np.zeros(512, dtype=np.float32)
    v_unknown[1] = 1.0 # orthogonal
    
    candidates = watchlist.search(v_unknown, top_k=1)
    
    if candidates and candidates[0][2] >= 0.5:
        match = candidates[0][0]
    else:
        match = "UNKNOWN"
        
    assert match == "UNKNOWN"
    
    # DB MUST remain unchanged
    assert len(identity_manager.get_all_persons()) == 1

def test_bulk_enrollment_success(enrollment_manager, identity_manager, watchlist):
    images = [np.zeros((200, 200, 3), dtype=np.uint8), np.zeros((200, 200, 3), dtype=np.uint8)]
    
    # Mock embedder to return slightly different embeddings so they don't trip duplicate check (or we just allow it since the duplicate check doesn't check against the in-progress list)
    # Actually duplicate check only checks against the *existing* DB. So we can just use the mock embedder as is.
    # Wait, if both return the exact same embedding, the second one might trip the duplicate check if the first was already inserted? 
    # bulk enrollment inserts everything at the END. So duplicate check won't find the first image in the DB.
    
    res = enrollment_manager.enroll_bulk("person_bulk_1", "Bulk Bob", "STAFF", images)
    
    assert res["status"] == "SUCCESS"
    assert res["person_id"] == "person_bulk_1"
    assert res["accepted"] == 2
    assert res["rejected"] == 0
    
    # DB checks
    persons = identity_manager.get_all_persons()
    assert len(persons) == 1
    assert persons[0].person_id == "person_bulk_1"
    
    embeddings = identity_manager.get_all_embeddings()
    assert len(embeddings) == 2
    assert embeddings[0].person_id == "person_bulk_1"
    assert embeddings[1].person_id == "person_bulk_1"
    
    with identity_manager._get_conn() as conn:
        imgs = conn.execute("SELECT * FROM face_images").fetchall()
        assert len(imgs) == 2
        assert imgs[0]["person_id"] == "person_bulk_1"
        assert imgs[1]["person_id"] == "person_bulk_1"
        
    # Check watchlist is updated
    assert watchlist.index.ntotal == 2

def test_bulk_enrollment_partial_failure(enrollment_manager, mock_quality_gate, identity_manager, watchlist):
    images = [np.zeros((200, 200, 3), dtype=np.uint8), np.zeros((200, 200, 3), dtype=np.uint8)]
    
    # Make the quality gate fail for the first image, pass for the second
    from edge.face.types import QualityDecision
    
    pass_decision = QualityDecision(frame_index=0, status="PASS", reason=None, sharpness=100.0, face_width_px=90, face_height_px=90, inter_ocular_distance=40.0)
    fail_decision = QualityDecision(frame_index=0, status="QUALITY_FAIL", reason="BLUR", sharpness=5.0, face_width_px=90, face_height_px=90, inter_ocular_distance=40.0)
    
    mock_quality_gate.evaluate.side_effect = [fail_decision, pass_decision]
    
    res = enrollment_manager.enroll_bulk("person_bulk_2", "Bulk Alice", "STAFF", images)
    
    assert res["status"] == "SUCCESS"
    assert res["accepted"] == 1
    assert res["rejected"] == 1
    assert "QUALITY_FAIL" in res["rejected_details"][0]["reason"]
    
    persons = identity_manager.get_all_persons()
    assert len(persons) == 1
    
    embeddings = identity_manager.get_all_embeddings()
    assert len(embeddings) == 1
    
    assert watchlist.index.ntotal == 1

def test_bulk_enrollment_zero_valid(enrollment_manager, mock_detector, identity_manager, watchlist):
    images = [np.zeros((200, 200, 3), dtype=np.uint8), np.zeros((200, 200, 3), dtype=np.uint8)]
    
    # Make detector fail for all
    mock_detector.detect.return_value = []
    
    with pytest.raises(EnrollmentError) as exc:
        enrollment_manager.enroll_bulk("person_bulk_3", "Bulk Charlie", "STAFF", images)
        
    assert "ZERO_VALID_IMAGES" in str(exc.value)
    
    # DB remains unchanged
    persons = identity_manager.get_all_persons()
    assert len(persons) == 0
    assert watchlist.index.ntotal == 0

