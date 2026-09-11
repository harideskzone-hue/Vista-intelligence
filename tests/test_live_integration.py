import os
import sys
import uuid
import pytest
import numpy as np
import cv2
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'face_api'))

# Mock app.core.encoder before importing FaceService to avoid insightface dependency error
import sys as _sys
from unittest.mock import MagicMock
if 'app.core.encoder' not in _sys.modules:
    _sys.modules['app.core.encoder'] = MagicMock()
    _sys.modules['app.core.encoder'].get_encoder = MagicMock()

from face_api.app.services.face_service import FaceService, _get_edge_components

@pytest.fixture
def test_image_bytes():
    # Create a small valid JPEG in memory
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()

@pytest.fixture
def mock_dependencies(tmp_path):
    os.environ["SIH26187_DATA"] = str(tmp_path)
    import face_api.app.services.face_service as fs
    fs._identity_manager = None
    fs._watchlist = None
    fs._temporal_matcher = None
    fs._face_matcher = None
    fs._best_frame_selector = None
    fs._duplicate_filter = None
    
    with patch("face_api.app.services.face_service.get_storage") as mock_get_storage, \
         patch("face_api.app.services.face_service.get_vector_db") as mock_get_vdb, \
         patch("face_api.app.services.face_service.get_encoder") as mock_get_encoder, \
         patch("cv2.imdecode") as mock_imdecode:
         
        mock_storage = MagicMock()
        mock_get_storage.return_value = mock_storage
        
        mock_encoder = MagicMock()
        mock_get_encoder.return_value = mock_encoder
        
        def side_effect_imdecode(buf, flags):
            # Generate a distinct gradient based on buffer length so pHash differs
            val = len(buf.tobytes())
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            for i in range(100):
                img[:, i] = (i * val) % 255
            return img
            
        mock_imdecode.side_effect = side_effect_imdecode
        
        yield mock_storage, mock_encoder

def test_actual_frame_bytes_are_saved(mock_dependencies, test_image_bytes):
    mock_storage, mock_encoder = mock_dependencies
    
    service = FaceService()
    im, wl, tm, fm, bfs, dup = _get_edge_components()
    
    # 1. Enroll Person 1
    person_id = "person_1"
    im.enroll_person(person_id, "Alice", "STAFF")
    
    # 2. Add an embedding so Watchlist can match
    v1 = np.zeros(512, dtype=np.float32)
    v1[0] = 1.0 # normalized
    im.add_embedding(person_id, v1.tobytes(), "enroll_hash", 0.9, "enr")
    wl.rebuild_index()
    
    # Setup mock encoder to return a perfect match
    face_info = {'bbox': [10, 10, 90, 90], 'det_score': 0.95}
    mock_encoder.detect_and_encode.return_value = (v1, face_info)
    
    # Temporal confirmation requires 2 frames by default.
    # Frame 1:
    res1 = service.recognize_live_frame(test_image_bytes, "CAM-01", "track-123")
    assert res1['match_status'] != 'MATCH' # unconfirmed
    
    # Frame 2:
    res2 = service.recognize_live_frame(test_image_bytes, "CAM-01", "track-123")
    assert res2['match_status'] == 'MATCH'
    assert res2['matched_identity_id'] == person_id
    assert res2['action'] == 'saved_evidence'
    
    # VERIFY EXACT BYTES SAVED
    mock_storage.save_image.assert_called_once_with(person_id, test_image_bytes, 0.95)

def test_duplicate_filtering_prevents_storage(mock_dependencies, test_image_bytes):
    mock_storage, mock_encoder = mock_dependencies
    
    service = FaceService()
    im, wl, tm, fm, bfs, dup = _get_edge_components()
    
    person_id = "person_1"
    im.enroll_person(person_id, "Alice", "STAFF")
    v1 = np.zeros(512, dtype=np.float32)
    v1[0] = 1.0 
    im.add_embedding(person_id, v1.tobytes(), "enroll_hash", 0.9, "enr")
    wl.rebuild_index()
    
    face_info = {'bbox': [10, 10, 90, 90], 'det_score': 0.95}
    mock_encoder.detect_and_encode.return_value = (v1, face_info)
    
    # Frame 1 & 2 -> Confirms track -> Saves evidence
    service.recognize_live_frame(test_image_bytes, "CAM-01", "track-123")
    service.recognize_live_frame(test_image_bytes, "CAM-01", "track-123")
    
    assert mock_storage.save_image.call_count == 1
    
    # Frame 3 -> Track continues -> Identical image -> Should be rejected by duplicate filter
    res3 = service.recognize_live_frame(test_image_bytes, "CAM-01", "track-123")
    assert res3['match_status'] == 'MATCH'
    assert res3['action'] == 'ignored_duplicate_or_suboptimal'
    
    assert mock_storage.save_image.call_count == 1

def test_unknown_never_enrolls(mock_dependencies, test_image_bytes):
    mock_storage, mock_encoder = mock_dependencies
    service = FaceService()
    im, wl, tm, fm, bfs, dup = _get_edge_components()
    
    v2 = np.zeros(512, dtype=np.float32)
    v2[1] = 1.0 
    face_info = {'bbox': [10, 10, 90, 90], 'det_score': 0.95}
    mock_encoder.detect_and_encode.return_value = (v2, face_info)
    
    for _ in range(10):
        res = service.recognize_live_frame(test_image_bytes, "CAM-01", "track-999")
        assert res['match_status'] != 'MATCH'
        
    mock_storage.save_image.assert_not_called()
    assert im.get_all_persons() == []

def test_ambiguous_face_is_unknown(mock_dependencies, test_image_bytes):
    mock_storage, mock_encoder = mock_dependencies
    service = FaceService()
    im, wl, tm, fm, bfs, dup = _get_edge_components()
    
    im.enroll_person("person_1", "Alice", "STAFF")
    v1 = np.zeros(512, dtype=np.float32)
    v1[0] = 1.0 
    im.add_embedding("person_1", v1.tobytes(), "h1", 0.9, "enr")
    
    im.enroll_person("person_2", "Bob", "STAFF")
    v2 = np.zeros(512, dtype=np.float32)
    v2[0] = 0.99
    v2[1] = 0.141 
    v2 = v2 / np.linalg.norm(v2)
    im.add_embedding("person_2", v2.tobytes(), "h2", 0.9, "enr")
    
    wl.rebuild_index()
    
    query = (v1 + v2) / 2
    query = query / np.linalg.norm(query)
    
    face_info = {'bbox': [10, 10, 90, 90], 'det_score': 0.95}
    mock_encoder.detect_and_encode.return_value = (query, face_info)
    
    for _ in range(5):
        res = service.recognize_live_frame(test_image_bytes, "CAM-01", "track-ambiguous")
        assert res['match_status'] != 'MATCH'
        
    mock_storage.save_image.assert_not_called()

def test_better_evidence_updates_storage(mock_dependencies):
    mock_storage, mock_encoder = mock_dependencies
    service = FaceService()
    im, wl, tm, fm, bfs, dup = _get_edge_components()
    
    person_id = "person_1"
    im.enroll_person(person_id, "Alice", "STAFF")
    v1 = np.zeros(512, dtype=np.float32)
    v1[0] = 1.0 
    im.add_embedding(person_id, v1.tobytes(), "enroll_hash", 0.9, "enr")
    wl.rebuild_index()
    
    face_info_blurry = {'bbox': [10, 10, 50, 50], 'det_score': 0.70}
    mock_encoder.detect_and_encode.return_value = (v1, face_info_blurry)
    
    blurry_bytes = b"BLURRY_FRAME_123"
    service.recognize_live_frame(blurry_bytes, "CAM-01", "track-1")
    res1 = service.recognize_live_frame(blurry_bytes, "CAM-01", "track-1")
    assert res1['match_status'] == 'MATCH'
    assert res1['action'] == 'saved_evidence'
    mock_storage.save_image.assert_called_with(person_id, blurry_bytes, 0.70)
    
    face_info_sharp = {'bbox': [10, 10, 150, 150], 'det_score': 0.98}
    mock_encoder.detect_and_encode.return_value = (v1, face_info_sharp)
    
    sharp_bytes = b"SHARP_FRAME_456789" # Different length so side_effect_imdecode returns different image
    res2 = service.recognize_live_frame(sharp_bytes, "CAM-01", "track-1")
    assert res2['match_status'] == 'MATCH'
    assert res2['action'] == 'saved_evidence'
    mock_storage.save_image.assert_called_with(person_id, sharp_bytes, 0.98)

@pytest.fixture
def real_storage_dependencies(tmp_path):
    os.environ["SIH26187_DATA"] = str(tmp_path)
    
    with patch("face_api.app.services.face_service.get_encoder") as mock_get_encoder, \
         patch("cv2.imdecode") as mock_imdecode:
         
        import face_api.app.core.storage as storage_mod
        
        # Manually construct Storage to avoid cached imports
        real_storage = storage_mod.Storage.__new__(storage_mod.Storage)
        real_storage.db_path = str(tmp_path / "test.db")
        real_storage.images_dir = str(tmp_path / "images")
        os.makedirs(real_storage.images_dir, exist_ok=True)
        real_storage._init_db()
        
        with patch("face_api.app.services.face_service.get_storage", return_value=real_storage):
            import face_api.app.services.face_service as fs
            fs._identity_manager = None
            fs._watchlist = None
            fs._temporal_matcher = None
            fs._face_matcher = None
            fs._best_frame_selector = None
            fs._duplicate_filter = None
            
            mock_encoder = MagicMock()
            mock_get_encoder.return_value = mock_encoder
            
            def side_effect_imdecode(buf, flags):
                val = len(buf.tobytes())
                img = np.zeros((100, 100, 3), dtype=np.uint8)
                for i in range(100):
                    img[:, i] = (i * val) % 255
                return img
                
            mock_imdecode.side_effect = side_effect_imdecode
            
            yield real_storage, mock_encoder, tmp_path

def test_filesystem_behavior_unknown_vs_match(real_storage_dependencies, test_image_bytes):
    real_storage, mock_encoder, tmp_path = real_storage_dependencies
    
    service = FaceService()
    im, wl, tm, fm, bfs, dup = _get_edge_components()
    
    images_dir = tmp_path / "images"
    assert not os.path.exists(images_dir) or len(os.listdir(images_dir)) == 0
    
    # 1. UNKNOWN case
    v_unknown = np.zeros(512, dtype=np.float32)
    v_unknown[1] = 1.0 
    face_info = {'bbox': [10, 10, 90, 90], 'det_score': 0.95}
    mock_encoder.detect_and_encode.return_value = (v_unknown, face_info)
    
    for _ in range(5):
        res = service.recognize_live_frame(test_image_bytes, "CAM-01", "track-999")
        assert res['match_status'] != 'MATCH'
        
    # Verify no files/folders created
    if os.path.exists(images_dir):
        assert len(os.listdir(images_dir)) == 0, f"UNKNOWN should not create artifacts. Found: {os.listdir(images_dir)}"
        
    # 2. MATCH case
    person_id = "person_real_test"
    im.enroll_person(person_id, "Bob", "STAFF")
    v_match = np.zeros(512, dtype=np.float32)
    v_match[0] = 1.0 
    im.add_embedding(person_id, v_match.tobytes(), "hash_bob", 0.9, "enr")
    wl.rebuild_index()
    
    mock_encoder.detect_and_encode.return_value = (v_match, face_info)
    
    # Frame 1 & 2 to confirm match
    service.recognize_live_frame(test_image_bytes, "CAM-01", "track-bob")
    res_match = service.recognize_live_frame(test_image_bytes, "CAM-01", "track-bob")
    
    assert res_match['match_status'] == 'MATCH'
    assert res_match['matched_identity_id'] == person_id
    
    # Verify EXACTLY one folder exists and it's the person_id
    assert os.path.exists(images_dir)
    items = os.listdir(images_dir)
    assert len(items) == 1
    assert items[0] == person_id
    
    person_dir = images_dir / person_id
    images = os.listdir(person_dir)
    assert len(images) == 1
    assert images[0].endswith(".jpg")
